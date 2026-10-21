"""What the automatic recovery paths share: finding a run's jobs and scripts,
resubmitting them without breaking the batch's dependency graph, archiving
CORVUS output, and escalating to a human.

Used by xas-rerun-orca (failed ORCA), the watchdog (hung jobs), xas-rerun-corvus
and its automatic triage (dead XANES legs).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import batch_log, jobscripts, rerun_state
from .scheduler import Scheduler

log = logging.getLogger(__name__)

# Job-name prefixes, longest first, ahead of the run id in batch-jobs.log.
_JOB_PREFIXES = ("submit-corvus-xas-", "rerun-corvus-xas-", "corvus-xas-", "orca-")
_RERUN_DECORATION = re.compile(r"^(?:stage-)?(?:hang-)?rerun\d+-")


def run_id_from_job_name(job_name: str) -> str | None:
    """The run id in a pipeline job name, or None for other jobs.

    ``orca-<id>``, ``orca-rerun2-<id>``, ``orca-hang-rerun1-<id>``,
    ``corvus-xas-<id>``, ``corvus-xas-hang-rerun1-<id>``,
    ``submit-corvus-xas-<id>``, ``rerun-corvus-xas-<id>``, ...
    """
    for prefix in _JOB_PREFIXES:
        if job_name.startswith(prefix):
            return _RERUN_DECORATION.sub("", job_name[len(prefix):]) or None
    return None


def is_corvus_job(job_name: str) -> bool:
    return job_name.startswith(batch_log.CORVUS_JOB_PREFIXES)


def latest_corvus_job_id(log_path: Path, run_id: str) -> str | None:
    """The most recently submitted CORVUS job of *run_id* (exact run-id match)."""
    latest = None
    for name, status, job_id in batch_log.records(log_path):
        if status == "SUBMITTED" and job_id and is_corvus_job(name) and run_id_from_job_name(name) == run_id:
            latest = job_id
    return latest


def find_script(run_dir: Path, run_id: str, kind: str) -> Path | None:
    """A run's generated ``orca`` job script or ``corvus`` wrapper, if present."""
    exact, pattern = {
        "orca": (f"generated-{run_id}-orca.script", "generated-*-orca.script"),
        "corvus": (f"generated-{run_id}-corvus-xas-wrapper.script", "generated-*-corvus-*-wrapper.script"),
    }[kind]
    if (run_dir / exact).is_file():
        return run_dir / exact
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def escalate(state_file: Path, state: rerun_state.RerunState, log_path: Path | None,
             job_label: str, reason: str, section: str) -> None:
    """Stop a ladder: ``resolution=needs_human`` in its state file (authoritative),
    plus a ``NEEDS_HUMAN`` outcome line in batch-jobs.log (for discoverability)."""
    state.resolution = rerun_state.RESOLUTION_NEEDS_HUMAN
    rerun_state.save_state(state_file, state)
    if log_path is not None:
        batch_log.append_outcomes(log_path, section, [(job_label, "NEEDS_HUMAN", reason)])
    log.warning("[%s] NEEDS_HUMAN: %s", state.run_id, reason)


def cancel_stale_corvus(sched: Scheduler, log_path: Path | None, run_id: str) -> str | None:
    """Cancel the CORVUS job queued afterok on a failed or hung ORCA run.

    It can never start now, and as a DependencyNeverSatisfied zombie it would
    hold the batch's afterany postprocess forever. Returns the id cancelled.
    """
    if log_path is None:
        return None
    job_id = latest_corvus_job_id(log_path, run_id)
    if job_id is None:
        return None
    ok = sched.cancel(job_id)
    batch_log.append_outcomes(log_path, "auto-rerun corvus-cancel", [(
        f"corvus-{run_id}", "CANCELLED",
        f"its ORCA job failed; cancelled stale CORVUS job {job_id}" + ("" if ok else " (cancel failed)"),
    )])
    log.info("[%s] cancelled stale dependent CORVUS job %s%s", run_id, job_id, "" if ok else " (cancel failed)")
    return job_id


def next_stage_record(run_dir: Path, run_id: str) -> dict | None:
    """The ``<run_id>-next-stage.json`` sidecar of a ``--pre`` stage, if usable."""
    path = run_dir / f"{run_id}-next-stage.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("[%s] ignoring unreadable next-stage sidecar: %s", run_id, exc)
        return None
    return data if isinstance(data, dict) and data.get("next_job_script") else None


def resubmit_orca(
    sched: Scheduler,
    run_dir: Path,
    run_id: str,
    log_path: Path,
    *,
    label: str,
    extra: tuple[str, ...] = (),
) -> tuple[str, str | None]:
    """Resubmit a run's ORCA job and re-attach what followed it.

    A ``--pre`` stage gets its next stage re-chained (the scheduler killed it
    when this stage failed); a final stage gets a fresh CORVUS wrapper afterok.
    Job names are ``orca-<label>-<id>`` and ``corvus-xas-<label>-<id>`` (or
    ``orca-stage-<label>-<next id>``). Returns ``(orca job id, follower job id)``;
    raises RuntimeError/ValueError if a submission is refused (after logging it).
    """
    script = find_script(run_dir, run_id, "orca")
    if script is None:
        raise RuntimeError(f"no generated ORCA job script in {run_dir}")
    orca_id = batch_log.submit_and_log(sched, log_path, f"orca-{label}-{run_id}", script, run_dir, extra=extra)
    log.info("[%s] resubmitted ORCA: %s%s", run_id, orca_id, f" ({' '.join(extra)})" if extra else "")

    record = next_stage_record(run_dir, run_id)
    if record is not None:
        next_dir = Path(record["next_run_dir"])
        next_id = record.get("next_run_id") or next_dir.name
        next_script = next_dir / record["next_job_script"]
        if not next_script.is_file():
            log.warning("[%s] next stage script missing, not re-chained: %s", run_id, next_script)
            return orca_id, None
        if record.get("next_job_id"):
            sched.cancel(str(record["next_job_id"]))
        follower = batch_log.submit_and_log(
            sched, log_path, f"orca-stage-{label}-{next_id}", next_script, next_dir, afterok=[orca_id]
        )
        record["next_job_id"] = follower
        (run_dir / f"{run_id}-next-stage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log.info("[%s] re-chained next stage %s: %s (afterok %s)", run_id, next_id, follower, orca_id)
        return orca_id, follower

    wrapper = jobscripts.corvus_wrapper(sched, run_dir, run_id)
    follower = batch_log.submit_and_log(
        sched, log_path, f"corvus-xas-{label}-{run_id}", wrapper, run_dir, afterok=[orca_id]
    )
    log.info("[%s] resubmitted CORVUS: %s (afterok %s)", run_id, follower, orca_id)
    return orca_id, follower


def _archive_target(path: Path, tag: str) -> Path:
    candidate = path.with_name(f"{path.name}.{tag}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{tag}-{counter}")
        counter += 1
    return candidate


def archive_corvus_outputs(run_dir: Path, output_dir: Path | None, run_id: str, tag: str,
                           *, dry_run: bool = False) -> list[str]:
    """Rename a run's CORVUS outputs aside so a recompute starts clean.

    ``Corvus3_cfavg_xas/``, ``Corvus.cfavg_xas.out`` and ``corvus-<id>-xas.out``
    get a ``.<tag>`` suffix; ``xas-<id>.dat`` / ``xmu-xas-<id>.dat`` move out of
    ``output-<id>/`` into ``xas-archive.<tag>/`` beside it. Tags start with
    ``rerun-``, so the postprocess cleanup deletes these snapshots (unless run
    with --skip-cleanup). Returns ``"src -> dst"`` for each move.
    """
    moves = []
    cfavg = run_dir / "Corvus3_cfavg_xas"
    if cfavg.is_dir():
        moves.append((cfavg, _archive_target(cfavg, tag)))
    for name in ("Corvus.cfavg_xas.out", f"corvus-{run_id}-xas.out"):
        if (run_dir / name).is_file():
            moves.append((run_dir / name, _archive_target(run_dir / name, tag)))
    if output_dir is not None and output_dir.is_dir():
        spectra = [p for p in (output_dir / f"xas-{run_id}.dat", output_dir / f"xmu-xas-{run_id}.dat") if p.is_file()]
        if spectra:
            archive_dir = _archive_target(output_dir.parent / "xas-archive", tag)
            moves += [(src, archive_dir / src.name) for src in spectra]
    for src, dst in moves:
        log.info("  archive: %s -> %s", src.name, dst)
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
    return [f"{src} -> {dst}" for src, dst in moves]
