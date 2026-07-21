#!/usr/bin/env python3
"""Diagnose a failed ORCA run and, if it is auto-remediable, resubmit it.

This is the fast per-structure path: the ORCA job's own end-of-run hook calls
``xas-rerun-orca <run_dir>`` the instant a run fails, instead of waiting for the
batch postprocess convergence check (which only fires once every structure in
the batch -- including any multi-day ones -- has finished).

For one run directory it:
  1) diagnoses the failure (:mod:`xas_pipeline.diagnosis`);
  2) if OK -> does nothing;
  3) on any real failure, cancels the dependent CORVUS job (submitted afterok on
     this ORCA run, so it can never run now) -- otherwise it lingers as a
     DependencyNeverSatisfied zombie blocking the afterany batch postprocess;
  4) if the failure is not auto-remediable (charge/mult, post-opt crash, generic
     crash, ...) -> escalates to a human (``resolution=needs_human`` in the state
     file + a ``NEEDS_HUMAN`` line in batch-jobs.log) and stops;
  5) otherwise selects the remedy for the next attempt (:mod:`xas_pipeline.remedy`),
     bounded by a persisted attempt counter (:mod:`xas_pipeline.rerun_state`);
  6) applies the remedy to ``<id>.in`` (:mod:`xas_pipeline.input_remedy`), backing
     up the prior input, bumping the job-script --mem on an OOM remedy;
  7) resubmits the ORCA job and a fresh dependent CORVUS job (afterok), so the
     DAG self-heals -- no lingering DependencyNeverSatisfied CORVUS zombie.

Submission machinery is reused from :mod:`xas_pipeline.orchestrate`.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from xas_pipeline import diagnosis, input_remedy, orchestrate as bp, rerun_state
from xas_pipeline.batch_log import append_outcomes
from xas_pipeline.remedy import MAX_ATTEMPTS, select_remedy

_MEM_RE = re.compile(r"(--mem[=\s])(\d+)([A-Za-z]*)")


def _find_orca_input(run_dir: Path, run_id: str) -> Path | None:
    candidate = run_dir / f"{run_id}.in"
    if candidate.is_file():
        return candidate
    matches = sorted(p for p in run_dir.glob("*.in") if not p.name.startswith("corvus-"))
    return matches[0] if matches else None


def _bump_job_script_mem(base_text: str, mult: float) -> tuple[str, str | None]:
    """Scale the ``#SBATCH --mem=<N>[unit]`` value. Returns (new_text, note)."""
    m = _MEM_RE.search(base_text)
    if not m:
        return base_text, None
    old = int(m.group(2))
    new = int(round(old * mult))
    new_text = base_text[: m.start()] + f"{m.group(1)}{new}{m.group(3)}" + base_text[m.end():]
    return new_text, f"--mem {old}{m.group(3)} -> {new}{m.group(3)}"


def _pristine_copy(history: Path, source: Path) -> Path:
    """Return a pristine backup of ``source``, creating it on first use.

    Remedies are always applied to this pristine text so cards never stack and
    multipliers stay relative to the original (not the previous attempt).
    """
    pristine = history / f"original-{source.name}"
    if not pristine.is_file():
        shutil.copy2(source, pristine)
    return pristine


def _escalate(
    run_dir: Path, run_id: str, state: rerun_state.RerunState, state_file: Path, reason: str
) -> None:
    """Record a terminal 'auto-rerun gave up' outcome in the canonical channels.

    Recorded in two existing places rather than a bespoke sidecar file:
      * ``<id>-rerun-state.json`` gets ``resolution=needs_human`` (structured,
        per-structure, no write contention) -- the authoritative signal, and what
        makes re-triage idempotent;
      * ``batch-jobs.log`` gets a ``NEEDS_HUMAN`` outcome line with the reason, so
        the give-up is visible next to every other outcome. Best-effort: this log
        is now appended from compute nodes, so it is discoverability, not the
        source of truth (the state file is).
    """
    state.resolution = rerun_state.RESOLUTION_NEEDS_HUMAN
    rerun_state.save_state(state_file, state)
    batch_log = run_dir.parent / "batch-jobs.log"
    if batch_log.is_file():
        append_outcomes(batch_log, "auto-rerun escalation", [(f"orca-{run_id}", "NEEDS_HUMAN", reason)])
    print(f"[{run_id}] NEEDS_HUMAN: {reason}")


_CORVUS_JOBID_RE = re.compile(r"job_id=(\S+)")


def _find_stale_corvus_job_id(batch_log_text: str, run_id: str) -> str | None:
    """Most-recent SUBMITTED CORVUS job id for run_id from batch-jobs.log, or None.

    Lines look like ``corvus-xas-<run_id>\\tSUBMITTED\\tjob_id=<jid>`` (and rerun
    variants ``corvus-xas-rerunN-<run_id>``). The last match wins -- that is the
    CORVUS job currently queued afterok on the ORCA that just failed.
    """
    found: str | None = None
    for line in batch_log_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, status = parts[0].strip(), parts[1].strip()
        if not name.startswith("corvus") or not name.endswith(run_id):
            continue
        if status != "SUBMITTED":
            continue
        m = _CORVUS_JOBID_RE.search(parts[2])
        if m:
            found = m.group(1)
    return found


def _cancel_stale_corvus(run_dir: Path, run_id: str, scheduler: str, *, no_submit: bool) -> None:
    """Cancel the dependent CORVUS job of a just-failed ORCA run.

    The CORVUS job was submitted ``afterok`` on this ORCA run, so once ORCA fails
    it can never start and lingers as a DependencyNeverSatisfied zombie (which
    also blocks the ``afterany`` batch postprocess). Cancel it; if we go on to
    resubmit ORCA below, a fresh CORVUS is queued on the new job.
    """
    batch_log = run_dir.parent / "batch-jobs.log"
    if not batch_log.is_file():
        return
    jid = _find_stale_corvus_job_id(batch_log.read_text(encoding="utf-8", errors="replace"), run_id)
    if jid is None:
        return
    print(f"[{run_id}] cancelling stale dependent CORVUS job {jid} (its ORCA failed)")
    if no_submit:
        return
    ok = bp._cancel_job(jid, scheduler)
    append_outcomes(
        batch_log, "auto-rerun corvus-cancel",
        [(f"corvus-{run_id}", "CANCELLED",
          f"dependent ORCA failed; cancelled stale CORVUS job {jid}"
          + ("" if ok else " (cancel command returned nonzero)"))],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a failed ORCA run and auto-resubmit it with SCF/OOM/opt-restart "
            "remedies when the failure is deterministically remediable."
        )
    )
    parser.add_argument("run_dir", type=Path, help="ORCA run directory (the job's submit dir)")
    parser.add_argument(
        "--scheduler",
        choices=sorted(bp.SCHEDULER_SUBMIT_COMMAND),
        default=bp._default_scheduler(),
        help="Scheduler backend (default: $PIPELINE_SCHEDULER or pbs).",
    )
    parser.add_argument(
        "--corvus-mode", choices=["xas"], default="xas",
        help="Dependent CORVUS target to resubmit (default: xas).",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=MAX_ATTEMPTS,
        help=f"Max automatic reruns per structure (default: {MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--no-submit", "--dry-run", dest="no_submit", action="store_true",
        help="Apply the remedy to <id>.in and update state, but do NOT sbatch anything.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"ERROR: run_dir is not a directory: {run_dir}")
    run_id = run_dir.name

    diag = diagnosis.diagnose(run_dir)
    print(f"[{run_id}] diagnosis: {diag.kind.value} -- {diag.reason}")
    if diag.ok:
        print(f"[{run_id}] terminated normally; nothing to do.")
        return 0

    state_file = rerun_state.state_path(run_dir, run_id)
    state = rerun_state.load_state(state_file, run_id)
    if state.is_terminal:
        # Already escalated on an earlier invocation; do not re-record.
        print(f"[{run_id}] already resolved '{state.resolution}'; nothing to do.")
        return 0

    # The ORCA run this batch's CORVUS job depends on (afterok) has failed, so
    # that CORVUS job can never run. Cancel it now regardless of what we do next
    # (resubmit or escalate) so it does not linger as a DependencyNeverSatisfied
    # zombie blocking the afterany postprocess.
    _cancel_stale_corvus(run_dir, run_id, args.scheduler, no_submit=args.no_submit)

    if not diag.auto_remediable:
        _escalate(run_dir, run_id, state, state_file,
                  f"'{diag.kind.value}' is not auto-remediable: {diag.reason}")
        return 0

    attempt = state.next_attempt
    if attempt > args.max_attempts:
        _escalate(run_dir, run_id, state, state_file,
                  f"auto-rerun ladder exhausted after {len(state.attempts)} attempt(s); "
                  f"last failure '{diag.kind.value}': {diag.reason}")
        return 0

    remedy = select_remedy(diag.kind, diag.evidence, attempt)
    if remedy is None:
        _escalate(run_dir, run_id, state, state_file,
                  f"no remedy for '{diag.kind.value}' at attempt {attempt}: {diag.reason}")
        return 0

    input_path = _find_orca_input(run_dir, run_id)
    if input_path is None:
        print(f"[{run_id}] ERROR: no ORCA .in found to remedy.")
        return 1

    # MOREAD only if a usable GBW is present; opt-restart only if a last geometry exists.
    gbw_name = f"{run_id}.gbw" if remedy.use_moread and (run_dir / f"{run_id}.gbw").is_file() else None
    last_geometry_name = None
    if remedy.opt_restart:
        last_geom = run_dir / f"{run_id}.xyz"
        if last_geom.is_file():
            last_geometry_name = last_geom.name
        else:
            print(f"[{run_id}] note: opt-restart requested but {last_geom.name} missing; "
                  "keeping original geometry.")

    # Always remedy FROM the pristine original (kept on first rerun) so remedy
    # cards never stack and %MaxCore/--mem multipliers stay relative to the
    # original. Keep the exact input that produced this failure for provenance.
    # The generated job script references <id>.in by name, so we keep the filename.
    history = run_dir / f"{run_id}-rerun-history"
    history.mkdir(exist_ok=True)
    pristine_in = _pristine_copy(history, input_path)
    backup = history / f"attempt{attempt}-{input_path.name}"
    shutil.copy2(input_path, backup)

    remedied = input_remedy.apply_remedy(
        pristine_in.read_text(encoding="utf-8"), remedy,
        gbw_name=gbw_name, last_geometry_name=last_geometry_name,
    )
    input_path.write_text(remedied, encoding="utf-8")

    mem_note = None
    if remedy.maxcore_mult != 1.0:
        job_script = run_dir / f"generated-{run_id}-orca.script"
        if not job_script.is_file():
            matches = sorted(run_dir.glob("generated-*-orca.script"))
            job_script = matches[0] if matches else job_script
        if job_script.is_file():
            pristine_script = _pristine_copy(history, job_script)
            new_text, mem_note = _bump_job_script_mem(
                pristine_script.read_text(encoding="utf-8"), remedy.maxcore_mult
            )
            job_script.write_text(new_text, encoding="utf-8")

    print(f"[{run_id}] attempt {attempt}/{args.max_attempts}: remedy '{remedy.label}' "
          f"(moread={bool(gbw_name)}, opt_restart={bool(last_geometry_name)}, "
          f"maxcore_x{remedy.maxcore_mult:g}{'' if not mem_note else '; ' + mem_note})")

    batch_root = run_dir.parent
    batch_log = batch_root / "batch-jobs.log"
    orca_job_id = "NO_SUBMIT"
    corvus_job_id = "NO_SUBMIT"

    if not args.no_submit:
        submit_command = bp.SCHEDULER_SUBMIT_COMMAND[args.scheduler]
        bp._check_executable(submit_command)
        orca_script = run_dir / f"generated-{run_id}-orca.script"
        if not orca_script.is_file():
            matches = sorted(run_dir.glob("generated-*-orca.script"))
            if not matches:
                print(f"[{run_id}] ERROR: no generated ORCA job script to resubmit.")
                return 1
            orca_script = matches[0]
        try:
            orca_job_id = bp._submit_job(orca_script, cwd=run_dir, scheduler=args.scheduler)
            bp._append_batch_job_log(
                batch_log, f"orca-rerun{attempt}-{run_id}", "SUBMITTED", job_id=orca_job_id
            )
            print(f"[{run_id}] resubmitted ORCA: {orca_job_id}")
        except Exception:
            bp._append_batch_job_log(batch_log, f"orca-rerun{attempt}-{run_id}", "SUBMIT_FAILED")
            raise

        mode = args.corvus_mode
        wrapper = run_dir / f"generated-{run_id}-corvus-{mode}-wrapper.script"
        bp._write_corvus_wrapper_script(wrapper, run_dir, run_id, args.scheduler, corvus_mode=mode)
        try:
            corvus_job_id = bp._submit_job(
                wrapper, cwd=run_dir, scheduler=args.scheduler, depend_afterok=[orca_job_id]
            )
            bp._append_batch_job_log(
                batch_log, f"corvus-{mode}-rerun{attempt}-{run_id}", "SUBMITTED",
                job_id=corvus_job_id,
            )
            print(f"[{run_id}] resubmitted CORVUS ({mode}, afterok:{orca_job_id}): {corvus_job_id}")
        except Exception:
            bp._append_batch_job_log(batch_log, f"corvus-{mode}-rerun{attempt}-{run_id}", "SUBMIT_FAILED")
            raise

    state.attempts.append(
        rerun_state.Attempt(
            attempt=attempt,
            kind=diag.kind.value,
            remedy=remedy.label,
            utc=bp._utc_now_iso(),
            orca_job_id=orca_job_id,
            input_backup=str(backup),
            note=mem_note,
        )
    )
    rerun_state.save_state(state_file, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
