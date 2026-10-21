"""xas-rerun-orca: diagnose a failed ORCA run and resubmit it if a remedy applies.

The ORCA job script calls this the moment a run fails, rather than waiting for
the batch postprocess. For one run dir it:

1. diagnoses the failure (:mod:`xas_pipeline.diagnosis`); a clean run needs nothing;
2. cancels the CORVUS job queued afterok on this run (it can never start now);
3. escalates to a human (state file + ``NEEDS_HUMAN`` in batch-jobs.log) when the
   failure is not auto-remediable or the ladder is spent;
4. otherwise applies the next remedy (:mod:`xas_pipeline.remedy`) to ``<id>.in``
   -- always from the pristine original, kept in ``<id>-rerun-history/`` -- and
   scales the job's memory request on an out-of-memory remedy;
5. resubmits ORCA plus whatever followed it: a fresh CORVUS wrapper, or for a
   ``--pre`` stage, the next stage.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

from xas_pipeline import batch_log, diagnosis, recovery, rerun_state
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.remedy import MAX_ATTEMPTS, apply_remedy, select_remedy
from xas_pipeline.scheduler import add_scheduler_argument, get_scheduler

log = logging.getLogger(__name__)

# The memory request in a job script: Slurm ``--mem=64G``, PBS ``mem=64gb``.
_MEM_RE = re.compile(r"(--mem[=\s]|\bmem=)(\d+)([A-Za-z]*)")


def bump_job_script_mem(text: str, mult: float) -> tuple[str, str | None]:
    """Scale the job script's memory request; returns (text, "old -> new" or None)."""
    match = _MEM_RE.search(text)
    if not match:
        return text, None
    old, unit = int(match.group(2)), match.group(3)
    new = int(round(old * mult))
    return text[:match.start()] + f"{match.group(1)}{new}{unit}" + text[match.end():], f"mem {old}{unit} -> {new}{unit}"


def _pristine(history: Path, source: Path) -> Path:
    """The first-seen copy of *source*: remedies apply to it, so they never stack."""
    pristine = history / f"original-{source.name}"
    if not pristine.is_file():
        shutil.copy2(source, pristine)
    return pristine


def rerun(run_dir: Path, *, scheduler: str, max_attempts: int = MAX_ATTEMPTS, submit: bool = True) -> str:
    """Triage one run dir; returns what happened (``ok``, ``escalated``, ``resubmitted``, ...)."""
    run_id = run_dir.name
    sched = get_scheduler(scheduler)
    diag = diagnosis.diagnose(run_dir)
    log.info("[%s] diagnosis: %s -- %s", run_id, diag.kind.value, diag.reason)
    if diag.ok:
        return "ok"

    state_file = rerun_state.state_path(run_dir, run_id)
    state = rerun_state.load_state(state_file, run_id)
    if state.is_terminal:
        log.info("[%s] already resolved '%s'; nothing to do.", run_id, state.resolution)
        return "terminal"

    attempt = state.next_attempt
    remedy = select_remedy(diag.kind, diag.evidence, attempt, max_attempts) if diag.auto_remediable else None
    input_path = run_dir / f"{run_id}.in"
    if not submit:
        log.info("[%s] preview: would cancel the stale CORVUS job and %s", run_id,
                 f"apply '{remedy.label}' (attempt {attempt})" if remedy else "escalate")
        return "preview"

    root = batch_log.find_batch_root(run_dir)
    log_path = root / batch_log.LOG_NAME if root else None
    recovery.cancel_stale_corvus(sched, log_path, run_id)

    if remedy is None or not input_path.is_file():
        if not diag.auto_remediable:
            reason = f"'{diag.kind.value}' is not auto-remediable: {diag.reason}"
        elif not input_path.is_file():
            reason = f"no {input_path.name} to remedy"
        else:
            reason = (f"no remedy for '{diag.kind.value}' at attempt {attempt} "
                      f"({len(state.attempts)} so far): {diag.reason}")
        recovery.escalate(state_file, state, log_path, f"orca-{run_id}", reason, "auto-rerun escalation")
        return "escalated"
    if log_path is None:
        log.error("[%s] no batch-jobs.log above %s; cannot resubmit safely.", run_id, run_dir)
        return "error"

    history = run_dir / f"{run_id}-rerun-history"
    history.mkdir(exist_ok=True)
    backup = history / f"attempt{attempt}-{input_path.name}"
    shutil.copy2(input_path, backup)

    # The guess must not be the .gbw this job is about to write (<id>.gbw), so
    # the previous orbitals are read from a copy (staged to scratch with *.gbw).
    gbw_name = None
    if remedy.use_moread and (run_dir / f"{run_id}.gbw").is_file():
        gbw_name = f"{run_id}-moread.gbw"
        shutil.copy2(run_dir / f"{run_id}.gbw", run_dir / gbw_name)
    last_geometry = None
    if remedy.opt_restart:
        if (run_dir / f"{run_id}.xyz").is_file():
            # A full path: a --pre stage's *xyzfile points into another dir.
            last_geometry = str(run_dir / f"{run_id}.xyz")
        else:
            log.info("[%s] opt restart requested but %s.xyz is missing; keeping the geometry.", run_id, run_id)
    input_path.write_text(
        apply_remedy(_pristine(history, input_path).read_text(encoding="utf-8"), remedy,
                     gbw_name=gbw_name, last_geometry_name=last_geometry),
        encoding="utf-8",
    )

    mem_note = None
    job_script = recovery.find_script(run_dir, run_id, "orca")
    if remedy.maxcore_mult != 1.0 and job_script is not None:
        text, mem_note = bump_job_script_mem(_pristine(history, job_script).read_text(encoding="utf-8"),
                                             remedy.maxcore_mult)
        job_script.write_text(text, encoding="utf-8")
    log.info("[%s] attempt %d/%d: %s (moread=%s, opt_restart=%s%s)", run_id, attempt, max_attempts,
             remedy.label, bool(gbw_name), bool(last_geometry), f"; {mem_note}" if mem_note else "")

    attempt_record = rerun_state.Attempt(
        attempt=attempt, kind=diag.kind.value, remedy=remedy.label, utc=batch_log.utc_now(),
        input_backup=str(backup), note=mem_note,
    )
    try:
        attempt_record.orca_job_id, attempt_record.corvus_job_id = recovery.resubmit_orca(
            sched, run_dir, run_id, log_path, label=f"rerun{attempt}"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        state.attempts.append(attempt_record)
        recovery.escalate(state_file, state, log_path, f"orca-{run_id}",
                          f"resubmission failed at attempt {attempt}: {exc}", "auto-rerun escalation")
        return "escalated"
    state.attempts.append(attempt_record)
    rerun_state.save_state(state_file, state)
    return "resubmitted"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a failed ORCA run and resubmit it when the failure is auto-remediable."
    )
    parser.add_argument("run_dir", type=Path, help="The ORCA run dir (the job's submit dir)")
    add_scheduler_argument(parser)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help="Automatic reruns per structure, at most the ladder length (default: %(default)s)")
    parser.add_argument("--no-submit", "--dry-run", dest="no_submit", action="store_true",
                        help="Report what would happen; change nothing")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {run_dir}")
    outcome = rerun(run_dir, scheduler=args.scheduler, max_attempts=args.max_attempts, submit=not args.no_submit)
    return 1 if outcome == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
