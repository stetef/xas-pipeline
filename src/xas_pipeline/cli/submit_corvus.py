"""xas-submit-corvus: submit only the CORVUS stage of a batch whose ORCA runs are done.

xas-run-batch always (re)submits ORCA, so this covers "ORCA finished, now run
CORVUS". Each eligible run gets a fresh CORVUS wrapper, submitted with no
dependency. Eligible: the run dir holds its ``<id>.hess``, or its mode builds
the Hessian from spring models inside the wrapper.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, layout, modes
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import add_scheduler_argument, get_scheduler

log = logging.getLogger(__name__)


def submittable_run_dirs(batch_root: Path, only_ids: set[str] | None = None) -> list[Path]:
    """Run dirs whose CORVUS stage can run now (see the module docstring)."""
    return [
        run_dir for run_dir in layout.iter_id_dirs(batch_root, only_ids=only_ids)
        if (run_dir / f"{run_dir.name}.hess").is_file()
        or layout.mode_from_run_id(run_dir.name) in modes.SPRING_HESSIAN_MODES
    ]


def submit_corvus(batch_root: Path, *, scheduler: str, only_ids: set[str] | None = None,
                  submit: bool = True) -> dict[str, str | None]:
    """Write and (with *submit*) submit each eligible run's CORVUS wrapper.

    Returns ``{run id: job id}``; None marks a refused submission.
    """
    sched = get_scheduler(scheduler)
    log_path = batch_root / batch_log.LOG_NAME
    batch_log.init(log_path, sched)
    jobs: dict[str, str | None] = {}
    for run_dir in submittable_run_dirs(batch_root, only_ids):
        run_id = run_dir.name
        wrapper = jobscripts.corvus_wrapper(sched, run_dir, run_id)
        name = f"submit-corvus-xas-{run_id}"
        if not submit:
            batch_log.append(log_path, name, "SKIPPED")
            jobs[run_id] = None
            log.info("  [dry-run] %s -> %s", run_id, wrapper.name)
            continue
        try:
            jobs[run_id] = batch_log.submit_and_log(sched, log_path, name, wrapper, run_dir)
            log.info("  submitted %s -> job %s", run_id, jobs[run_id])
        except (RuntimeError, ValueError, OSError) as exc:
            jobs[run_id] = None
            log.error("  FAILED %s: %s", run_id, exc)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit only the CORVUS stage for runs whose ORCA is done.")
    parser.add_argument("batch_dir", type=Path, help="The batch root")
    add_scheduler_argument(parser)
    parser.add_argument("--ids", default=None,
                        help="Comma-separated run ids or structure names (default: every eligible run)")
    parser.add_argument("--no-submit", "--dry-run", dest="no_submit", action="store_true",
                        help="Write the wrappers only")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    args = parser.parse_args()
    setup_cli_logging()

    batch_root = args.batch_dir.expanduser().resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"Not a directory: {batch_root}")
    only_ids = {t.strip() for t in args.ids.split(",") if t.strip()} if args.ids else None
    if not submittable_run_dirs(batch_root, only_ids):
        raise SystemExit(f"No submittable run dirs under {batch_root} (need <id>.hess, or a spring mode)")
    jobs = submit_corvus(batch_root, scheduler=args.scheduler, only_ids=only_ids, submit=not args.no_submit)
    failed = [run_id for run_id, job in jobs.items() if job is None and not args.no_submit]
    print(f"{'Wrote' if args.no_submit else 'Submitted'} {len(jobs) - len(failed)} CORVUS job(s)"
          + (f"; {len(failed)} failed: {', '.join(failed)}" if failed else ""), file=sys.stderr if failed else sys.stdout)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
