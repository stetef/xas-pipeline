#!/usr/bin/env python3
"""Run (or submit) the batch postprocess over a batch root.

``xas-run-batch`` submits a postprocess automatically, so this is for the cases
where that is not what you want:

* the batch grew after the fact and you would rather gate the postprocess
  yourself than let the orchestrator replace jobs;
* ``--no-postprocess`` was used;
* the postprocess failed, or you want to re-derive spectra after editing a
  template, without resubmitting any ORCA/CORVUS work.

Two ways to run it::

    xas-postprocess <batch_root>            # here and now, on this node
    xas-postprocess <batch_root> --submit   # as a job, waiting on outstanding CORVUS

``--submit`` looks up the batch's CORVUS jobs in batch-jobs.log, keeps the ones
the scheduler still knows about, and makes the job depend on them (``afterany``),
so it runs once the batch is genuinely finished. With none outstanding it runs
immediately.

The stages are the same ones the orchestrator wires up, in the same order:
orca-check -> process-feff -> cleanup -> auto-rerun-corvus -> download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from xas_pipeline import orchestrate as rbp
from xas_pipeline import scheduler as _sched

# (module, label) in dependency order. cleanup runs after process-feff so the
# deliverables are captured before scratch is pruned, and before download so the
# copies mirror the pruned tree. The CORVUS auto-rerun triage sits between
# cleanup and download: after cleanup so nothing prunes a recompute mid-flight,
# before download so the ids it resubmits are out of corvus-failed-ids.txt before
# the quarantine pass reads it.
STAGES = (
    ("xas_pipeline.stages.orca_check", "check ORCA convergence"),
    ("xas_pipeline.stages.feff_process", "generate spectra"),
    ("xas_pipeline.stages.cleanup", "reclaim disk"),
    ("xas_pipeline.cli.auto_rerun_corvus", "recompute dead XANES legs"),
    ("xas_pipeline.stages.download", "stage results for download"),
)


def _stage_args(
    module: str, batch_root: Path, destination: Path, refresh: bool, scheduler: str
) -> list[str]:
    if module.endswith("orca_check"):
        return [str(batch_root), "--output-dir", str(batch_root)]
    if module.endswith("feff_process"):
        return [str(batch_root), "--recursive"]
    if module.endswith("cleanup"):
        return [str(batch_root), "--execute"]
    if module.endswith("auto_rerun_corvus"):
        # Needs a scheduler: its remedy is to resubmit corvus jobs. The follow-up
        # postprocess it queues inherits this run's download destination.
        return [
            str(batch_root), "--scheduler", scheduler,
            "--download-destination", str(destination),
        ]
    args = [str(batch_root), "-d", str(destination)]
    if refresh:
        args.append("--refresh")
    return args


def run_inline(batch_root: Path, destination: Path, refresh: bool, scheduler: str) -> int:
    """Run every stage here. Returns the first non-zero exit code, else 0."""
    first_failure = 0
    for module, label in STAGES:
        argv = [
            sys.executable, "-m", module,
            *_stage_args(module, batch_root, destination, refresh, scheduler),
        ]
        print(f"\n=== {label} ({module}) ===", flush=True)
        result = subprocess.run(argv)
        if result.returncode != 0:
            print(f"    stage failed with exit code {result.returncode}", file=sys.stderr)
            # Keep going: the later stages are still useful (e.g. download can
            # stage whatever spectra did get generated), and the batch log
            # records the per-run outcomes either way.
            first_failure = first_failure or result.returncode
    return first_failure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path, help="Batch output root to post-process")
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit as a scheduler job depending on the batch's outstanding CORVUS "
            "jobs, instead of running here."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(rbp.SCHEDULER_SUBMIT_COMMAND),
        default=rbp._default_scheduler(),
        help="Scheduler backend (default: PIPELINE_SCHEDULER env, else pbs).",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=Path,
        default=None,
        help="Download destination (default: <batch_root>/downloading-station).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing download copies instead of skipping them.",
    )
    parser.add_argument(
        "--after",
        default=None,
        help=(
            "Comma-separated job ids to depend on, instead of discovering the "
            "batch's outstanding CORVUS jobs. Only meaningful with --submit."
        ),
    )
    parser.add_argument(
        "--no-submit",
        "--dry-run",
        dest="no_submit",
        action="store_true",
        help="With --submit, write the job script but do not submit it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        print(f"Error: not a directory: {batch_root}", file=sys.stderr)
        return 1
    destination = (
        args.destination.expanduser().resolve()
        if args.destination is not None
        else batch_root / "downloading-station"
    )

    if not args.submit:
        return run_inline(batch_root, destination, args.refresh, args.scheduler)

    script = batch_root / f"generated-postprocess-{batch_root.name}.script"
    rbp._write_postprocess_script(
        script,
        args.scheduler,
        batch_root,
        destination,
        skip_extract=False,
        skip_process_feff=False,
        skip_prepare_download=False,
        prepare_download_refresh=args.refresh,
        skip_cleanup=False,
    )

    batch_log = batch_root / "batch-jobs.log"
    if args.after is not None:
        depends = [jid.strip() for jid in args.after.split(",") if jid.strip()]
    else:
        depends = rbp.outstanding_corvus_job_ids(batch_log, args.scheduler)

    if depends:
        print(f"Waiting on {len(depends)} outstanding CORVUS job(s): {', '.join(depends)}")
    else:
        print("No outstanding CORVUS jobs; the postprocess will start immediately.")

    if args.no_submit:
        print(f"Dry run: wrote {script.name}, not submitted.")
        return 0

    rbp._initialize_batch_log(batch_log, args.scheduler)
    try:
        job_id = rbp._submit_job(
            script, cwd=batch_root, scheduler=args.scheduler, depend_afterany=depends
        )
    except Exception:
        rbp._append_batch_job_log(batch_log, f"postprocess-{batch_root.name}", "SUBMIT_FAILED")
        raise

    rbp._append_batch_job_log(
        batch_log, f"postprocess-{batch_root.name}", "SUBMITTED", job_id=job_id
    )
    print(f"Submitted postprocess job {job_id}")
    print(f"  debug: {_sched.get_scheduler(args.scheduler).debug_command(job_id)}")
    return 0


if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
