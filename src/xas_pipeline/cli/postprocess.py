"""xas-postprocess: run, or submit, the batch postprocess over a batch root.

For when the automatic one is not what you want: the batch grew, --no-postprocess
was used, or spectra need re-deriving after a template edit. The stages are the
ones xas-run-batch wires up, in the same order: ORCA check -> FEFF processing ->
cleanup -> CORVUS auto-rerun triage -> download staging.

    xas-postprocess <batch_root>            # here, now
    xas-postprocess <batch_root> --submit   # as a job, after outstanding CORVUS work

``--submit`` waits (afterany) on the batch's CORVUS jobs still queued or running,
and replaces any earlier postprocess still pending.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, orchestrate
from xas_pipeline.scheduler import add_scheduler_argument, get_scheduler


def stage_commands(batch_root: Path, destination: Path, *, refresh: bool, scheduler: str) -> list[list[str]]:
    """``python -m`` argv for each stage, in order."""
    root, dest = str(batch_root), str(destination)
    return [
        ["xas_pipeline.stages.orca_check", root, "--output-dir", root],
        ["xas_pipeline.stages.feff_process", root, "--recursive"],
        ["xas_pipeline.stages.cleanup", root, "--execute"],
        ["xas_pipeline.cli.auto_rerun_corvus", root, "--scheduler", scheduler, "--download-destination", dest],
        ["xas_pipeline.stages.download", root, "-d", dest, *(["--refresh"] if refresh else [])],
    ]


def run_inline(batch_root: Path, destination: Path, *, refresh: bool, scheduler: str) -> int:
    """Run every stage here; a failed stage does not stop the later ones.

    Returns the first non-zero exit code, else 0.
    """
    first_failure = 0
    for module, *args in stage_commands(batch_root, destination, refresh=refresh, scheduler=scheduler):
        print(f"\n=== {module} ===", flush=True)
        code = subprocess.run([sys.executable, "-m", module, *args], check=False).returncode
        if code:
            print(f"    exit code {code}", file=sys.stderr)
            first_failure = first_failure or code
    return first_failure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run, or submit, the batch postprocess.")
    parser.add_argument("batch_root", type=Path, help="The batch root")
    parser.add_argument("--submit", action="store_true", help="Submit as a job instead of running here")
    add_scheduler_argument(parser)
    parser.add_argument("-d", "--destination", type=Path, default=None,
                        help="Download destination (default: <batch_root>/downloading-station)")
    parser.add_argument("--refresh", action="store_true", help="Overwrite existing download copies")
    parser.add_argument("--after", default=None,
                        help="With --submit: comma-separated job ids to wait on, instead of the batch's own")
    parser.add_argument("--no-submit", "--dry-run", dest="no_submit", action="store_true",
                        help="With --submit: write the job script only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        print(f"Error: not a directory: {batch_root}", file=sys.stderr)
        return 1
    destination = (args.destination.expanduser().resolve() if args.destination
                   else batch_root / "downloading-station")
    if not args.submit:
        return run_inline(batch_root, destination, refresh=args.refresh, scheduler=args.scheduler)

    sched = get_scheduler(args.scheduler)
    script = jobscripts.postprocess(sched, batch_root, destination, refresh_download=args.refresh)
    if args.no_submit:
        print(f"Dry run: wrote {script.name}, not submitted.")
        return 0
    batch_log.init(batch_root / batch_log.LOG_NAME, sched)
    if args.after is not None:
        depends = [j.strip() for j in args.after.split(",") if j.strip()]
        job_id = batch_log.submit_and_log(sched, batch_root / batch_log.LOG_NAME,
                                          f"postprocess-{batch_root.name}", script, batch_root, afterany=depends)
    else:
        job_id = orchestrate.submit_postprocess(sched, batch_root, script, [])
    print(f"Submitted postprocess job {job_id}\n  debug: {sched.debug_command(job_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
