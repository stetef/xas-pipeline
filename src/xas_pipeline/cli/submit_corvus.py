#!/usr/bin/env python3
"""Submit ONLY the CORVUS stage for a batch whose ORCA runs are already complete.

run-batch-pipeline.py always resubmits ORCA (no skip-on-existing), so this helper
covers the common "ORCA is done, now run the default XANES + EXAFS CORVUS jobs"
case. For each run directory that already contains its <ID>.hess, it regenerates
the corvus wrapper script (per mode) from slurm-scripts/corvus-wrapper.script using
the pipeline's own _write_corvus_wrapper_script (so templating cannot drift), then
sbatch-es it with NO ORCA dependency. The wrapper runs prepare-corvus.py inside the
job and then executes the generated corvus-job-<mode>.script inline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Reuse the pipeline's tested wrapper-templating + job-id parsing helpers.
from xas_pipeline import orchestrate as rbp


def _discover_run_dirs(batch_dir: Path) -> list[Path]:
    run_dirs = []
    for child in sorted(batch_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / f"{child.name}.hess").is_file():
            run_dirs.append(child)
    return run_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="Batch dir containing per-ID run dirs")
    parser.add_argument(
        "--scheduler",
        choices=sorted(rbp.SCHEDULER_SUBMIT_COMMAND),
        default=rbp._default_scheduler(),
        help="Scheduler backend (default: PIPELINE_SCHEDULER env, else pbs).",
    )
    parser.add_argument(
        "--corvus-mode",
        choices=["both", "exafs", "xanes"],
        default="both",
        help="CORVUS mode(s) to run: 'both' (default), 'exafs', or 'xanes'.",
    )
    parser.add_argument(
        "--no-submit",
        "--dry-run",
        dest="no_submit",
        action="store_true",
        help="Generate wrapper scripts only; do not sbatch.",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir.expanduser().resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"Not a directory: {batch_dir}")

    run_dirs = _discover_run_dirs(batch_dir)
    if not run_dirs:
        raise SystemExit(f"No run dirs with <ID>.hess found under {batch_dir}")

    modes = ["exafs", "xanes"] if args.corvus_mode == "both" else [args.corvus_mode]
    submit_command = rbp.SCHEDULER_SUBMIT_COMMAND[args.scheduler]

    print(f"Batch: {batch_dir}")
    print(f"Run dirs with .hess: {len(run_dirs)}  |  modes: {modes}  |  scheduler: {args.scheduler}")

    total = 0
    for run_dir in run_dirs:
        run_id = run_dir.name
        for mode in modes:
            wrapper = run_dir / f"generated-{run_id}-corvus-{mode}-wrapper.script"
            rbp._write_corvus_wrapper_script(
                wrapper,
                run_dir,
                run_id,
                args.scheduler,
                corvus_mode=mode,
            )
            if args.no_submit:
                print(f"  [dry-run] {run_id} ({mode}) -> {wrapper.name}")
                continue
            result = subprocess.run(
                [submit_command, wrapper.name],
                cwd=run_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  FAILED {run_id} ({mode}): {result.stderr.strip()}", file=sys.stderr)
                continue
            job_id = rbp._parse_submitted_job_id(result.stdout)
            print(f"  submitted {run_id} ({mode}) -> job {job_id}")
            total += 1

    if args.no_submit:
        print(f"Dry run complete: generated {len(run_dirs) * len(modes)} wrapper script(s), none submitted.")
    else:
        print(f"Submitted {total} CORVUS job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
