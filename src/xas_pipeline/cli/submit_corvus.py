#!/usr/bin/env python3
"""Submit ONLY the CORVUS stage for a batch whose ORCA runs are already complete.

run-batch-pipeline.py always resubmits ORCA (no skip-on-existing), so this helper
covers the common "ORCA is done, now run the combined XAS CORVUS job" case. For
each eligible run directory it regenerates the corvus wrapper script from
slurm-scripts/corvus-wrapper.script using the pipeline's own
_write_corvus_wrapper_script (so templating cannot drift), then sbatch-es it with
NO ORCA dependency. The wrapper runs prepare-corvus.py inside the job and then
executes the generated corvus-job-xas.script inline.

Eligible means the run dir already holds its <ID>.hess -- or is an ``interp``
run, whose Hessian does not exist yet because the wrapper itself builds it from
the ligand spring models. Run dirs are found via layout.iter_id_dirs, so both
grouped (``<id>/<id>-<mode>/``) and pre-grouping flat batches work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Reuse the pipeline's tested wrapper-templating + job-id parsing helpers.
from xas_pipeline import layout, orchestrate as rbp
from xas_pipeline.stages.orca_prep import SPRING_HESSIAN_MODES


def _discover_run_dirs(batch_dir: Path, only_ids: set[str] | None = None) -> list[Path]:
    """Run dirs whose CORVUS stage can be submitted now.

    An interp run legitimately has no .hess at this point: its ORCA step ran no
    AnFreq, and the wrapper interpolates the Hessian just before prepare-corvus.
    Requiring the file here would silently skip exactly those runs.

    ``only_ids`` restricts to named runs (or every mode run of a named structure),
    which is what makes it safe to submit one set of runs while another set in the
    same batch root is still going.
    """
    run_dirs = []
    for run_dir in layout.iter_id_dirs(batch_dir, only_ids=only_ids):
        if (run_dir / f"{run_dir.name}.hess").is_file():
            run_dirs.append(run_dir)
        elif layout.mode_from_run_id(run_dir.name) in SPRING_HESSIAN_MODES:
            run_dirs.append(run_dir)
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
        choices=["xas"],
        default="xas",
        help="CORVUS target to run. Only the combined 'xas' target is supported.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "Comma-separated run ids (or structure names, which select every mode "
            "run for that structure) to submit. Default: every eligible run dir. "
            "Use this to submit one set while another set in the same batch root "
            "is still running."
        ),
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

    only_ids = (
        {token.strip() for token in args.ids.split(",") if token.strip()}
        if args.ids
        else None
    )
    run_dirs = _discover_run_dirs(batch_dir, only_ids)
    if not run_dirs:
        raise SystemExit(
            f"No submittable run dirs found under {batch_dir} "
            "(need <ID>.hess, or an interp run dir whose wrapper will build it)"
        )

    modes = [args.corvus_mode]
    submit_command = rbp.SCHEDULER_SUBMIT_COMMAND[args.scheduler]

    # Record submissions in the batch log (fix #2: this entry point used to be
    # silent). Same SUBMITTED / SUBMIT_FAILED / SKIPPED vocabulary as run-batch.
    batch_log = batch_dir / "batch-jobs.log"
    rbp._initialize_batch_log(batch_log, args.scheduler)

    print(f"Batch: {batch_dir}")
    print(f"Submittable run dirs: {len(run_dirs)}  |  modes: {modes}  |  scheduler: {args.scheduler}")

    total = 0
    for run_dir in run_dirs:
        run_id = run_dir.name
        for mode in modes:
            job_name = f"submit-corvus-{mode}-{run_id}"
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
                rbp._append_batch_job_log(batch_log, job_name, "SKIPPED")
                continue
            result = subprocess.run(
                [submit_command, wrapper.name],
                cwd=run_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  FAILED {run_id} ({mode}): {result.stderr.strip()}", file=sys.stderr)
                rbp._append_batch_job_log(batch_log, job_name, "SUBMIT_FAILED")
                continue
            job_id = rbp._parse_submitted_job_id(result.stdout)
            print(f"  submitted {run_id} ({mode}) -> job {job_id}")
            rbp._append_batch_job_log(batch_log, job_name, "SUBMITTED", job_id=job_id)
            total += 1

    if args.no_submit:
        print(f"Dry run complete: generated {len(run_dirs) * len(modes)} wrapper script(s), none submitted.")
    else:
        print(f"Submitted {total} CORVUS job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
