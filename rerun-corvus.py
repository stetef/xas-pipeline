#!/usr/bin/env python3
"""Re-run a single CORVUS mode (XANES or EXAFS) on an already-completed batch.

Use this when ORCA optimizations are done and you only want to recompute one
spectroscopy mode -- typically after editing ``corvus-template-xanes.in`` -- without
touching the ORCA artifacts or the other mode's CORVUS results.

For each id directory under ``batch_root`` it:
  1) Resolves the run layout. A post-processed run is split into
     ``<id>/working-<id>/`` (all artifacts) + ``<id>/output-<id>/`` (spectra); a
     not-yet-post-processed run is flat (``<id>/`` holds everything). Either works.
  2) Archives (renames, does not delete) the prior artifacts for the selected mode
     so the recompute is clean and the old spectrum stays available for comparison:
       - working dir: ``Corvus3_cfavg_<mode>/`` -> ``Corvus3_cfavg_<mode>.<tag>/``,
         ``Corvus.cfavg_<mode>.out`` and ``corvus-<id>-<mode>.out`` -> ``.<tag>``
       - output dir: ``<mode>-<id>.dat`` and ``xmu-<mode>-<id>.dat`` are moved into
         ``<id>/<mode>-archive-<tag>/`` (kept OUT of ``output-<id>`` so they are not
         swept into the download copy)
     ORCA artifacts and the other mode's files are never touched.
  3) Regenerates the corvus wrapper and submits it (no ORCA dependency). The wrapper
     re-runs ``prepare-corvus.py --corvus-mode <mode>`` (picking up the edited
     template) and then the corvus job inline.
  4) Unless --no-postprocess, submits one batch postprocess job (afterany on the
     rerun corvus jobs) that re-runs script-process-feff-output.py (refreshing
     ``output-<id>``) and script-prepare-files-for-download.py --refresh (so the new
     spectra overwrite the previous copies in the download station). The ORCA
     convergence/extract stage is skipped (ORCA did not change).

Generation/submission machinery is reused from run-batch-pipeline.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Child directories under the batch root that are not per-id run directories.
NON_ID_DIRS = {
    "failed-orca",
    "failed-corvus",
    "downloading-station",
    "xyz_files",
    "optimized_xyz_files",
    "__pycache__",
}


def _load_batch_pipeline():
    """Import run-batch-pipeline.py (hyphenated filename) as a module for reuse."""
    path = SCRIPT_DIR / "run-batch-pipeline.py"
    if not path.exists():
        raise SystemExit(f"ERROR: run-batch-pipeline.py not found next to {Path(__file__).name}")
    spec = importlib.util.spec_from_file_location("run_batch_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution (the module uses
    # `from __future__ import annotations`) can find the module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bp = _load_batch_pipeline()


@dataclass
class RerunRecord:
    run_id: str
    run_dir: str
    corvus_mode: str
    wrapper_script: str
    corvus_job_id: str
    archived: list[str]


def _resolve_run_dir(id_dir: Path) -> tuple[Path | None, str]:
    """Return (run_dir, layout) for an id directory, or (None, reason) if not runnable.

    layout is 'split' (artifacts in working-<id>/) or 'flat' (artifacts in id_dir).
    """
    run_id = id_dir.name
    split_working = id_dir / f"working-{run_id}"
    if split_working.is_dir() and (split_working / f"{run_id}.hess").is_file():
        return split_working, "split"
    if (id_dir / f"{run_id}.hess").is_file():
        return id_dir, "flat"
    return None, "no <id>.hess found (in id dir or working-<id>/)"


def _archive_target(path: Path, tag: str) -> Path:
    """Return a non-colliding archive path for `path` using suffix `.<tag>`."""
    candidate = path.with_name(f"{path.name}.{tag}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{tag}-{counter}")
        counter += 1
    return candidate


def _archive_mode_artifacts(
    run_dir: Path, output_dir: Path | None, run_id: str, mode: str, tag: str, dry_run: bool
) -> list[str]:
    """Rename the prior artifacts for `mode` aside. Returns the list of moves made."""
    moves: list[tuple[Path, Path]] = []

    # Working-dir CORVUS artifacts for this mode.
    cfavg_dir = run_dir / f"Corvus3_cfavg_{mode}"
    if cfavg_dir.is_dir():
        moves.append((cfavg_dir, _archive_target(cfavg_dir, tag)))
    for fname in (f"Corvus.cfavg_{mode}.out", f"corvus-{run_id}-{mode}.out"):
        src = run_dir / fname
        if src.is_file():
            moves.append((src, _archive_target(src, tag)))

    # Output-dir spectra for this mode -> a sibling archive dir, kept out of output-<id>
    # so the refreshed download copy stays clean.
    if output_dir is not None and output_dir.is_dir():
        spectra = [
            output_dir / f"{mode}-{run_id}.dat",
            output_dir / f"xmu-{mode}-{run_id}.dat",
        ]
        present = [p for p in spectra if p.is_file()]
        if present:
            archive_dir = _archive_target(output_dir.parent / f"{mode}-archive", tag)
            for src in present:
                moves.append((src, archive_dir / src.name))

    made: list[str] = []
    for src, dst in moves:
        made.append(f"{src} -> {dst}")
        print(f"  ARCHIVE: {src.name} -> {dst.relative_to(run_dir.parent) if dst.is_relative_to(run_dir.parent) else dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
    if not made:
        print(f"  (nothing to archive for mode '{mode}')")
    return made


def _iter_id_dirs(batch_root: Path, only_ids: set[str] | None):
    for child in sorted(batch_root.iterdir()):
        if not child.is_dir() or child.name in NON_ID_DIRS:
            continue
        if only_ids is not None and child.name not in only_ids:
            continue
        yield child


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run a single CORVUS mode on a completed batch, archiving the prior "
            "results and refreshing the download station."
        )
    )
    parser.add_argument("batch_root", type=Path, help="Batch output root (parent of the per-id dirs)")
    parser.add_argument(
        "--corvus-mode",
        choices=["xanes", "exafs"],
        default="xanes",
        help="Which single mode to re-run (default: xanes).",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated subset of id directory names to re-run (default: all).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix used for archived prior artifacts (default: rerun-<UTC timestamp>).",
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(bp.SCHEDULER_SUBMIT_COMMAND),
        default=bp._default_scheduler(),
        help="Scheduler backend (default: $PIPELINE_SCHEDULER or pbs).",
    )
    parser.add_argument(
        "--download-destination",
        type=Path,
        default=None,
        help="Download destination (default: <batch_root>/downloading-station).",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Submit only the rerun corvus jobs; skip the batch postprocess/refresh job.",
    )
    parser.add_argument(
        "--no-submit",
        "--dry-run",
        dest="no_submit",
        action="store_true",
        help="Preview only: generate wrappers but do NOT archive or submit anything.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"ERROR: batch_root is not a directory: {batch_root}")

    prepare_corvus_py = SCRIPT_DIR / "prepare-corvus.py"
    if not prepare_corvus_py.exists():
        raise SystemExit("ERROR: prepare-corvus.py not found next to this script")

    submit_command = bp.SCHEDULER_SUBMIT_COMMAND[args.scheduler]
    if not args.no_submit:
        bp._check_executable(submit_command)

    mode = args.corvus_mode
    # Default archive tag: rerun-YYYYMMDDTHHMMSSZ (filename-safe, no ':' '-' '+').
    _stamp = bp._utc_now_iso().split("+", 1)[0].replace(":", "").replace("-", "") + "Z"
    tag = args.tag if args.tag else f"rerun-{_stamp}"
    only_ids = set(s for s in args.ids.split(",") if s) if args.ids else None

    download_destination = (
        args.download_destination.expanduser().resolve()
        if args.download_destination is not None
        else batch_root / "downloading-station"
    )

    batch_log = batch_root / "batch-jobs.log"
    bp._initialize_batch_log(batch_log, args.scheduler)

    records: list[RerunRecord] = []
    skipped: list[str] = []

    for id_dir in _iter_id_dirs(batch_root, only_ids):
        run_id = id_dir.name
        run_dir, layout = _resolve_run_dir(id_dir)
        if run_dir is None:
            print(f"SKIP {run_id}: {layout}")
            skipped.append(f"{run_id}: {layout}")
            continue

        output_dir = id_dir / f"output-{run_id}"
        output_dir = output_dir if output_dir.is_dir() else None

        print(f"\n=== {run_id} (layout={layout}, mode={mode}) ===")
        print(f"  run_dir: {run_dir}")

        # Archive prior artifacts for this mode (only when actually rerunning).
        if args.no_submit:
            print("  (--no-submit: skipping archive; would archive prior "
                  f"Corvus3_cfavg_{mode}/ and {mode} spectra)")
            archived: list[str] = []
        else:
            archived = _archive_mode_artifacts(run_dir, output_dir, run_id, mode, tag, dry_run=False)

        wrapper = run_dir / f"generated-{run_id}-corvus-{mode}-wrapper.script"
        bp._write_corvus_wrapper_script(
            wrapper, run_dir, run_id, prepare_corvus_py, args.scheduler, corvus_mode=mode
        )
        print(f"  wrapper: {wrapper.name}")

        if args.no_submit:
            corvus_job_id = "NO_SUBMIT"
            bp._append_batch_job_log(batch_log, args.scheduler, f"rerun-corvus-{mode}-{run_id}", "SKIPPED")
        else:
            try:
                corvus_job_id = bp._submit_job(wrapper, cwd=run_dir, scheduler=args.scheduler)
                bp._append_batch_job_log(
                    batch_log, args.scheduler, f"rerun-corvus-{mode}-{run_id}", "SUCCEEDED",
                    job_id=corvus_job_id,
                )
                print(f"  submitted: {corvus_job_id}")
            except Exception:
                bp._append_batch_job_log(batch_log, args.scheduler, f"rerun-corvus-{mode}-{run_id}", "FAILED")
                raise

        records.append(
            RerunRecord(
                run_id=run_id,
                run_dir=str(run_dir),
                corvus_mode=mode,
                wrapper_script=str(wrapper),
                corvus_job_id=corvus_job_id,
                archived=archived,
            )
        )

    if not records:
        print("\nNo runnable id directories found; nothing to do.")
        return 1

    # Batch postprocess: refresh output-<id> and overwrite the download copies.
    postprocess_job_id: str | None = None
    if not args.no_postprocess:
        postprocess_script = batch_root / f"generated-rerun-postprocess-{batch_root.name}-{mode}.script"
        bp._write_postprocess_script(
            postprocess_script,
            SCRIPT_DIR,
            args.scheduler,
            batch_root,
            download_destination,
            skip_extract=True,          # ORCA unchanged
            skip_process_feff=False,    # re-derive spectra (refreshes output-<id>)
            skip_prepare_download=False,
            prepare_download_refresh=True,  # overwrite previous download copies
        )
        if args.no_submit:
            postprocess_job_id = "NO_SUBMIT"
            bp._append_batch_job_log(batch_log, args.scheduler, f"rerun-postprocess-{batch_root.name}", "SKIPPED")
        else:
            corvus_ids = [r.corvus_job_id for r in records]
            try:
                # afterok (not afterany): the postprocess re-runs script-process-feff-output.py,
                # which treats a missing <mode> FEFF dir as CORVUS FAILED and quarantines the
                # whole id dir into failed-corvus/. On a rerun we have already archived the prior
                # live spectrum aside, so if the rerun corvus job fails (e.g. prepare-corvus.py
                # aborts) the postprocess must NOT run -- otherwise it would relocate an otherwise
                # healthy run. afterok holds the postprocess unless every rerun corvus job succeeds.
                postprocess_job_id = bp._submit_job(
                    postprocess_script, cwd=batch_root, scheduler=args.scheduler,
                    depend_afterok=corvus_ids,
                )
                bp._append_batch_job_log(
                    batch_log, args.scheduler, f"rerun-postprocess-{batch_root.name}", "SUCCEEDED",
                    job_id=postprocess_job_id,
                )
            except Exception:
                bp._append_batch_job_log(batch_log, args.scheduler, f"rerun-postprocess-{batch_root.name}", "FAILED")
                raise

    state_file = batch_root / f"rerun-state-{batch_root.name}-{mode}-{tag}.json"
    state = {
        "created_utc": bp._utc_now_iso(),
        "batch_root": str(batch_root),
        "corvus_mode": mode,
        "tag": tag,
        "scheduler": args.scheduler,
        "download_destination": str(download_destination),
        "postprocess_job_id": postprocess_job_id,
        "skipped": skipped,
        "runs": [asdict(r) for r in records],
    }
    state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    print(f"\nRe-ran mode '{mode}' for {len(records)} run(s); skipped {len(skipped)}.")
    for r in records:
        print(f"  {r.run_id}: corvus={r.corvus_job_id}")
    print(f"Postprocess job: {postprocess_job_id}")
    print(f"Archive tag: {tag}")
    print(f"State file: {state_file}")
    print(f"Batch job log: {batch_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
