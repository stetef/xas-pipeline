#!/usr/bin/env python3
"""Collect surviving jobs' output dirs for download and quarantine failed CORVUS runs.

For each id directory under ``parent_dir``:
  - If the id is listed in ``corvus-failed-ids.txt`` (written by the FEFF
    post-processing stage), the whole id directory is MOVED into a
    ``failed-corvus/`` directory under ``parent_dir`` (the batch root).
  - Otherwise its ``output-*`` directory is copied into the download destination
    (default: ``./downloading-station`` in the current working directory).

Ids already pulled out by the ORCA convergence stage (failed-orca/) are never seen
here, so the download destination ends up holding only the passing/surviving jobs.

Example:
    python script-prepare-files-for-download.py /path/to/parent -d ./downloading-station
"""

import argparse
import shutil
import sys
from pathlib import Path

from xas_pipeline import layout

FAILED_CORVUS_MANIFEST = "corvus-failed-ids.txt"


def read_failed_corvus_ids(parent_dir: Path) -> set:
    """Read the set of CORVUS-failed ids written by script-process-feff-output.py."""
    manifest = parent_dir / FAILED_CORVUS_MANIFEST
    if not manifest.is_file():
        return set()
    return {
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def iter_id_dirs(parent_dir: Path):
    """Yield first-level id directories under parent_dir, skipping helper dirs."""
    return layout.iter_id_dirs(parent_dir)


def move_to_failed_corvus(id_dir: Path, failed_corvus_dir: Path, dry_run: bool) -> bool:
    failed_corvus_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_corvus_dir / id_dir.name
    print(f"FAILED-CORVUS: {id_dir} -> {destination}")
    if dry_run:
        return True
    layout.quarantine_move(id_dir, failed_corvus_dir)  # mkdir idempotent
    return True


def download_subpath(id_dir: Path, parent_dir: Path) -> Path:
    """Where an id's outputs land under the download destination.

    Mirrors the batch tree, so a structure's per-mode runs stay grouped in the
    download station exactly as they are on disk (``<id>/<id>-<mode>/``), and a
    pre-grouping flat run dir still lands at ``<id>/``.
    """
    try:
        return id_dir.relative_to(parent_dir)
    except ValueError:
        return Path(id_dir.name)


def copy_output_dirs(
    id_dir: Path,
    destination_dir: Path,
    dry_run: bool,
    refresh: bool = False,
    parent_dir: Path | None = None,
) -> tuple[int, int]:
    """Copy an id's output-* directory(ies) into destination_dir/<subpath>.

    When ``refresh`` is True, an existing target is overwritten (the destination's
    files are replaced with the current output-* contents) instead of skipped. This
    is what lets a rerun propagate freshly recomputed spectra into a download
    destination that already holds the previous run's outputs.
    """
    copied = 0
    skipped = 0
    subpath = download_subpath(id_dir, parent_dir) if parent_dir else Path(id_dir.name)
    output_dirs = [p for p in sorted(id_dir.glob("output*")) if p.is_dir()]
    if not output_dirs:
        print(f"warning: no output* directory in {id_dir}; nothing to copy")
        return copied, skipped

    for output_dir in output_dirs:
        target_dir = destination_dir / subpath
        if target_dir.exists():
            if not refresh:
                print(f"SKIP (exists): {target_dir}")
                skipped += 1
                continue
            print(f"REFRESH (overwrite existing): {output_dir} -> {target_dir}")
            if not dry_run:
                shutil.rmtree(target_dir)
                shutil.copytree(output_dir, target_dir)
            copied += 1
            continue
        print(f"COPY: {output_dir} -> {target_dir}")
        if not dry_run:
            shutil.copytree(output_dir, target_dir)
        copied += 1
    return copied, skipped


def prepare_downloads(
    parent_dir: Path,
    destination_dir: Path,
    failed_corvus_dir: Path,
    dry_run: bool = False,
    refresh: bool = False,
) -> tuple[int, int, int]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = read_failed_corvus_ids(parent_dir)

    copied = 0
    skipped = 0
    quarantined = 0

    for id_dir in iter_id_dirs(parent_dir):
        if id_dir.name in failed_ids:
            move_to_failed_corvus(id_dir, failed_corvus_dir, dry_run)
            quarantined += 1
            continue

        dir_copied, dir_skipped = copy_output_dirs(
            id_dir, destination_dir, dry_run, refresh, parent_dir=parent_dir
        )
        copied += dir_copied
        skipped += dir_skipped

    return copied, skipped, quarantined


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy surviving jobs' output* directories to a download destination and "
            "move CORVUS-failed ids into failed-corvus/."
        )
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing id directories to scan.",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=Path,
        default=Path("downloading-station"),
        help=(
            "Destination directory for copied output directories "
            "(default: ./downloading-station in the current working directory)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied/moved without changing files.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Overwrite an existing destination/<id> directory with the current output-* "
            "contents instead of skipping it. Use after a rerun so freshly recomputed "
            "spectra replace the previous run's copies."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    parent_dir = args.parent_dir.expanduser().resolve()
    destination_dir = args.destination.expanduser().resolve()
    # failed-corvus/ lives under the batch root (the scanned parent_dir), matching
    # failed-orca/ from the ORCA-convergence stage (fix #3). Previously it was
    # Path.cwd()/failed-corvus, which only happened to coincide with the batch root
    # because the postprocess job cd's there first.
    failed_corvus_dir = (parent_dir / "failed-corvus").resolve()

    if not parent_dir.exists() or not parent_dir.is_dir():
        print(f"Error: parent_dir does not exist or is not a directory: {parent_dir}", file=sys.stderr)
        return 1

    copied, skipped, quarantined = prepare_downloads(
        parent_dir, destination_dir, failed_corvus_dir, dry_run=args.dry_run, refresh=args.refresh
    )
    print(
        f"Done. Copied: {copied}, Skipped: {skipped}, "
        f"Moved to failed-corvus: {quarantined}"
    )
    print(f"Download destination: {destination_dir}")
    if quarantined:
        print(f"Failed-CORVUS quarantine: {failed_corvus_dir}")
    return 0

if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
