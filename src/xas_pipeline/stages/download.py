"""Stage surviving runs' outputs for download and quarantine CORVUS failures.

For each run dir under the batch root:

- if its id is in ``corvus-failed-ids.txt`` (written by xas-process-feff), the
  run dir is moved into ``<batch_root>/failed-corvus/``;
- otherwise its ``output-*`` dir is copied to ``<destination>/<path under the
  batch root>``, so per-mode runs stay grouped as on disk.

Runs the ORCA check already moved into ``failed-orca/`` are never seen here, so
the destination holds only the survivors.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from xas_pipeline import layout
from xas_pipeline.logs import setup_cli_logging

log = logging.getLogger(__name__)


@dataclass
class DownloadSummary:
    copied: int = 0
    skipped: int = 0
    quarantined: int = 0


def download_subpath(run_dir: Path, batch_root: Path) -> Path:
    """Where a run's outputs land under the destination: its path below the batch root."""
    try:
        return run_dir.relative_to(batch_root)
    except ValueError:
        return Path(run_dir.name)


def copy_output_dirs(
    run_dir: Path, destination: Path, batch_root: Path, *, dry_run: bool = False, refresh: bool = False
) -> tuple[int, int]:
    """Copy a run's ``output*`` dir(s) to ``destination/<subpath>``; returns (copied, skipped).

    An existing target is skipped, or replaced when ``refresh`` is set (so a rerun's
    recomputed spectra replace the previous copies).
    """
    output_dirs = [p for p in sorted(run_dir.glob("output*")) if p.is_dir()]
    if not output_dirs:
        log.warning("warning: no output* directory in %s; nothing to copy", run_dir)
        return 0, 0
    target = destination / download_subpath(run_dir, batch_root)
    copied = skipped = 0
    for output_dir in output_dirs:
        if target.exists() and not refresh:
            log.info("SKIP (exists): %s", target)
            skipped += 1
            continue
        if target.exists():
            log.info("REFRESH (overwrite existing): %s -> %s", output_dir, target)
            if not dry_run:
                shutil.rmtree(target)
        else:
            log.info("COPY: %s -> %s", output_dir, target)
        if not dry_run:
            shutil.copytree(output_dir, target)
        copied += 1
    return copied, skipped


def prepare_downloads(
    batch_root: Path, destination: Path, *, dry_run: bool = False, refresh: bool = False
) -> DownloadSummary:
    """Copy survivors to ``destination`` and move CORVUS failures to ``failed-corvus/``."""
    failed_ids = set(layout.read_failed_corvus_ids(batch_root))
    failed_corvus_dir = batch_root / "failed-corvus"
    summary = DownloadSummary()
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    for run_dir in layout.iter_id_dirs(batch_root):
        if run_dir.name in failed_ids:
            log.info("FAILED-CORVUS: %s -> %s", run_dir, failed_corvus_dir / run_dir.name)
            if not dry_run:
                layout.quarantine_move(run_dir, failed_corvus_dir)
            summary.quarantined += 1
            continue
        copied, skipped = copy_output_dirs(run_dir, destination, batch_root, dry_run=dry_run, refresh=refresh)
        summary.copied += copied
        summary.skipped += skipped
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy surviving runs' output* dirs to a download destination and "
        "move CORVUS-failed runs into failed-corvus/."
    )
    parser.add_argument("parent_dir", type=Path, help="Batch root to scan")
    parser.add_argument("-d", "--destination", type=Path, default=Path("downloading-station"),
                        help="Where to copy outputs (default: ./downloading-station)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be copied/moved; change nothing")
    parser.add_argument("--refresh", action="store_true",
                        help="Replace existing copies (use after a rerun) instead of skipping them")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    batch_root = args.parent_dir.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not batch_root.is_dir():
        log.error("Error: parent_dir does not exist or is not a directory: %s", batch_root)
        return 1
    summary = prepare_downloads(batch_root, destination, dry_run=args.dry_run, refresh=args.refresh)
    log.info("Done. Copied: %d, Skipped: %d, Moved to failed-corvus: %d",
             summary.copied, summary.skipped, summary.quarantined)
    log.info("Download destination: %s", destination)
    if summary.quarantined:
        log.info("Failed-CORVUS quarantine: %s", batch_root / "failed-corvus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
