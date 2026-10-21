"""Count CORVUS imaginary-frequency warnings per run (standalone diagnostic).

Writes ``imaginary_frequencies.csv`` (cluster, imaginary_freq_count) into the
batch dir. The count is ``N/A`` when a run has no ``working-*`` dir or no
readable ``corvus-*.out``. Not part of the run-batch flow.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from xas_pipeline import layout
from xas_pipeline.logs import setup_cli_logging

log = logging.getLogger(__name__)

MARKER = "Found imaginary frequency with large weight"


def count_in_run(run_dir: Path) -> int | None:
    """Marker count in the run's ``working-<id>/corvus-*.out`` (None if absent/unreadable)."""
    working_dirs = sorted(run_dir.glob(f"working-{run_dir.name}")) or sorted(run_dir.glob("working-*"))
    if not working_dirs:
        log.info("  [SKIP] No working-* dir found in: %s", run_dir)
        return None
    out_files = sorted(working_dirs[0].glob("corvus-*.out"))
    if not out_files:
        log.info("  [SKIP] No corvus-*.out file found in: %s", working_dirs[0])
        return None
    try:
        # CORVUS .out files contain NUL bytes; read as text regardless.
        with out_files[0].open(encoding="utf-8", errors="replace") as handle:
            return sum(line.count(MARKER) for line in handle)
    except OSError as exc:
        log.info("  [ERROR] Could not read %s: %s", out_files[0], exc)
        return None


def count_imaginary_frequencies(family_dir: Path) -> dict[str, int | None]:
    """Marker count per run id (grouped per-mode runs each get a row)."""
    results: dict[str, int | None] = {}
    for run_dir in layout.iter_id_dirs(Path(family_dir).resolve()):
        results[run_dir.name] = count = count_in_run(run_dir)
        if count is not None:
            log.info("  %s: %d", run_dir.name, count)
    return results


def write_csv(family_dir: Path, results: dict[str, int | None]) -> Path:
    csv_path = Path(family_dir).resolve() / "imaginary_frequencies.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "imaginary_freq_count"])
        for name, count in results.items():
            writer.writerow([name, "N/A" if count is None else count])
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count CORVUS imaginary-frequency warnings per run.")
    parser.add_argument("family_dir", type=Path, help="Batch dir containing the run dirs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    family_dir = args.family_dir.expanduser()
    if not family_dir.is_dir():
        log.error("Error: '%s' is not a valid directory.", family_dir)
        return 1
    log.info("Scanning: %s\n", family_dir)
    csv_path = write_csv(family_dir, count_imaginary_frequencies(family_dir))
    log.info("\nResults written to: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
