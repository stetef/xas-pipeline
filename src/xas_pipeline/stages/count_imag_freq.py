#!/usr/bin/env python3
"""Scan a family directory for imaginary-frequency warnings in corvus-*.out files.

Writes ``imaginary_frequencies.csv`` (columns: cluster, imaginary_freq_count) into
the family directory. The count is ``None`` (rendered "N/A") when a run's
``working-*`` dir or ``corvus-*.out`` is missing or unreadable.

Standalone diagnostic, not part of the run-batch flow. Run via
``xas-count-imag-freq <family_dir>`` or ``python -m xas_pipeline.stages.count_imag_freq``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MARKER = "Found imaginary frequency with large weight"


def count_imaginary_frequencies(family_dir: Path) -> dict[str, int | None]:
    """Count the imaginary-frequency marker in each id's corvus-*.out."""
    family_path = Path(family_dir).resolve()
    results: dict[str, int | None] = {}

    for entry in sorted(family_path.iterdir()):
        if not entry.is_dir():
            continue

        subdir_name = entry.name

        # Prefer the exact working-<id>/ dir; fall back to a looser glob.
        working_dirs = list(entry.glob(f"working-{subdir_name}")) or list(entry.glob("working-*"))
        if not working_dirs:
            print(f"  [SKIP] No working-* dir found in: {entry}")
            results[subdir_name] = None
            continue

        working_dir = working_dirs[0]
        out_files = list(working_dir.glob("corvus-*.out"))
        if not out_files:
            print(f"  [SKIP] No corvus-*.out file found in: {working_dir}")
            results[subdir_name] = None
            continue

        out_file = out_files[0]
        count = 0
        try:
            with open(out_file, "r", errors="replace") as handle:
                for line in handle:
                    count += line.count(MARKER)
        except OSError as exc:
            print(f"  [ERROR] Could not read {out_file}: {exc}")
            results[subdir_name] = None
            continue

        print(f"  {subdir_name}: {count}")
        results[subdir_name] = count

    return results


def write_csv(family_dir: Path, results: dict[str, int | None]) -> Path:
    csv_path = Path(family_dir).resolve() / "imaginary_frequencies.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cluster", "imaginary_freq_count"])
        for name, count in results.items():
            writer.writerow([name, count if count is not None else "N/A"])
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "family_dir",
        type=Path,
        help="Family directory containing per-id run directories.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    family_dir = args.family_dir.expanduser()
    if not family_dir.is_dir():
        print(f"Error: '{family_dir}' is not a valid directory.")
        return 1

    print(f"Scanning: {family_dir}\n")
    results = count_imaginary_frequencies(family_dir)
    csv_path = write_csv(family_dir, results)
    print(f"\nResults written to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
