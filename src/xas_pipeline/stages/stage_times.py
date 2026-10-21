"""Collect each run's ORCA and CORVUS wall time into ``pipeline-stage-times.csv``.

The ORCA job writes ``<run_id>-orca.timing`` and the CORVUS wrapper
``<run_id>-corvus-<mode>.timing``, as ``key=value`` lines (``elapsed_seconds``,
``exit_code``, ...), in the run dir; FEFF processing later moves them into
``working-<id>/``. Both places are searched. Missing values are left blank.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from xas_pipeline import layout
from xas_pipeline.logs import setup_cli_logging

log = logging.getLogger(__name__)

CSV_NAME = "pipeline-stage-times.csv"
COLUMNS = ("run_id", "orca_elapsed_seconds", "orca_exit_code", "corvus_elapsed_seconds", "corvus_exit_code")


def read_timing(path: Path) -> dict[str, str]:
    """The ``key=value`` pairs of a .timing file."""
    pairs = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            pairs[key.strip()] = value.strip()
    return pairs


def find_timing(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    """The first of *names* found in the run's working roots."""
    for root in layout.working_roots(run_dir):
        for name in names:
            if (root / name).is_file():
                return root / name
    return None


def run_times(run_dir: Path) -> dict[str, str]:
    """One CSV row for a run dir."""
    run_id = run_dir.name
    row = {"run_id": run_id}
    for stage, names in (
        ("orca", (f"{run_id}-orca.timing",)),
        ("corvus", (f"{run_id}-corvus-xas.timing", f"{run_id}-corvus.timing")),
    ):
        path = find_timing(run_dir, names)
        timing = read_timing(path) if path else {}
        row[f"{stage}_elapsed_seconds"] = timing.get("elapsed_seconds", "")
        row[f"{stage}_exit_code"] = timing.get("exit_code", "")
    return row


def write_stage_times(batch_root: Path) -> Path:
    """Write ``<batch_root>/pipeline-stage-times.csv`` (one row per run dir)."""
    batch_root = Path(batch_root)
    out_csv = batch_root / CSV_NAME
    rows = [run_times(run_dir) for run_dir in layout.iter_id_dirs(batch_root)]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote timing summary: %s", out_csv)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Write {CSV_NAME}: ORCA and CORVUS wall time per run.")
    parser.add_argument("batch_root", type=Path, help="The batch root")
    args = parser.parse_args()
    setup_cli_logging()
    if not args.batch_root.is_dir():
        log.error("error: not a directory: %s", args.batch_root)
        return 2
    write_stage_times(args.batch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
