#!/usr/bin/env python3
"""Check ORCA convergence and extract run times for a batch of run directories.

For each first-level run directory under ``parent_dir`` this script:

1. Locates the ``*-orca.log`` (flat ``<id>/<id>-orca.log`` or already split into
   ``<id>/working-<id>/<id>-orca.log``).
2. Decides whether the ORCA job succeeded. A run is treated as FAILED when:
     - the ORCA log is missing/unreadable, or
     - the optimization explicitly did not converge
       ("The optimization did not converge but reached the maximum number of"), or
     - the log never reached "****ORCA TERMINATED NORMALLY****" (crashed/killed,
       e.g. segfault or OOM). This is the decisive crash signal: a job can write a
       multi-frame trajectory yet still die mid-SCF, so a frame-count heuristic
       alone is not enough.
3. FAILED runs are *moved* into a sibling ``failed-orca/`` directory (created under
   ``parent_dir``) so downstream postprocess steps never see them.
4. SUCCESSFUL runs contribute a row (TOTAL RUN TIME + Final Gibbs free energy) to
   ``<parent_dir_name>-orca-compute-times.csv``.

A human-readable ``orca-convergence-report.log`` summarising every decision is
written next to the CSV. The script always exits 0 on normal operation (handled
failures are not errors) so the batch postprocess job continues to the FEFF and
download stages.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path


RUNTIME_LINE_RE = re.compile(r"^TOTAL RUN TIME:\s*(.+?)\s*$")
FINAL_GIBBS_RE = re.compile(
    r"^\s*Final Gibbs free energy\s*\.\.\.\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh\s*$"
)
TERMINATION_MARKER = "****ORCA TERMINATED NORMALLY****"
NON_CONVERGENCE_TEXT = "The optimization did not converge but reached the maximum number of"
OPTIMIZATION_RUN_DONE = "OPTIMIZATION RUN DONE"
MIN_TRAJECTORY_FRAMES = 2

# Directories under parent_dir that are never ORCA run directories.
SKIP_DIR_NAMES = {
    "failed-orca",
    "failed-corvus",
    "downloading-station",
    "xyz_files",
    "optimized_xyz_files",
}


def extract_runtime_from_log(log_path: Path) -> str | None:
    """Return the last TOTAL RUN TIME value from a normally terminated ORCA log."""
    last_runtime = None
    saw_termination_marker = False

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if TERMINATION_MARKER in line:
                saw_termination_marker = True
                continue

            match = RUNTIME_LINE_RE.match(line)
            if saw_termination_marker and match:
                last_runtime = match.group(1)
                saw_termination_marker = False

    return last_runtime


def extract_final_gibbs_from_log(log_path: Path) -> str | None:
    """Return the last 'Final Gibbs free energy' value (in Eh) from an ORCA log."""
    last_final_gibbs = None

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = FINAL_GIBBS_RE.match(line)
            if match:
                last_final_gibbs = match.group(1)

    return last_final_gibbs


def count_trajectory_frames(trj_path: Path) -> int | None:
    """Count geometry frames in an XYZ trajectory file. Returns None on error."""
    if not trj_path.exists():
        return None
    try:
        lines = trj_path.read_text(errors="ignore").splitlines()
        if len(lines) < 1:
            return 0
        first_line = lines[0].strip().split()
        if not first_line or not first_line[0].isdigit():
            return None
        nat = int(first_line[0])
        if nat <= 0:
            return None
        # Each frame is: 1 line (atom count) + nat lines (atoms) + 1 line (comment).
        return len(lines) // (nat + 2)
    except (ValueError, IndexError):
        return None


def find_orca_log(run_dir: Path) -> Path | None:
    """Locate the ORCA log inside a run dir (flat or split into working-<id>)."""
    preferred = list(run_dir.rglob(f"{run_dir.name}-orca.log"))
    if preferred:
        return min(preferred, key=lambda p: len(p.parts))
    any_log = [p for p in run_dir.rglob("*-orca.log") if p.is_file()]
    if any_log:
        return min(any_log, key=lambda p: len(p.parts))
    return None


def looks_like_run_dir(run_dir: Path) -> bool:
    """A child dir is an ORCA run dir if it has an ORCA log or a generated ORCA script."""
    if find_orca_log(run_dir) is not None:
        return True
    return any(run_dir.rglob("generated-*-orca.script"))


def classify_orca_run(run_dir: Path) -> tuple[bool, str]:
    """Return (ok, reason) describing whether the ORCA job succeeded."""
    log_path = find_orca_log(run_dir)
    if log_path is None:
        return False, "ORCA log not found (job produced no log; likely never started)"

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"failed to read {log_path.name} ({exc})"

    if NON_CONVERGENCE_TEXT in log_text:
        return False, "optimization did not converge (reached the maximum number of cycles)"

    if TERMINATION_MARKER not in log_text:
        stem = log_path.name.removesuffix("-orca.log")
        trj_path = log_path.parent / f"{stem}_trj.xyz"
        trj_frames = count_trajectory_frames(trj_path)
        has_run_done = OPTIMIZATION_RUN_DONE in log_text
        if not has_run_done and (
            trj_frames is None or trj_frames < MIN_TRAJECTORY_FRAMES
        ):
            return False, (
                "no normal termination and no completed optimization step "
                "— job likely crashed or was killed (OOM/segfault?) early. "
                "Check: sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS"
            )
        return False, (
            "ORCA did not terminate normally (no '****ORCA TERMINATED NORMALLY****'); "
            "crashed or was killed mid-run"
        )

    return True, "terminated normally"


def find_run_dirs(parent_dir: Path):
    """Yield first-level run directories under parent_dir."""
    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIR_NAMES:
            continue
        if looks_like_run_dir(child):
            yield child


def move_to_failed_orca(run_dir: Path, failed_dir: Path) -> Path:
    """Move a failed run dir into failed-orca/, replacing any stale copy."""
    failed_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_dir / run_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(run_dir), str(destination))
    return destination


def write_csv(output_path: Path, rows: list[tuple[str, str, str | None]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "total_run_time", "final_gibbs_free_energy_eh"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check ORCA convergence for each run directory under a parent directory: "
            "move failed runs into failed-orca/, and write TOTAL RUN TIME / Final Gibbs "
            "free energy for the survivors to a CSV."
        )
    )
    parser.add_argument("parent_dir", type=Path, help="Parent directory containing ORCA run directories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the CSV and report are written (default: current working directory)",
    )
    args = parser.parse_args()
    parent_dir = args.parent_dir.resolve()

    if not parent_dir.is_dir():
        raise SystemExit(f"Error: '{parent_dir}' is not a directory.")

    failed_dir = parent_dir / "failed-orca"

    rows: list[tuple[str, str, str | None]] = []
    report_lines = [
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Parent directory: {parent_dir}",
        "",
    ]

    scanned = 0
    failed = 0
    for run_dir in find_run_dirs(parent_dir):
        scanned += 1
        ok, reason = classify_orca_run(run_dir)
        if not ok:
            failed += 1
            destination = move_to_failed_orca(run_dir, failed_dir)
            report_lines.append(f"{run_dir.name}: FAILED ({reason}); moved to {destination}")
            print(f"FAILED ORCA: {run_dir.name} -> {destination} ({reason})")
            continue

        log_path = find_orca_log(run_dir)
        runtime = extract_runtime_from_log(log_path) if log_path else None
        final_gibbs = extract_final_gibbs_from_log(log_path) if log_path else None
        rows.append((run_dir.name, runtime if runtime is not None else "", final_gibbs))
        report_lines.append(f"{run_dir.name}: OK ({reason})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{parent_dir.name}-orca-compute-times.csv"
    write_csv(csv_path, rows)

    report_lines.extend(
        [
            "",
            f"Scanned run directories: {scanned}",
            f"Failed (moved to failed-orca): {failed}",
            f"Survivors written to CSV: {len(rows)}",
            f"CSV: {csv_path}",
        ]
    )
    report_path = args.output_dir / "orca-convergence-report.log"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Scanned {scanned} run dir(s); {failed} failed ORCA moved to {failed_dir}")
    print(f"Wrote {len(rows)} survivor row(s) to {csv_path}")
    print(f"Wrote convergence report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
