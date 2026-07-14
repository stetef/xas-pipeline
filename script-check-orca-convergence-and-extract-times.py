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
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline import layout
from xas_pipeline.batch_log import append_outcomes, find_batch_log


RUNTIME_LINE_RE = re.compile(r"^TOTAL RUN TIME:\s*(.+?)\s*$")
FINAL_GIBBS_RE = re.compile(
    r"^\s*Final Gibbs free energy\s*\.\.\.\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh\s*$"
)
TERMINATION_MARKER = "****ORCA TERMINATED NORMALLY****"
NON_CONVERGENCE_TEXT = "The optimization did not converge but reached the maximum number of"
OPTIMIZATION_RUN_DONE = "OPTIMIZATION RUN DONE"
MIN_TRAJECTORY_FRAMES = 2

# Specific ORCA failure signatures, checked before the generic
# "did not terminate normally" fallback so the report/log can name the actual
# cause and the fix. Order matters: the first match that fires wins.
#
# CHARGE_MULT_RE   -- e.g. "multiplicity (1) is odd and number of electrons (793)
#                     is odd -> impossible": the requested CHARGE/MULT is
#                     inconsistent with the cluster's electron count (bad input),
#                     dies in ~1s before any SCF.
# OOM_COSX_TEXT    -- the RIJCOSX exchange build ran out of per-process memory,
#                     typically during the analytic-frequency (AnFreq) step; ORCA
#                     itself tells us to raise %MaxCore.
# SCF_NONCONV_TEXT -- the SCF failed to converge (error terminates in LEANSCF).
# ERROR_TERM_RE    -- generic "ORCA finished by error termination in <MODULE>";
#                     used to name the failing module when nothing more specific
#                     matched.
CHARGE_MULT_RE = re.compile(
    r"multiplicity \((\d+)\) is (?:odd|even) and number of electrons \((\d+)\) is "
    r"(?:odd|even) -> impossible"
)
OOM_COSX_TEXT = "No memory left for COSX"
OOM_MAXCORE_TEXT = "Increase the %MAXCORE"
SCF_NONCONV_TEXT = "SCF has not converged"
SCF_NONCONV_TEXT_ALT = "SCF NOT CONVERGED"
ERROR_TERM_RE = re.compile(r"ORCA finished by error termination in ([A-Za-z0-9_]+)")
# Modules that run after the geometry optimization (frequencies / properties). A
# crash here means the optimization itself was fine but no Hessian (.hess) was
# written, which is what later makes the CORVUS/FEFF stage fail with a missing
# Hessian — so we call that out explicitly.
POST_OPT_MODULES = {"PROPINT", "NUMFREQ", "ANFREQ", "FREQ", "HESSIAN"}

# Directories under parent_dir that are never ORCA run directories.
SKIP_DIR_NAMES = layout.SKIP_DIR_NAMES


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
        # No normal termination: try to name the specific cause before falling
        # back to the generic "crashed/killed" message. See the *_RE / *_TEXT
        # constants above for what each signature means and how to fix it.
        charge_mult = CHARGE_MULT_RE.search(log_text)
        if charge_mult:
            mult, nelec = charge_mult.group(1), charge_mult.group(2)
            return False, (
                f"charge/multiplicity error: requested multiplicity {mult} is "
                f"impossible for {nelec} electrons (parity mismatch) — the ORCA "
                "input never ran. Fix CHARGE/MULT in the .in (or the carved cluster "
                "composition) and resubmit"
            )

        error_term = ERROR_TERM_RE.search(log_text)
        failed_module = error_term.group(1).upper() if error_term else None

        if OOM_COSX_TEXT in log_text or OOM_MAXCORE_TEXT in log_text:
            where = (
                f" in the {failed_module} step" if failed_module else ""
            )
            opt_ok = OPTIMIZATION_RUN_DONE in log_text
            hess_note = (
                " The optimization finished but no Hessian (.hess) was written, "
                "so the CORVUS/FEFF stage cannot run."
                if opt_ok
                else ""
            )
            return False, (
                f"out of memory{where}: RIJCOSX exchange build exhausted per-process "
                "memory ('No memory left for COSX RHS'). Raise ORCA %MaxCore and/or the "
                f"SBATCH --mem for this job, then resubmit.{hess_note}"
            )

        if SCF_NONCONV_TEXT in log_text or SCF_NONCONV_TEXT_ALT in log_text:
            where = f" (error terminated in {failed_module})" if failed_module else ""
            return False, (
                f"SCF did not converge{where} — try SlowConv/NRSCF, a better initial "
                "guess, or a smaller convergence target, then resubmit"
            )

        if failed_module in POST_OPT_MODULES:
            return False, (
                f"post-optimization step failed (error terminated in {failed_module}); "
                "the geometry optimization completed but no Hessian (.hess) was written, "
                "so the CORVUS/FEFF stage cannot run. Check the ORCA log tail for the cause"
            )

        if failed_module:
            return False, (
                f"ORCA error-terminated in {failed_module} (no "
                "'****ORCA TERMINATED NORMALLY****'); check the ORCA log tail for the cause"
            )

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
    for child in layout.iter_id_dirs(parent_dir):
        if looks_like_run_dir(child):
            yield child


def move_to_failed_orca(run_dir: Path, failed_dir: Path) -> Path:
    """Move a failed run dir into failed-orca/, replacing any stale copy."""
    return layout.quarantine_move(run_dir, failed_dir)


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
    parser.add_argument(
        "--no-batch-log",
        action="store_true",
        help="Do not append authoritative ORCA outcomes to batch-jobs.log.",
    )
    args = parser.parse_args()
    parent_dir = args.parent_dir.resolve()

    if not parent_dir.is_dir():
        raise SystemExit(f"Error: '{parent_dir}' is not a directory.")

    failed_dir = parent_dir / "failed-orca"
    # (job_name, status, reason) tuples appended to batch-jobs.log after the scan.
    batch_outcomes: list[tuple[str, str, str | None]] = []

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
            batch_outcomes.append((f"orca-{run_dir.name}", "FAILED", reason))
            continue

        log_path = find_orca_log(run_dir)
        runtime = extract_runtime_from_log(log_path) if log_path else None
        final_gibbs = extract_final_gibbs_from_log(log_path) if log_path else None
        rows.append((run_dir.name, runtime if runtime is not None else "", final_gibbs))
        report_lines.append(f"{run_dir.name}: OK ({reason})")
        batch_outcomes.append((f"orca-{run_dir.name}", "OK", None))

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

    if not args.no_batch_log:
        batch_log = find_batch_log(parent_dir)
        if batch_log is not None:
            append_outcomes(batch_log, "ORCA outcomes", batch_outcomes)
            print(f"Appended {len(batch_outcomes)} ORCA outcome(s) to {batch_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
