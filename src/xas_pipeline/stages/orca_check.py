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
written next to the CSV. Runs whose ORCA job is still running -- or still queued,
when the scheduler can say so -- are left alone. Exits 0 on normal operation
(handled failures are not errors), so the postprocess job carries on.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from xas_pipeline import batch_log, layout, recovery
from xas_pipeline.scheduler import SCHEDULERS, default_scheduler_name, get_scheduler


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


def _run_file(run_dir: Path, suffix: str) -> Path | None:
    """``<name><suffix>`` (else any ``*<suffix>``) in the run dir or its own working-<name>/.

    Never deeper: a legacy run dir can also host newer ``<id>-<mode>`` runs, and
    their files are not its own.
    """
    places = [run_dir, run_dir / f"working-{run_dir.name}"]
    for place in places:
        if (place / f"{run_dir.name}{suffix}").is_file():
            return place / f"{run_dir.name}{suffix}"
    for place in places:
        matches = sorted(p for p in place.glob(f"*{suffix}") if p.is_file())
        if matches:
            return matches[0]
    return None


def find_orca_log(run_dir: Path) -> Path | None:
    return _run_file(run_dir, "-orca.log")


def find_orca_timing(run_dir: Path) -> Path | None:
    return _run_file(run_dir, "-orca.timing")


def looks_like_run_dir(run_dir: Path) -> bool:
    """An ORCA run dir: it has an ORCA log or a generated ORCA job script."""
    return find_orca_log(run_dir) is not None or _run_file(run_dir, "-orca.script") is not None


def orca_run_is_in_flight(run_dir: Path) -> bool:
    """True when this run's ORCA job started but has not finished.

    The ORCA job script writes ``<id>-orca.timing`` with ``start_epoch`` up front
    and appends ``exit_code=`` only after ORCA returns, so a timing file without
    an exit code means the job is still going.

    This matters because several modes can be submitted into one batch root at
    different times. Without the check, a postprocess job that fires while
    another mode is still optimizing would see a half-written log, judge the run
    crashed, and move a live run dir into failed-orca/ mid-calculation.

    Absence of a timing file is NOT treated as in-flight: batches that predate
    the sidecar, and jobs that never started at all, must still be classified.
    """
    timing = find_orca_timing(run_dir)
    if timing is None:
        return False
    try:
        return "exit_code=" not in timing.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


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


def find_run_dirs(parent_dir: Path) -> list[Path]:
    """The ORCA run dirs under a batch root."""
    return [d for d in layout.iter_id_dirs(parent_dir) if looks_like_run_dir(d)]


def queued_orca_runs(parent_dir: Path, scheduler: str | None) -> set[str]:
    """Run ids whose latest ORCA job the scheduler still lists (empty if it cannot say)."""
    log_path = batch_log.find_batch_log(parent_dir)
    if log_path is None or scheduler not in SCHEDULERS:
        return set()
    latest: dict[str, str] = {}
    for name, status, job_id in batch_log.records(log_path):
        if status == "SUBMITTED" and job_id and name.startswith("orca-"):
            latest[recovery.run_id_from_job_name(name)] = job_id
    if not latest:
        return set()
    active = set(get_scheduler(scheduler).active_job_ids(list(latest.values())))
    return {run_id for run_id, job_id in latest.items() if job_id in active}


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
    parser.add_argument(
        "--scheduler",
        choices=sorted(SCHEDULERS),
        default=None,
        help="Ask this scheduler which runs are still queued, so they are not "
             "classified (default: $PIPELINE_SCHEDULER when set).",
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

    import os
    scheduler = args.scheduler or (default_scheduler_name() if os.environ.get("PIPELINE_SCHEDULER") else None)
    queued = queued_orca_runs(parent_dir, scheduler)

    scanned = 0
    failed = 0
    in_flight = 0
    for run_dir in find_run_dirs(parent_dir):
        if run_dir.name in queued and find_orca_log(run_dir) is None:
            in_flight += 1
            report_lines.append(f"{run_dir.name}: SKIPPED (ORCA job still queued)")
            print(f"QUEUED: {run_dir.name} (ORCA job not started yet; not classified)")
            continue
        if orca_run_is_in_flight(run_dir):
            # Another mode submitted into this batch root is still running. Leave
            # it completely alone -- classifying it now would quarantine a live
            # calculation. The postprocess pass that runs after it finishes will
            # pick it up.
            in_flight += 1
            report_lines.append(f"{run_dir.name}: SKIPPED (ORCA job still running)")
            print(f"IN FLIGHT: {run_dir.name} (ORCA still running; not classified)")
            continue

        scanned += 1
        ok, reason = classify_orca_run(run_dir)
        if not ok:
            failed += 1
            destination = layout.quarantine_move(run_dir, failed_dir)
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
            f"Skipped (ORCA still running): {in_flight}",
            f"Failed (moved to failed-orca): {failed}",
            f"Survivors written to CSV: {len(rows)}",
            f"CSV: {csv_path}",
        ]
    )
    report_path = args.output_dir / "orca-convergence-report.log"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if in_flight:
        print(f"Skipped {in_flight} run dir(s) whose ORCA job is still running")
    print(f"Scanned {scanned} run dir(s); {failed} failed ORCA moved to {failed_dir}")
    print(f"Wrote {len(rows)} survivor row(s) to {csv_path}")
    print(f"Wrote convergence report to {report_path}")

    if not args.no_batch_log:
        log_path = batch_log.find_batch_log(parent_dir)
        if log_path is not None:
            batch_log.append_outcomes(log_path, "ORCA outcomes", batch_outcomes)
            print(f"Appended {len(batch_outcomes)} ORCA outcome(s) to {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
