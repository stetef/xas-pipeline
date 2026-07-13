"""Characterization test: `script-check-orca-convergence-and-extract-times.py`.

Pins the classify/extract/move behavior before the reorg rewires this script
through xas_pipeline.layout. Two synthetic ORCA logs exercise the two dominant
branches: a normally-terminated run (extract runtime + Final Gibbs -> CSV row)
and an SCF-non-convergence failure (named reason -> moved to failed-orca/).

Golden: the CSV (deterministic, no paths) byte-for-byte; the report with its
timestamp and absolute paths normalized. Regenerate with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_orca_convergence_check.py
"""

import os
import re
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "golden" / "orca-convergence"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
_TS_RE = re.compile(r"Run timestamp: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

SUCCESS_LOG = """\
                                 *****************
                                 * O   R   C   A *
                                 *****************
OPTIMIZATION RUN DONE
Final Gibbs free energy         ...   -2934.1234567 Eh
****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 2 hours 15 minutes 6 seconds 789 msec
"""

SCF_FAIL_LOG = """\
                                 *****************
                                 * O   R   C   A *
                                 *****************
SCF has not converged after 125 iterations
ORCA finished by error termination in SCF
"""


def _normalize(text: str, *, batch: Path) -> str:
    text = text.replace(str(batch), "<BATCH>")
    return _TS_RE.sub("Run timestamp: <TS>", text)


@pytest.fixture(scope="module")
def check_run(tmp_path_factory):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("orcachk") / "batch-out"
    for run_id, log in (("run-ok", SUCCESS_LOG), ("run-scf-fail", SCF_FAIL_LOG)):
        working = batch / run_id / f"working-{run_id}"
        working.mkdir(parents=True)
        (working / f"{run_id}-orca.log").write_text(log, encoding="utf-8")

    result = run_script(
        "script-check-orca-convergence-and-extract-times.py",
        str(batch), "--output-dir", str(batch), "--no-batch-log",
    )
    return {"result": result, "batch": batch}


def test_exits_clean(check_run):
    result = check_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_failed_run_moved_survivor_kept(check_run):
    batch = check_run["batch"]
    assert (batch / "failed-orca" / "run-scf-fail").is_dir()
    assert not (batch / "run-scf-fail").exists()
    assert (batch / "run-ok").is_dir()  # survivor stays in place


def test_csv_and_report_match_golden(check_run):
    batch = check_run["batch"]
    artifacts = {
        "batch-out-orca-compute-times.csv": (batch / "batch-out-orca-compute-times.csv").read_text(encoding="utf-8"),
        "orca-convergence-report.log": _normalize(
            (batch / "orca-convergence-report.log").read_text(encoding="utf-8"), batch=batch
        ),
    }
    mismatches, missing = [], []
    for name, actual in artifacts.items():
        golden = GOLDEN_DIR / name
        if UPDATE:
            golden.parent.mkdir(parents=True, exist_ok=True)
            golden.write_text(actual, encoding="utf-8")
            continue
        if not golden.is_file():
            missing.append(name)
        elif actual != golden.read_text(encoding="utf-8"):
            mismatches.append(name)
    if UPDATE:
        pytest.skip("GOLDEN_UPDATE=1: wrote convergence goldens")
    assert not missing, f"missing goldens (run GOLDEN_UPDATE=1): {missing}"
    assert not mismatches, f"content drift: {mismatches}"
