"""Golden test: `xas-rerun-corvus --no-submit`.

xas-rerun-corvus recomputes CORVUS on a finished batch. With --no-submit it
archives, submits and logs nothing (batch-jobs.log is not even created -- a
deliberate change in the clean-up: a dry run has no side effects on the batch's
record), but still: resolves each id's run dir (split layout working-<id>/<id>.hess
here), writes the CORVUS wrapper (xas_pipeline.jobscripts), writes the
rerun-postprocess script, and writes a plain-text state file.

Determinism: --tag pins the otherwise-timestamped archive tag; the state file's
created_utc is a live timestamp so it is asserted on by field, not snapshotted.
The .hess is existence-probed only, so the batch is built in-tmp with an empty
marker.

The submit path (SUBMITTED / SUBMIT_FAILED lines) is covered offline by
tests/unit/test_orchestrate_submission.py through the scheduler.run_command seam.

Regenerate goldens after an intentional change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_rerun_corvus_dryrun.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES / "golden" / "rerun-corvus"
RUN_ID = "2j6a_ZN_homo_d2.60_cluster1"
MODE = "xas"
TAG = "testtag"
BATCH_NAME = "batch-out"

UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")

# Deterministic generated scripts (paths normalized).
SCRIPT_ARTIFACTS = (
    f"{RUN_ID}/working-{RUN_ID}/generated-{RUN_ID}-corvus-{MODE}-wrapper.script",
    f"generated-rerun-postprocess-{BATCH_NAME}-{MODE}.script",
)


def _normalize(text: str, *, batch: Path, repo_root: Path) -> str:
    text = text.replace(str(batch), "<BATCH>").replace(str(repo_root), "<REPO>")
    return _TS_RE.sub("<UTC>", text)


def _make_batch(parent: Path) -> Path:
    batch = parent / BATCH_NAME
    working = batch / RUN_ID / f"working-{RUN_ID}"
    working.mkdir(parents=True)
    (working / f"{RUN_ID}.hess").write_text("", encoding="utf-8")  # existence marker
    # A prior CORVUS output a real (submitting) rerun would archive aside.
    (working / "Corvus.cfavg_xas.out").write_text("prior\n", encoding="utf-8")
    return batch


@pytest.fixture(scope="module")
def rerun(tmp_path_factory, repo_root):
    from conftest import run_script

    batch = _make_batch(tmp_path_factory.mktemp("rerun"))
    result = run_script(
        "rerun-corvus.py", str(batch),
        "--corvus-mode", MODE, "--tag", TAG, "--scheduler", "slurm", "--no-submit",
    )
    return {"result": result, "batch": batch, "repo_root": repo_root}


_SUMMARY = "Re-ran 1 run(s), skipped 0, failed 0; postprocess NO_SUBMIT; tag testtag"


def test_exits_clean(rerun):
    result = rerun["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_console_script_reports_the_rerun(tmp_path):
    """The operator entry point (console script) prints the run summary."""
    exe = Path(sys.executable).parent / "xas-rerun-corvus"
    if not exe.is_file():
        pytest.skip("xas-rerun-corvus console script not installed in this venv")
    batch = _make_batch(tmp_path)
    result = subprocess.run(
        [str(exe), str(batch), "--tag", TAG, "--scheduler", "slurm", "--no-submit"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _SUMMARY in result.stdout


def test_python_m_reports_the_rerun(rerun):
    """Job scripts invoke every stage as `python -m ...`; their log lines must survive."""
    assert _SUMMARY in rerun["result"].stdout


def test_generated_scripts_match_golden(rerun):
    batch, repo_root = rerun["batch"], rerun["repo_root"]
    mismatches, missing = [], []
    for rel in SCRIPT_ARTIFACTS:
        produced = batch / rel
        assert produced.is_file(), f"artifact not generated: {rel}"
        actual = _normalize(produced.read_text(encoding="utf-8"), batch=batch, repo_root=repo_root)
        golden = GOLDEN_DIR / rel
        if UPDATE:
            golden.parent.mkdir(parents=True, exist_ok=True)
            golden.write_text(actual, encoding="utf-8")
            continue
        if not golden.is_file():
            missing.append(rel)
        elif actual != golden.read_text(encoding="utf-8"):
            mismatches.append(rel)
    if UPDATE:
        pytest.skip("GOLDEN_UPDATE=1: wrote rerun goldens")
    assert not missing, f"missing goldens (run GOLDEN_UPDATE=1): {missing}"
    assert not mismatches, f"content drift: {mismatches}"


def test_dry_run_leaves_no_batch_log(rerun):
    """Behaviour change (clean-up): --no-submit no longer writes SKIPPED lines.

    A dry run must not touch the batch's record; the old SKIPPED lines also made
    a later reader count a rerun that never happened.
    """
    assert not (rerun["batch"] / "batch-jobs.log").exists()
    # ... and archived nothing: the prior output is in place, and the only
    # tagged path is the state file.
    working = rerun["batch"] / RUN_ID / f"working-{RUN_ID}"
    assert (working / "Corvus.cfavg_xas.out").read_text(encoding="utf-8") == "prior\n"
    tagged = [p.name for p in rerun["batch"].rglob(f"*{TAG}*")]
    assert tagged == [f"rerun-state-{BATCH_NAME}-{MODE}-{TAG}.log"]


def test_state_file_invariants(rerun):
    # Fix #4: state is now a plain-text .log (was .json), consistent with the
    # run-batch submission-state log.
    state_path = rerun["batch"] / f"rerun-state-{BATCH_NAME}-{MODE}-{TAG}.log"
    assert state_path.is_file()
    text = state_path.read_text(encoding="utf-8")
    assert f"corvus_mode:          {MODE}" in text
    assert f"tag:                  {TAG}" in text
    assert "scheduler:            slurm" in text
    assert "postprocess_job_id:   NO_SUBMIT" in text
    assert "Runs (1):" in text
    assert f"  {RUN_ID}" in text
    assert "corvus_job_id:  NO_SUBMIT" in text
