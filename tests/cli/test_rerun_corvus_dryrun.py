"""Golden test: `rerun-corvus.py --no-submit`.

rerun-corvus re-runs a single CORVUS mode on a finished batch. In --no-submit
mode it skips archiving and submission but still: resolves each id's run dir
(split layout working-<id>/<id>.hess here), writes the per-mode wrapper via the
orchestrator's shared helper, writes the rerun-postprocess script, records
SKIPPED lines in batch-jobs.log, and writes a plain-text state file (fix #4:
was JSON). It reuses xas_pipeline.orchestrate directly.

Determinism: --tag pins the otherwise-timestamped archive tag; the state file's
created_utc is a live timestamp so it is asserted on by field, not snapshotted.
The .hess is existence-probed only, so the batch is built in-tmp with an empty
marker.

COVERAGE NOTE: issue #2 (rerun-corvus logging a *submission* as SUCCEEDED rather
than SUBMITTED) lives in the submit path, which needs a real scheduler and
cannot run offline. This --no-submit golden pins the SKIPPED vocabulary instead;
the #2 fix is verified by inspection when applied.

Regenerate goldens after an intentional change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_rerun_corvus_dryrun.py
"""

import os
import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES / "golden" / "rerun-corvus"
RUN_ID = "2j6a_ZN_homo_d2.60_cluster1"
MODE = "xanes"
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


@pytest.fixture(scope="module")
def rerun(tmp_path_factory, repo_root):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("rerun") / BATCH_NAME
    working = batch / RUN_ID / f"working-{RUN_ID}"
    working.mkdir(parents=True)
    (working / f"{RUN_ID}.hess").write_text("", encoding="utf-8")  # existence marker

    result = run_script(
        "rerun-corvus.py", str(batch),
        "--corvus-mode", MODE, "--tag", TAG, "--scheduler", "slurm", "--no-submit",
    )
    return {"result": result, "batch": batch, "repo_root": repo_root}


def test_exits_clean(rerun):
    result = rerun["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert f"Re-ran mode '{MODE}' for 1 run(s)" in result.stdout


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


def test_batch_log_records_skips(rerun):
    log = (rerun["batch"] / "batch-jobs.log").read_text(encoding="utf-8")
    assert f"rerun-corvus-{MODE}-{RUN_ID}\tSKIPPED" in log
    assert f"rerun-postprocess-{BATCH_NAME}\tSKIPPED" in log


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
