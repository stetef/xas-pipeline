"""Golden test: `submit-corvus-only.py --no-submit`.

submit-corvus-only regenerates the per-mode CORVUS wrapper scripts for a batch
whose ORCA runs are already done (flat layout: batch/<id>/<id>.hess) and, in
--no-submit mode, writes them without touching the scheduler. It reuses
run-batch-pipeline's _write_corvus_wrapper_script via the importlib-by-path hack
that the package reorg will remove -- so this snapshot guards that behavior
across the restructure.

The .hess is only probed for existence (never read), so the fixture batch is
built in-tmp with empty markers -- no committed fixture files.

NOTE: the current --scheduler default is "slurm" (issue #1: diverges from the
other entry points' env default). This test passes --scheduler explicitly so it
does not depend on that default; a separate assertion documents the default.

Regenerate goldens after an intentional change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_submit_corvus_only_dryrun.py
"""

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES / "golden" / "submit-corvus"
IDS = ("2j6a_ZN_homo_d2.60_cluster1", "5c1z_ZN_homo_d2.60_cluster15")
MODES = ("exafs", "xanes")

UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


def _normalize(text: str, *, batch: Path, repo_root: Path) -> str:
    return text.replace(str(batch), "<BATCH>").replace(str(repo_root), "<REPO>")


@pytest.fixture(scope="module")
def submit_run(tmp_path_factory, repo_root):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("submit") / "batch-out"
    for run_id in IDS:
        run_dir = batch / run_id
        run_dir.mkdir(parents=True)
        (run_dir / f"{run_id}.hess").write_text("", encoding="utf-8")  # existence marker

    result = run_script("submit-corvus-only.py", str(batch), "--scheduler", "slurm", "--no-submit")
    return {"result": result, "batch": batch, "repo_root": repo_root}


def test_exits_clean(submit_run):
    result = submit_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "generated 4 wrapper script(s), none submitted." in result.stdout


def test_wrappers_match_golden(submit_run):
    batch, repo_root = submit_run["batch"], submit_run["repo_root"]
    mismatches, missing = [], []
    for run_id in IDS:
        for mode in MODES:
            rel = f"{run_id}/generated-{run_id}-corvus-{mode}-wrapper.script"
            produced = batch / rel
            assert produced.is_file(), f"wrapper not generated: {rel}"
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
        pytest.skip("GOLDEN_UPDATE=1: wrote wrapper goldens")
    assert not missing, f"missing goldens (run GOLDEN_UPDATE=1): {missing}"
    assert not mismatches, f"wrapper content drift: {mismatches}"
