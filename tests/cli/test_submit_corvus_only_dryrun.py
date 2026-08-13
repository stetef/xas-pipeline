"""Golden test: `submit-corvus-only.py --no-submit`.

submit-corvus-only regenerates the per-mode CORVUS wrapper scripts for a batch
whose ORCA runs are already done (flat layout: batch/<id>/<id>.hess) and, in
--no-submit mode, writes them without touching the scheduler. It reuses the
orchestrator's _write_corvus_wrapper_script (xas_pipeline.orchestrate) -- so this
snapshot guards that behavior across the restructure.

The .hess is only probed for existence (never read), so the fixture batch is
built in-tmp with empty markers -- no committed fixture files.

This test passes --scheduler explicitly so it does not depend on the default.
(Fix #1 made the default PIPELINE_SCHEDULER env -> pbs, matching the other
entry points; it was formerly hardcoded "slurm".)

Regenerate goldens after an intentional change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_submit_corvus_only_dryrun.py
"""

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES / "golden" / "submit-corvus"
IDS = ("2j6a_ZN_homo_d2.60_cluster1", "5c1z_ZN_homo_d2.60_cluster15")
MODES = ("xas",)

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
    assert "generated 2 wrapper script(s), none submitted." in result.stdout


def test_batch_log_records_skipped(submit_run):
    # Fix #2: submit-corvus-only used to be silent; it now writes a batch-jobs.log
    # with the shared vocabulary (SKIPPED in --no-submit, one line per id x mode).
    log = (submit_run["batch"] / "batch-jobs.log").read_text(encoding="utf-8")
    for run_id in IDS:
        for mode in MODES:
            assert f"submit-corvus-{mode}-{run_id}\tSKIPPED" in log


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


def test_discovers_nested_and_interp_run_dirs(tmp_path):
    """Grouped batches, and interp runs that have no .hess yet.

    Nesting run dirs under a per-structure group dir broke a first-level-only
    scan; and an interp run legitimately has no Hessian at submit time, because
    its wrapper builds one from the spring models. Both must be found.
    """
    from xas_pipeline.cli.submit_corvus import _discover_run_dirs

    batch = tmp_path / "batch"
    # Grouped: one mode with an ORCA Hessian, one interp mode without.
    (batch / "CL1" / "CL1-ca-fixed").mkdir(parents=True)
    (batch / "CL1" / "CL1-ca-fixed" / "CL1-ca-fixed.hess").write_text("", encoding="utf-8")
    (batch / "CL1" / "CL1-interp").mkdir(parents=True)
    (batch / "CL1" / "CL1-interp" / "CL1-interp.xyz").write_text("", encoding="utf-8")
    # A grouped run that is neither: no Hessian and not an interp run.
    (batch / "CL1" / "CL1-free").mkdir(parents=True)
    (batch / "CL1" / "CL1-free" / "CL1-free.xyz").write_text("", encoding="utf-8")
    # Pre-grouping flat run dir, still supported.
    (batch / "OLD").mkdir(parents=True)
    (batch / "OLD" / "OLD.hess").write_text("", encoding="utf-8")

    found = sorted(p.name for p in _discover_run_dirs(batch))
    assert found == ["CL1-ca-fixed", "CL1-interp", "OLD"]


def test_ids_filter_selects_a_single_set(tmp_path):
    """Submitting one set while another set in the same batch root still runs."""
    from xas_pipeline.cli.submit_corvus import _discover_run_dirs

    batch = tmp_path / "batch"
    for mode in ("interp", "opt-interp"):
        run = batch / "CL1" / f"CL1-{mode}"
        run.mkdir(parents=True)
        (run / f"CL1-{mode}.xyz").write_text("", encoding="utf-8")
    legacy = batch / "CL1" / "working-CL1"
    legacy.mkdir(parents=True)

    assert sorted(p.name for p in _discover_run_dirs(batch)) == ["CL1-interp", "CL1-opt-interp"]
    # Only the second set -- the first is mid-flight and must not be resubmitted.
    picked = _discover_run_dirs(batch, {"CL1-opt-interp"})
    assert [p.name for p in picked] == ["CL1-opt-interp"]
