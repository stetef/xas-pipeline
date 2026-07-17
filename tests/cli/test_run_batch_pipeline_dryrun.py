"""Integration golden test: `run-batch-pipeline.py --no-submit` (slurm).

This is the pipeline's submission spine. In --no-submit mode it runs the real
prepare-orca step and generates every artifact that would otherwise be submitted
(ORCA job scripts, per-mode CORVUS wrapper scripts, the batch postprocess
script, ORCA .in inputs, cleaned xyz + comments sidecars, batch-jobs.log) but
touches no scheduler. Snapshotting those artifacts locks the end-to-end behavior
of BOTH run-batch-pipeline and prepare-orca before the refactor.

The generated files embed absolute paths but no timestamps, so after path
normalization they are byte-for-byte deterministic and safe to snapshot. The
pipeline state log DOES contain timestamps and a nondeterministically-ordered
prepare-orca stdout block, so it is checked by invariant assertions instead of a
byte snapshot.

Regenerate the golden snapshots after an intentional behavior change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_run_batch_pipeline_dryrun.py

Review the resulting diff carefully — that diff IS the behavior change.
"""

import os
import re
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_XYZ = FIXTURES / "xyz_files"
GOLDEN_DIR = FIXTURES / "golden" / "rbp-slurm"

# Fixed output dir name so filenames derived from the batch name
# (generated-postprocess-<name>.script, pipeline-state-<name>.log) are stable.
OUT_DIRNAME = "batch-out"

# Deterministic artifacts worth locking byte-for-byte (after path normalization).
GOLDEN_NAME_SUFFIXES = (".script", ".in", "_comments.txt", "_clean.xyz")
GOLDEN_EXACT_NAMES = ("batch-jobs.log",)

UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")


def _normalize(text: str, *, out_root: Path, in_root: Path, repo_root: Path) -> str:
    """Replace machine-specific paths and UTC timestamps with stable tokens."""
    text = text.replace(str(out_root), "<OUT>")
    text = text.replace(str(in_root), "<IN>")
    text = text.replace(str(repo_root), "<REPO>")
    return _TIMESTAMP_RE.sub("<UTC>", text)


def _is_golden_artifact(name: str) -> bool:
    return name in GOLDEN_EXACT_NAMES or name.endswith(GOLDEN_NAME_SUFFIXES)


def _collect(out_root: Path) -> list[Path]:
    return sorted(
        p.relative_to(out_root)
        for p in out_root.rglob("*")
        if p.is_file() and _is_golden_artifact(p.name)
    )


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory, repo_root):
    """Run the dry-run batch once per module into an isolated tmp tree."""
    from conftest import run_script  # module-scoped: can't use the function fixture

    tmp = tmp_path_factory.mktemp("rbp")
    in_root = tmp / "xyz_files"
    shutil.copytree(FIXTURE_XYZ, in_root)
    out_root = tmp / OUT_DIRNAME

    result = run_script(
        "run-batch-pipeline.py",
        str(in_root),
        "--out-dir",
        str(out_root),
        "--scheduler",
        "slurm",
        "--no-submit",
    )
    return {
        "result": result,
        "out_root": out_root,
        "in_root": in_root,
        "repo_root": repo_root,
    }


def test_dryrun_exits_clean(batch_run):
    result = batch_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Prepared runs: 2" in result.stdout
    assert "Postprocess job: NO_SUBMIT" in result.stdout


def test_generated_file_set_matches_golden(batch_run):
    produced = set(_collect(batch_run["out_root"]))
    if UPDATE:
        pytest.skip("GOLDEN_UPDATE=1: snapshots written by test_generated_files_match_golden")
    golden = {
        p.relative_to(GOLDEN_DIR)
        for p in GOLDEN_DIR.rglob("*")
        if p.is_file()
    }
    assert produced == golden, (
        f"generated artifact set drifted from golden.\n"
        f"  only produced: {sorted(map(str, produced - golden))}\n"
        f"  only golden:   {sorted(map(str, golden - produced))}"
    )


def test_generated_files_match_golden(batch_run):
    out_root = batch_run["out_root"]
    produced = _collect(out_root)
    assert produced, "no artifacts were generated"

    mismatches = []
    for rel in produced:
        actual = _normalize(
            (out_root / rel).read_text(encoding="utf-8"),
            out_root=out_root,
            in_root=batch_run["in_root"],
            repo_root=batch_run["repo_root"],
        )
        golden_path = GOLDEN_DIR / rel
        if UPDATE:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(actual, encoding="utf-8")
            continue
        assert golden_path.is_file(), (
            f"missing golden for {rel}; run with GOLDEN_UPDATE=1 to create it"
        )
        if actual != golden_path.read_text(encoding="utf-8"):
            mismatches.append(str(rel))
    if UPDATE:
        pytest.skip(f"GOLDEN_UPDATE=1: wrote {len(produced)} golden snapshot(s)")
    assert not mismatches, f"content drift in: {mismatches}"


def test_batch_log_records_dryrun_skips(batch_run):
    log = (batch_run["out_root"] / "batch-jobs.log").read_text(encoding="utf-8")
    assert "prepare-orca\tSUCCEEDED" in log
    for run_id in ("2j6a_ZN_homo_d2.60_cluster1", "5c1z_ZN_homo_d2.60_cluster15"):
        assert f"orca-{run_id}\tSKIPPED" in log
        assert f"corvus-xas-{run_id}\tSKIPPED" in log
    assert "postprocess-batch-out\tSKIPPED" in log


def test_state_file_invariants(batch_run):
    state = next(batch_run["out_root"].glob("pipeline-state-*.log")).read_text(encoding="utf-8")
    assert "scheduler:            slurm" in state
    assert "optimization_mode:    ca-fixed" in state
    assert "corvus_mode:          xas" in state
    assert "postprocess_job_id:   NO_SUBMIT" in state
    for run_id in ("2j6a_ZN_homo_d2.60_cluster1", "5c1z_ZN_homo_d2.60_cluster15"):
        assert run_id in state
