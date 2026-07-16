"""Integration golden test: `prepare-corvus.py` up to the `.dym`.

prepare-corvus converts a completed ORCA run (.hess + .xyz + _trj.xyz) into FEFF
inputs: it writes the standardized ``corvus-begin-<id>.xyz``, the dynamical
matrix ``<id>.dym``, and per-mode CORVUS ``.in`` inputs + job scripts. The one
step we cannot run offline is the ``dym2feffinp`` binary (FEFF10, not installed
in CI), which centers the .dym on the Zn absorber. So we stub it via
``DYM2FEFFINP_BIN`` (resolved first, before PATH/fallbacks) and snapshot every
deterministic artifact produced *up to and around* that call — the boundary the
refactor is allowed to freeze. The stubbed ``corvus-<id>.dym`` output is NOT
snapshotted (its contents are the binary's job, not ours).

Snapshotting these locks the hess->dym numerical transform and the CORVUS
template-filling before the reorg moves them into ``chem/hessian.py`` etc.

Regenerate after an intentional behavior change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_prepare_corvus_dryrun.py

Review the resulting diff carefully — that diff IS the behavior change.
"""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
RUN_ID = "2j6a_ZN_homo_d2.60_cluster1"
FIXTURE_RUN = FIXTURES / "corvus_run" / RUN_ID
GOLDEN_DIR = FIXTURES / "golden" / "prepare-corvus"

UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"

# Deterministic artifacts produced up to the .dym boundary (byte-snapshotted
# after path normalization). Excludes corvus-<id>.dym (the stubbed binary output)
# and the copied-in inputs (.hess/.xyz/_trj.xyz).
GOLDEN_NAMES = (
    f"corvus-begin-{RUN_ID}.xyz",
    f"{RUN_ID}.dym",
    f"corvus-{RUN_ID}-exafs.in",
    f"corvus-{RUN_ID}-xanes.in",
    "corvus-job-exafs.script",
    "corvus-job-xanes.script",
)

# A dym2feffinp that only needs to satisfy check=True and create its --d output,
# so prepare-corvus main() runs to completion. We never snapshot that output.
_STUB = """#!/usr/bin/env bash
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--d" ]; then out="$2"; shift 2; continue; fi
  shift
done
: > "$out"
"""


def _normalize(text: str, *, run_dir: Path, repo_root: Path) -> str:
    return text.replace(str(run_dir), "<RUN>").replace(str(repo_root), "<REPO>")


@pytest.fixture(scope="module")
def corvus_run(tmp_path_factory, repo_root):
    tmp = tmp_path_factory.mktemp("corvus")
    run_dir = tmp / RUN_ID
    shutil.copytree(FIXTURE_RUN, run_dir)

    stub = tmp / "dym2feffinp"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["DYM2FEFFINP_BIN"] = str(stub)  # resolved before PATH/fallbacks

    result = subprocess.run(
        [sys.executable, "-m", "xas_pipeline.stages.corvus_prep", str(run_dir),
         "--scheduler", "slurm"],
        capture_output=True, text=True, env=env,
    )
    return {"result": result, "run_dir": run_dir, "repo_root": repo_root}


def test_exits_clean(corvus_run):
    result = corvus_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_dym_and_begin_xyz_exist(corvus_run):
    run_dir = corvus_run["run_dir"]
    assert (run_dir / f"{RUN_ID}.dym").is_file()
    assert (run_dir / f"corvus-begin-{RUN_ID}.xyz").is_file()


def test_artifacts_match_golden(corvus_run):
    run_dir = corvus_run["run_dir"]
    mismatches = []
    missing = []
    for name in GOLDEN_NAMES:
        produced = run_dir / name
        assert produced.is_file(), f"expected artifact not produced: {name}"
        actual = _normalize(
            produced.read_text(encoding="utf-8"),
            run_dir=run_dir,
            repo_root=corvus_run["repo_root"],
        )
        golden_path = GOLDEN_DIR / name
        if UPDATE:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(actual, encoding="utf-8")
            continue
        if not golden_path.is_file():
            missing.append(name)
        elif actual != golden_path.read_text(encoding="utf-8"):
            mismatches.append(name)
    if UPDATE:
        pytest.skip(f"GOLDEN_UPDATE=1: wrote {len(GOLDEN_NAMES)} golden snapshot(s)")
    assert not missing, f"missing goldens (run GOLDEN_UPDATE=1 to create): {missing}"
    assert not mismatches, f"content drift in: {mismatches}"
