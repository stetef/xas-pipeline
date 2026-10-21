"""Golden snapshots of everything ``xas-run-batch --no-submit`` writes, per mode.

One structure, every optimization mode, both schedulers: ORCA inputs, job
scripts, CORVUS wrappers, postprocess/watchdog scripts, cleaned geometries and
the batch log. Plus the staged prepare-orca variants (``--pre``, ``--anfreq``).
These pin the ORCA/CORVUS inputs the pipeline hands to the codes, so refactors
can be checked for "inputs identical".

Regenerate after an intentional change with ``GOLDEN_UPDATE=1`` and review the diff.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES, GOLDEN_ROOT, REPO_ROOT, assert_matches_golden, normalize, run_module

XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"

# run-batch flag -> run-dir mode name (the golden dir name).
MODE_FLAGS = {
    None: "caopt-anfreq",
    "--H": "hopt-anfreq",
    "--single": "carved-anfreq",
    "--free": "no-constraints",
    "--backbone": "backbone",
    "--quick": "quick",
    "--quick-ca-fixed": "quick-ca-fixed",
    "--xtb-free": "xtb-free",
    "--xtb-constrained": "xtb-constrained",
    "--interp": "hopt-spring",
    "--interp-raw": "asis-spring",
    "--caopt-spring": "caopt-spring",
}

STAGED = {
    "pre-quick-ca-fixed": ["--pre", "quick-ca-fixed"],
    "anfreq-on-interp": ["--anfreq", "--interp"],
    "no-anfreq": ["--no-anfreq"],
}

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)?")
# Point charges for --backbone (the content is copied verbatim, never parsed).
_PC = "2\n0.5 1.0 2.0 3.0\n-0.5 4.0 5.0 6.0\n"


def _inputs(tmp: Path) -> Path:
    in_root = tmp / "xyz"
    in_root.mkdir()
    shutil.copy2(XYZ, in_root / XYZ.name)
    (in_root / XYZ.with_suffix(".pc").name).write_text(_PC)
    return in_root


def _collect(out_root: Path, in_root: Path) -> dict[str, str]:
    paths = {"OUT": out_root, "IN": in_root, "REPO": REPO_ROOT}
    return {
        str(path.relative_to(out_root)): _TIMESTAMP.sub(
            "<UTC>", normalize(path.read_text(encoding="utf-8"), paths)
        )
        for path in sorted(out_root.rglob("*"))
        if path.is_file() and not path.name.startswith("pipeline-state-")
    }


@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
@pytest.mark.parametrize("flag", list(MODE_FLAGS), ids=lambda f: MODE_FLAGS[f])
def test_run_batch_outputs_match_golden(tmp_path, scheduler, flag):
    in_root = _inputs(tmp_path)
    out_root = tmp_path / "batch-out"
    args = [str(in_root), "--out-dir", str(out_root), "--scheduler", scheduler, "--no-submit"]
    result = run_module("xas_pipeline.orchestrate", *args, *([flag] if flag else []))
    assert result.returncode == 0, result.stderr

    golden = GOLDEN_ROOT / "modes" / scheduler / MODE_FLAGS[flag]
    assert_matches_golden(golden, _collect(out_root, in_root))


@pytest.mark.parametrize("name", list(STAGED))
def test_staged_prepare_orca_matches_golden(tmp_path, name):
    in_root = _inputs(tmp_path)
    out_root = tmp_path / "out"
    result = run_module(
        "xas_pipeline.stages.orca_prep", str(in_root), "--out-dir", str(out_root),
        "--scheduler", "slurm", "--dry-run", *STAGED[name],
    )
    assert result.returncode == 0, result.stderr
    assert_matches_golden(GOLDEN_ROOT / "staged" / name, _collect(out_root, in_root))
