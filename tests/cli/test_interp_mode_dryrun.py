"""CLI test: ``run-batch-pipeline.py --interp`` and mode coexistence.

Two things are checked end to end without touching a scheduler:

1. ``--interp`` generates an ORCA input with no ``! AnFreq`` and a CORVUS wrapper
   that builds the Hessian from spring models before prepare-corvus -- while
   every other mode's wrapper leaves the Hessian step a no-op.
2. Several modes run from the *same* starting structure land in separate run
   dirs under one group dir, so nothing is overwritten.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"
ID_NAME = "2j6a_ZN_homo_d2.60_cluster1"
MODES = {"--interp": "interp", "--free": "no-constraints", None: "ca-fixed"}


@pytest.fixture(scope="module")
def multi_mode_batch(tmp_path_factory):
    """Run the same XYZ through three modes into one batch root."""
    from conftest import run_script  # module-scoped: can't use the function fixture

    tmp = tmp_path_factory.mktemp("interp-modes")
    in_root = tmp / "xyz_files"
    in_root.mkdir()
    shutil.copy2(FIXTURE_XYZ, in_root / FIXTURE_XYZ.name)
    out_root = tmp / "batch-out"

    results = {}
    for flag, mode in MODES.items():
        args = [str(in_root), "--out-dir", str(out_root), "--scheduler", "slurm", "--no-submit"]
        if flag:
            args.append(flag)
        results[mode] = run_script("run-batch-pipeline.py", *args)
    return {"out_root": out_root, "results": results}


def _run_dir(out_root: Path, mode: str) -> Path:
    return out_root / ID_NAME / f"{ID_NAME}-{mode}"


def test_every_mode_exits_clean(multi_mode_batch):
    for mode, result in multi_mode_batch["results"].items():
        assert result.returncode == 0, f"{mode} failed:\n{result.stderr}"


def test_modes_coexist_without_overwriting(multi_mode_batch):
    out_root = multi_mode_batch["out_root"]
    group = out_root / ID_NAME
    assert group.is_dir()

    present = sorted(p.name for p in group.iterdir() if p.is_dir())
    assert present == sorted(f"{ID_NAME}-{mode}" for mode in MODES.values())

    # Each run dir holds its own ORCA input, named for its run id, so no mode's
    # artifacts can collide with another's.
    for mode in MODES.values():
        assert (_run_dir(out_root, mode) / f"{ID_NAME}-{mode}.in").is_file()


def _orca_directives(text: str) -> list[str]:
    """The ``!`` keyword lines ORCA actually acts on (not the ``#`` comments).

    Matters here because the interp template *explains* in a comment that it
    omits "! AnFreq", so a plain substring search would find the word either way.
    """
    return [line.strip() for line in text.splitlines() if line.strip().startswith("!")]


def test_interp_input_has_no_anfreq(multi_mode_batch):
    """The whole point of the mode: ORCA computes the energy, not the Hessian."""
    text = (_run_dir(multi_mode_batch["out_root"], "interp") / f"{ID_NAME}-interp.in").read_text()
    directives = _orca_directives(text)
    assert not any("anfreq" in line.lower() for line in directives), directives
    # It is still a real single-point input at the same level of theory.
    assert any("PBE0" in line for line in directives)
    assert "*xyzfile" in text


def test_other_modes_keep_anfreq(multi_mode_batch):
    for mode in ("ca-fixed", "no-constraints"):
        text = (_run_dir(multi_mode_batch["out_root"], mode) / f"{ID_NAME}-{mode}.in").read_text()
        directives = _orca_directives(text)
        assert any("anfreq" in line.lower() for line in directives), f"{mode} lost its AnFreq step"


def _executable_lines(out_root: Path, mode: str) -> list[str]:
    """The wrapper lines that actually run: no blanks, comments, or echoed prose.

    The failure diagnostics inside the wrapper *mention* the interp_hessian
    command so a human can rerun it by hand, so matching the raw text would say
    every wrapper runs it.
    """
    wrappers = list(_run_dir(out_root, mode).glob("generated-*-corvus-*-wrapper.script"))
    assert len(wrappers) == 1, f"expected one wrapper for {mode}, got {wrappers}"
    return [
        line.strip()
        for line in wrappers[0].read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "echo "))
    ]


def test_interp_wrapper_builds_the_hessian_before_corvus(multi_mode_batch):
    lines = _executable_lines(multi_mode_batch["out_root"], "interp")
    assert "OPTIMIZATION_MODE=interp" in lines

    interp_at = [i for i, line in enumerate(lines) if "stages.interp_hessian" in line]
    corvus_at = [i for i, line in enumerate(lines) if "stages.corvus_prep" in line]
    assert interp_at, "interp wrapper never runs the Hessian interpolation"
    assert corvus_at, "wrapper never runs prepare-corvus"
    # Ordering is load-bearing: prepare-corvus fails fast on a missing .hess, so
    # the interpolation has to have produced it by then.
    assert interp_at[0] < corvus_at[0]


def test_non_interp_wrappers_leave_the_hessian_step_a_noop(multi_mode_batch):
    for mode in ("ca-fixed", "no-constraints"):
        lines = _executable_lines(multi_mode_batch["out_root"], mode)
        assert f"OPTIMIZATION_MODE={mode}" in lines
        assert not any("stages.interp_hessian" in line for line in lines), (
            f"{mode} should use ORCA's own Hessian"
        )
        # The injected step is present but inert, so the wrapper stays uniform.
        assert "true" in lines


def test_state_file_records_the_interp_mode(multi_mode_batch):
    state = next(multi_mode_batch["out_root"].glob("pipeline-state-*.log")).read_text()
    # The last run wins the shared state file; assert the mode is recorded at all
    # rather than which, since all three write to the same batch root.
    assert "optimization_mode:" in state
