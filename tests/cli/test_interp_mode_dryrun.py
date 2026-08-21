"""CLI test: the interp-family modes, and mode coexistence.

Three things are checked end to end without touching a scheduler:

1. ``--interp`` (mode ``interp-hopt``) generates an ORCA input that optimizes the
   hydrogens and has no ``! AnFreq``, plus a CORVUS wrapper that builds the
   Hessian from spring models before prepare-corvus.
2. ``--interp-raw`` (mode ``interp-raw``) generates *no* ORCA input and no ORCA
   job script at all, writes the geometry of record itself, and still gets a
   wrapper that interpolates the Hessian.
3. Several modes run from the *same* starting structure land in separate run dirs
   under one group dir, so nothing is overwritten.

Every other mode's wrapper leaves the Hessian step a no-op and keeps its AnFreq.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"
ID_NAME = "2j6a_ZN_homo_d2.60_cluster1"

# flag -> run-dir mode suffix. --interp selects the hydrogen-optimizing variant;
# the older single-point `interp` mode is still recognized for the run dirs
# already on disk but no flag produces it.
MODES = {
    "--interp": "interp-hopt",
    "--interp-raw": "interp-raw",
    "--free": "no-constraints",
    None: "ca-fixed",
}

# Modes that run no ORCA stage, so no ORCA input or job script is written for them.
NO_ORCA = {"interp-raw"}

# Modes whose Hessian is interpolated from the ligand spring models.
SPRING_HESSIAN = {"interp-hopt", "interp-raw"}


@pytest.fixture(scope="module")
def multi_mode_batch(tmp_path_factory):
    """Run the same XYZ through every mode above into one batch root."""
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

    # Each ORCA-running mode holds its own input, named for its run id, so no
    # mode's artifacts can collide with another's.
    for mode in MODES.values():
        orca_input = _run_dir(out_root, mode) / f"{ID_NAME}-{mode}.in"
        assert orca_input.is_file() is (mode not in NO_ORCA), (
            f"{mode}: unexpected ORCA input presence ({orca_input})"
        )


def _orca_directives(text: str) -> list[str]:
    """The ``!`` keyword lines ORCA actually acts on (not the ``#`` comments).

    Matters here because the interp templates *explain* in a comment that they
    omit "! AnFreq", so a plain substring search would find the word either way.
    """
    return [line.strip() for line in text.splitlines() if line.strip().startswith("!")]


def test_interp_hopt_input_optimizes_hydrogens_without_anfreq(multi_mode_batch):
    """The whole point of the mode: relax the protons, don't compute the Hessian."""
    path = _run_dir(multi_mode_batch["out_root"], "interp-hopt") / f"{ID_NAME}-interp-hopt.in"
    text = path.read_text()
    directives = _orca_directives(text)

    assert not any("anfreq" in line.lower() for line in directives), directives
    # It is a real optimization at the same level of theory, restricted to H.
    assert any("TightOPT" in line for line in directives), directives
    assert any("PBE0" in line for line in directives), directives
    assert "optimizehydrogens true" in text.lower()
    assert "*xyzfile" in text


def test_other_modes_keep_anfreq(multi_mode_batch):
    for mode in ("ca-fixed", "no-constraints"):
        text = (_run_dir(multi_mode_batch["out_root"], mode) / f"{ID_NAME}-{mode}.in").read_text()
        directives = _orca_directives(text)
        assert any("anfreq" in line.lower() for line in directives), f"{mode} lost its AnFreq step"


def test_interp_raw_writes_no_orca_stage(multi_mode_batch):
    """No ORCA input, no ORCA job script -- nothing downstream can submit one."""
    run_dir = _run_dir(multi_mode_batch["out_root"], "interp-raw")
    assert run_dir.is_dir()

    assert not (run_dir / f"{ID_NAME}-interp-raw.in").exists()
    assert list(run_dir.glob("generated-*-orca.script")) == []


def test_interp_raw_writes_the_geometry_of_record(multi_mode_batch):
    """``<run_id>.xyz`` puts the mode on select_run_xyz's primary path.

    Without it the geometry would be resolved by the fallback branch, which
    filters the directory by exclusion -- the same fallback whose mtime tiebreak
    once fed CORVUS the unoptimized structure.
    """
    from xas_pipeline.chem import xyz as chem_xyz

    run_dir = _run_dir(multi_mode_batch["out_root"], "interp-raw")
    run_id = f"{ID_NAME}-interp-raw"
    geometry = run_dir / f"{run_id}.xyz"

    assert geometry.is_file()
    assert chem_xyz.select_run_xyz(run_dir, run_id) == geometry

    # Same atoms as the input it was derived from: this is a copy, not a rewrite.
    assert geometry.read_text().splitlines()[0].strip() == (
        (run_dir / f"{ID_NAME}.xyz").read_text().splitlines()[0].strip()
    )


def test_interp_raw_records_the_absent_orca_stage(multi_mode_batch):
    """batch-jobs.log must say ORCA was skipped, not just omit the line.

    An absent entry is indistinguishable from a submission that failed to log.
    """
    out_root = multi_mode_batch["out_root"]
    log = (out_root / "batch-jobs.log").read_text()
    assert f"orca-{ID_NAME}-interp-raw" in log


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


@pytest.mark.parametrize("mode", sorted(SPRING_HESSIAN))
def test_spring_hessian_wrappers_build_the_hessian_before_corvus(multi_mode_batch, mode):
    lines = _executable_lines(multi_mode_batch["out_root"], mode)
    assert f"OPTIMIZATION_MODE={mode}" in lines

    interp_at = [i for i, line in enumerate(lines) if "stages.interp_hessian" in line]
    corvus_at = [i for i, line in enumerate(lines) if "stages.corvus_prep" in line]
    assert interp_at, f"{mode} wrapper never runs the Hessian interpolation"
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
    # rather than which, since every mode writes to the same batch root.
    assert "optimization_mode:" in state
