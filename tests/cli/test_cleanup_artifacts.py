"""Characterization test: `script-cleanup-calc-artifacts.py --execute`.

Pins the deny-list deletion behavior before the reorg rewires this script
through xas_pipeline.layout (and before the copy-back change makes the ORCA
scratch pass mostly a no-op for new runs). Asserts the regenerable ORCA/FEFF
scratch is deleted while real outputs are kept, in a cluster that has an
output-* dir (so it is not skipped).
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cleanup_run(tmp_path_factory):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("clean") / "batch-out"
    cluster = batch / "run1"
    (cluster / "output-run1").mkdir(parents=True)
    (cluster / "output-run1" / "keep.dat").write_text("keep", encoding="utf-8")

    working = cluster / "working-run1"
    working.mkdir(parents=True)
    for name in ("a.densities", "b.engrad", "c.cpcm"):  # ORCA scratch
        (working / name).write_text("scratch", encoding="utf-8")
    (working / "run1.hess").write_text("hess", encoding="utf-8")  # keeper

    # Legacy flat layout: scratch directly in the FEFF dir.
    feff = working / "Corvus3_cfavg_exafs" / "Corvus1Zn_FEFF"
    feff.mkdir(parents=True)
    for name in ("gg.dat", "x.bin"):  # FEFF scratch
        (feff / name).write_text("scratch", encoding="utf-8")
    (feff / "xmu.dat").write_text("spectrum", encoding="utf-8")  # keeper

    # Combined-xas layout: scratch nested one level deeper in xanes/ and exafs/
    # subdirs of Corvus1Zn_<idx>_FEFF (idx varies per structure -> globbed).
    xas_feff = working / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    xas_subdirs = [xas_feff / "xanes", xas_feff / "exafs"]
    for sub in xas_subdirs:
        sub.mkdir(parents=True)
        for name in ("gg.dat", "feff.bin", "apot.bin"):  # nested FEFF scratch
            (sub / name).write_text("scratch", encoding="utf-8")
        (sub / "xmu.dat").write_text("spectrum", encoding="utf-8")  # keeper

    result = run_script("script-cleanup-calc-artifacts.py", str(batch), "--execute")
    return {
        "result": result,
        "batch": batch,
        "working": working,
        "feff": feff,
        "xas_subdirs": xas_subdirs,
    }


def test_exits_clean(cleanup_run):
    result = cleanup_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_scratch_deleted(cleanup_run):
    working, feff = cleanup_run["working"], cleanup_run["feff"]
    for gone in ("a.densities", "b.engrad", "c.cpcm"):
        assert not (working / gone).exists(), f"ORCA scratch not deleted: {gone}"
    for gone in ("gg.dat", "x.bin"):
        assert not (feff / gone).exists(), f"FEFF scratch not deleted: {gone}"


def test_nested_xas_scratch_deleted(cleanup_run):
    """Combined-xas scratch lives in xanes/ and exafs/ subdirs of the FEFF dir."""
    for sub in cleanup_run["xas_subdirs"]:
        for gone in ("gg.dat", "feff.bin", "apot.bin"):
            assert not (sub / gone).exists(), f"nested FEFF scratch not deleted: {sub.name}/{gone}"


def test_outputs_kept(cleanup_run):
    working, feff = cleanup_run["working"], cleanup_run["feff"]
    assert (working / "run1.hess").is_file()
    assert (feff / "xmu.dat").is_file()
    assert (cleanup_run["batch"] / "run1" / "output-run1" / "keep.dat").is_file()
    # Nested deliverables must survive the recursive scratch sweep.
    for sub in cleanup_run["xas_subdirs"]:
        assert (sub / "xmu.dat").is_file(), f"nested deliverable deleted: {sub.name}/xmu.dat"
