"""Characterization test: `script-cleanup-calc-artifacts.py --execute`.

Covers the deny-list deletion behavior:
  * ORCA scratch (.densities/.engrad/.cpcm) and FEFF scratch (*.bin/gg.dat/
    dmdw.out) are deleted; real outputs are kept.
  * FEFF scratch is pruned in BOTH the combined-xas nested layout
    (Corvus1Zn_<idx>_FEFF/{xanes,exafs}/) and the legacy flat layout.
  * Legacy per-mode trees (Corvus3_cfavg_xanes/_exafs) and component copies
    (xanes-<id>.dat/exafs-<id>.dat) are removed ONLY when the combined-xas
    result exists for that cluster; otherwise they are kept (guard).

Two clusters exercise the guard both ways:
  run1 has the combined-xas result -> legacy modes are superseded and removed.
  run2 has no xas result -> legacy modes are kept (only their scratch is pruned).
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cleanup_run(tmp_path_factory):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("clean") / "batch-out"

    # --- run1: combined-xas present -> legacy xanes/exafs are superseded ------
    r1 = batch / "run1"
    out1 = r1 / "output-run1"
    out1.mkdir(parents=True)
    (out1 / "keep.dat").write_text("keep", encoding="utf-8")
    (out1 / "xas-run1.dat").write_text("combined", encoding="utf-8")        # guard: xas exists
    (out1 / "xmu-xanes-run1.dat").write_text("keep", encoding="utf-8")      # new-style -> keep
    (out1 / "xanes-run1.dat").write_text("legacy", encoding="utf-8")        # superseded -> remove
    (out1 / "exafs-run1.dat").write_text("legacy", encoding="utf-8")        # superseded -> remove

    w1 = r1 / "working-run1"
    w1.mkdir()
    for name in ("a.densities", "b.engrad", "c.cpcm"):  # ORCA scratch
        (w1 / name).write_text("scratch", encoding="utf-8")
    (w1 / "run1.hess").write_text("hess", encoding="utf-8")  # keeper

    # Combined-xas tree: scratch nested in xanes/ + exafs/ subdirs.
    xas = w1 / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    xas_subdirs = [xas / "xanes", xas / "exafs"]
    for sub in xas_subdirs:
        sub.mkdir(parents=True)
        for name in ("gg.dat", "feff.bin", "apot.bin"):  # nested FEFF scratch
            (sub / name).write_text("scratch", encoding="utf-8")
        (sub / "xmu.dat").write_text("spectrum", encoding="utf-8")  # keeper

    # Legacy per-mode trees -> removed wholesale (xas present).
    legacy_trees_r1 = []
    for mode in ("xanes", "exafs"):
        leg = w1 / f"Corvus3_cfavg_{mode}" / "Corvus1Zn_FEFF"
        leg.mkdir(parents=True)
        (leg / "xmu.dat").write_text("legacy-spectrum", encoding="utf-8")
        legacy_trees_r1.append(w1 / f"Corvus3_cfavg_{mode}")

    # --- run2: NO combined-xas -> legacy kept (guard), scratch still pruned ---
    r2 = batch / "run2"
    out2 = r2 / "output-run2"
    out2.mkdir(parents=True)
    (out2 / "keep.dat").write_text("keep", encoding="utf-8")
    (out2 / "xanes-run2.dat").write_text("legacy", encoding="utf-8")  # kept (no xas)

    w2 = r2 / "working-run2"
    w2.mkdir()
    leg2 = w2 / "Corvus3_cfavg_exafs" / "Corvus1Zn_FEFF"
    leg2.mkdir(parents=True)
    for name in ("gg.dat", "x.bin"):  # flat-layout FEFF scratch
        (leg2 / name).write_text("scratch", encoding="utf-8")
    (leg2 / "xmu.dat").write_text("legacy-spectrum", encoding="utf-8")  # keeper (tree not removed)

    result = run_script("script-cleanup-calc-artifacts.py", str(batch), "--execute")
    return {
        "result": result,
        "batch": batch,
        "w1": w1,
        "out1": out1,
        "xas_subdirs": xas_subdirs,
        "legacy_trees_r1": legacy_trees_r1,
        "leg2": leg2,
        "out2": out2,
    }


def test_exits_clean(cleanup_run):
    result = cleanup_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_orca_scratch_deleted(cleanup_run):
    w1 = cleanup_run["w1"]
    for gone in ("a.densities", "b.engrad", "c.cpcm"):
        assert not (w1 / gone).exists(), f"ORCA scratch not deleted: {gone}"


def test_nested_xas_scratch_deleted(cleanup_run):
    """Combined-xas scratch lives in xanes/ and exafs/ subdirs of the FEFF dir."""
    for sub in cleanup_run["xas_subdirs"]:
        for gone in ("gg.dat", "feff.bin", "apot.bin"):
            assert not (sub / gone).exists(), f"nested FEFF scratch not deleted: {sub.name}/{gone}"
        assert (sub / "xmu.dat").is_file(), f"nested deliverable deleted: {sub.name}/xmu.dat"


def test_superseded_removed_when_xas_present(cleanup_run):
    for tree in cleanup_run["legacy_trees_r1"]:
        assert not tree.exists(), f"superseded legacy tree not removed: {tree.name}"
    for gone in ("xanes-run1.dat", "exafs-run1.dat"):
        assert not (cleanup_run["out1"] / gone).exists(), f"superseded component not removed: {gone}"
    # New-style / combined deliverables must survive.
    for keep in ("xas-run1.dat", "xmu-xanes-run1.dat", "keep.dat"):
        assert (cleanup_run["out1"] / keep).is_file(), f"deliverable deleted: {keep}"
    assert (cleanup_run["w1"] / "run1.hess").is_file()


def test_superseded_kept_without_xas(cleanup_run):
    """No combined xas -> legacy tree + component copy survive; scratch still pruned."""
    leg2 = cleanup_run["leg2"]
    assert leg2.is_dir(), "legacy exafs tree wrongly removed for a cluster with no xas"
    assert (leg2 / "xmu.dat").is_file(), "legacy spectrum wrongly removed for a cluster with no xas"
    assert (cleanup_run["out2"] / "xanes-run2.dat").is_file(), "legacy component wrongly removed (no xas)"
    for gone in ("gg.dat", "x.bin"):
        assert not (leg2 / gone).exists(), f"flat-layout FEFF scratch not deleted: {gone}"
