"""CLI test: ``xas-run-batch --ligand-dir`` hands a run its own ligand database.

The database is copied into each run dir's ``interp-ligands/``, which the
Hessian-interpolation stage prefers over the packaged ligand files -- so every
route that regenerates the CORVUS wrapper later (reruns, submit-corvus) uses the
same database without being told about it. A missing database must stop the
batch at submission, before any run dir is written.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from xas_pipeline import resources

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"
ID_NAME = "2j6a_ZN_homo_d2.60_cluster1"


def _batch(tmp_path: Path, run_cli, *extra: str):
    in_root = tmp_path / "xyz_files"
    in_root.mkdir(exist_ok=True)
    shutil.copy2(FIXTURE_XYZ, in_root / FIXTURE_XYZ.name)
    out_root = tmp_path / "batch-out"
    result = run_cli(
        "run-batch-pipeline.py",
        str(in_root), "--out-dir", str(out_root), "--scheduler", "slurm", "--no-submit",
        *extra,
    )
    return result, out_root


def _cys_only_db(root: Path) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(resources.interp_ligand_root() / "ZnCys.interp", root / "ZnCys.interp")
    return root


def test_per_structure_database_is_copied_into_the_run_dir(tmp_path, run_cli):
    _cys_only_db(tmp_path / "db" / ID_NAME)
    result, out_root = _batch(
        tmp_path, run_cli, "--interp-raw", "--ligand-dir", str(tmp_path / "db" / "{stem}")
    )
    assert result.returncode == 0, result.stderr

    run_ligands = out_root / ID_NAME / f"{ID_NAME}-asis-spring" / "interp-ligands"
    assert sorted(p.name for p in run_ligands.glob("*.interp")) == ["ZnCys.interp"]


def test_resubmitting_replaces_the_previous_database(tmp_path, run_cli):
    first = _cys_only_db(tmp_path / "db-first")
    shutil.copy2(resources.interp_ligand_root() / "ZnHis.interp", first / "ZnHis.interp")
    _batch(tmp_path, run_cli, "--interp-raw", "--ligand-dir", str(first))

    second = _cys_only_db(tmp_path / "db-second")
    result, out_root = _batch(tmp_path, run_cli, "--interp-raw", "--ligand-dir", str(second))
    assert result.returncode == 0, result.stderr

    run_ligands = out_root / ID_NAME / f"{ID_NAME}-asis-spring" / "interp-ligands"
    assert sorted(p.name for p in run_ligands.glob("*.interp")) == ["ZnCys.interp"]


def test_missing_database_stops_the_batch_before_anything_is_written(tmp_path, run_cli):
    (tmp_path / "db").mkdir()
    result, out_root = _batch(
        tmp_path, run_cli, "--interp-raw", "--ligand-dir", str(tmp_path / "db" / "{stem}")
    )
    assert result.returncode != 0
    assert ID_NAME in result.stderr
    assert not (out_root / ID_NAME).exists()


def test_ligand_dir_is_refused_for_modes_with_an_orca_hessian(tmp_path, run_cli):
    db = _cys_only_db(tmp_path / "db")
    result, out_root = _batch(tmp_path, run_cli, "--free", "--ligand-dir", str(db))
    assert result.returncode != 0
    # Message reworded in WIP 77e7365 ("...only applies to spring modes, not <mode>").
    assert "only applies to spring modes, not no-constraints" in result.stderr
    assert not (out_root / ID_NAME).exists()


def test_without_ligand_dir_no_run_database_is_written(tmp_path, run_cli):
    result, out_root = _batch(tmp_path, run_cli, "--interp-raw")
    assert result.returncode == 0, result.stderr
    assert not (out_root / ID_NAME / f"{ID_NAME}-asis-spring" / "interp-ligands").exists()
