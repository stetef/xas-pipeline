"""Tests for the ``--interp`` Hessian route (spring models instead of AnFreq).

The numerics are vendored from DW_Interpolation and are not re-derived here.
What these tests pin is the contract the pipeline depends on:

* the interpolated ``.hess`` is readable by the *same* parser used for a real
  ORCA Hessian, with the shape and symmetry a Hessian must have;
* it describes the same atoms, in the same order, as the geometry CORVUS will
  write into the ``.dym`` -- if that ever drifts, FEFF silently matches the
  wrong atoms;
* the physics that must hold regardless of the spring constants: the acoustic
  sum rule, and translational invariance.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from xas_pipeline import resources
from xas_pipeline.chem import xyz as chem_xyz
from xas_pipeline.chem.hessian import read_orca_hessian
from xas_pipeline.stages.interp_hessian import build_interp_hessian

FIXTURE_XYZ = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "xyz_files"
    / "2j6a_ZN_homo_d2.60_cluster1.xyz"
)
RUN_ID = "cluster-interp"

# The .hess block is written with 8 decimals in scientific notation, so an
# invariant that holds exactly in memory only holds to ~1e-8 per element after
# the round trip, and accumulates over the ~30 atoms being summed. Still orders
# of magnitude below the smallest physically meaningful force constant (bonded
# springs here are ~1e-1, long-range ones ~1e-4 Ha/Bohr^2).
ROUNDTRIP_TOL = 1e-7


@pytest.fixture(scope="module")
def interp_run(tmp_path_factory):
    """Build one interpolated Hessian and share it across the assertions."""
    run_dir = tmp_path_factory.mktemp("interp") / RUN_ID
    run_dir.mkdir(parents=True)
    shutil.copy2(FIXTURE_XYZ, run_dir / f"{RUN_ID}.xyz")

    result = build_interp_hessian(run_dir, RUN_ID)
    return {"run_dir": run_dir, "result": result}


def test_packaged_ligand_files_are_shipped():
    """The .interp database must travel with the package, not the checkout."""
    names = {path.name for path in resources.interp_ligand_files()}
    # Both histidine files are used: the two ring nitrogens coordinate
    # differently, so each gets its own interpolation.
    assert {"ZnHis.interp", "ZnHis_2.interp", "ZnCys.interp"} <= names


def test_writes_the_hess_the_corvus_stage_expects(interp_run):
    run_dir = interp_run["run_dir"]
    assert (run_dir / f"{RUN_ID}.hess").is_file()
    # The merged spring model is kept as the record of what the subgraph search
    # matched; it is the only provenance for the constants used.
    assert (run_dir / "spring.model").is_file()


def test_hess_is_readable_by_the_orca_hessian_parser(interp_run):
    """The CORVUS stage parses this with read_orca_hessian and must not care
    whether ORCA or the interpolation produced it."""
    hess_path = interp_run["run_dir"] / f"{RUN_ID}.hess"
    matrix, natoms = read_orca_hessian(hess_path)

    expected_atoms = chem_xyz.count_atoms_xyz(FIXTURE_XYZ)
    assert natoms == expected_atoms
    assert matrix.shape == (3 * expected_atoms, 3 * expected_atoms)
    assert np.isfinite(matrix).all()


def test_hess_is_symmetric_after_a_round_trip(interp_run):
    matrix, _ = read_orca_hessian(interp_run["run_dir"] / f"{RUN_ID}.hess")
    assert np.max(np.abs(matrix - matrix.T)) == pytest.approx(0.0, abs=1e-10)


def test_acoustic_sum_rule_holds(interp_run):
    """Each 3x3 row block must sum to zero: a rigid translation costs no energy.

    This is the invariant the Hessian construction enforces by building the
    diagonal blocks from the off-diagonals, and it is what makes the six
    trans/rot modes come out at ~0 in the frequency check.
    """
    matrix, natoms = read_orca_hessian(interp_run["run_dir"] / f"{RUN_ID}.hess")
    for i in range(natoms):
        block_row_sum = matrix[3 * i : 3 * i + 3, :].reshape(3, natoms, 3).sum(axis=1)
        assert np.max(np.abs(block_row_sum)) == pytest.approx(0.0, abs=ROUNDTRIP_TOL)


def test_uniform_translation_is_a_zero_mode(interp_run):
    """H @ (uniform shift) == 0, the direct consequence of the sum rule."""
    matrix, natoms = read_orca_hessian(interp_run["run_dir"] / f"{RUN_ID}.hess")
    for axis in range(3):
        displacement = np.zeros(3 * natoms)
        displacement[axis::3] = 1.0
        assert np.max(np.abs(matrix @ displacement)) == pytest.approx(0.0, abs=ROUNDTRIP_TOL)


def test_atom_order_matches_the_geometry_corvus_will_use(interp_run):
    """The Hessian and the .dym must index the same atoms in the same order.

    Both stages resolve the geometry through chem.xyz.select_run_xyz, so this
    asserts the shared selection actually agrees with what was built.
    """
    run_dir = interp_run["run_dir"]
    selected = chem_xyz.select_run_xyz(run_dir, RUN_ID)
    assert selected == run_dir / f"{RUN_ID}.xyz"

    atomic_numbers, _masses, _coords = chem_xyz.read_xyz(selected)
    _matrix, natoms = read_orca_hessian(run_dir / f"{RUN_ID}.hess")
    assert natoms == len(atomic_numbers)


def test_result_reports_the_frequency_check(interp_run):
    result = interp_run["result"]
    assert result.n_atoms == chem_xyz.count_atoms_xyz(FIXTURE_XYZ)
    assert result.n_pairs > 0
    # The check ran, so a count is present (it may legitimately be non-zero: a
    # spring model of an unoptimized cluster need not sit at a minimum).
    assert result.n_imaginary is not None
    assert result.wavenumbers is not None


def test_missing_geometry_is_reported_not_crashed(tmp_path):
    run_dir = tmp_path / "empty-interp"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_interp_hessian(run_dir, "empty-interp")


def test_explicit_ligand_selection_is_honored(tmp_path):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    shutil.copy2(FIXTURE_XYZ, run_dir / f"{RUN_ID}.xyz")

    only_cys = [p for p in resources.interp_ligand_files() if p.name == "ZnCys.interp"]
    result = build_interp_hessian(run_dir, RUN_ID, ligand_files=only_cys, freq_check=False)

    # Fewer ligand types matched -> a valid Hessian is still produced, and the
    # frequency check was skipped as asked.
    assert result.n_imaginary is None
    assert (run_dir / f"{RUN_ID}.hess").is_file()


class TestHydrogenSwapConflicts:
    """Conflicts that are only interchangeable H being labelled differently.

    A ligand with equivalent hydrogens has graph automorphisms, so the subgraph
    search returns one match per H permutation. Those all describe the same
    physical bond, so they must not be reported as ambiguities -- but a
    disagreement involving heavy atoms still must be.
    """

    SYMBOLS = ["Zn", "N", "C", "H", "H", "H"]  # 3, 4, 5 are interchangeable H

    @staticmethod
    def _conflict(refs):
        # (physical_i, physical_j, values, contributing_ref_pairs)
        return (0, 7, {(0.1,), (0.2,)}, refs)

    def test_differing_only_in_which_h_is_degenerate(self):
        from xas_pipeline.chem.springs import partition_h_swap_conflicts

        conflict = self._conflict([(2, 3), (2, 4), (2, 5)])  # C-H, three ways
        genuine, degenerate = partition_h_swap_conflicts([conflict], self.SYMBOLS)
        assert genuine == []
        assert degenerate == [conflict]

    def test_differing_in_a_heavy_atom_is_genuine(self):
        from xas_pipeline.chem.springs import partition_h_swap_conflicts

        conflict = self._conflict([(0, 1), (0, 2)])  # Zn-N vs Zn-C: real ambiguity
        genuine, degenerate = partition_h_swap_conflicts([conflict], self.SYMBOLS)
        assert genuine == [conflict]
        assert degenerate == []

    def test_same_heavy_partner_different_h_is_degenerate(self):
        from xas_pipeline.chem.springs import partition_h_swap_conflicts

        conflict = self._conflict([(0, 3), (0, 5)])  # Zn-H either way
        genuine, degenerate = partition_h_swap_conflicts([conflict], self.SYMBOLS)
        assert (genuine, degenerate) == ([], [conflict])

    def test_mixed_batch_is_split(self):
        from xas_pipeline.chem.springs import partition_h_swap_conflicts

        h_only = self._conflict([(2, 3), (2, 4)])
        heavy = self._conflict([(0, 1), (0, 2)])
        genuine, degenerate = partition_h_swap_conflicts([h_only, heavy], self.SYMBOLS)
        assert genuine == [heavy]
        assert degenerate == [h_only]


def test_real_cluster_reports_no_genuine_conflicts(capsys, tmp_path):
    """The packaged ligands on a real cluster: all conflicts are H swaps.

    Guards the log against regressing to hundreds of WARNING lines about
    hydrogen relabeling, which is what the raw upstream merge reports.
    """
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    shutil.copy2(FIXTURE_XYZ, run_dir / f"{RUN_ID}.xyz")

    build_interp_hessian(run_dir, RUN_ID, freq_check=False)
    out = capsys.readouterr().out

    assert "Hydrogen-swap degeneracies" in out
    assert "got inconsistent data" not in out
    assert "genuinely inconsistent data" not in out
