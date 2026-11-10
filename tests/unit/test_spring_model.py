"""Tests for building ligand spring models from ORCA Hessians (database side).

The test ligand is a five-atom chain of distinct elements, so its bond graph has
exactly one automorphism and every subgraph match is unambiguous. Its Hessians
are *built* from known spring constants, which gives each check an exact answer:

* projecting a pure spring Hessian recovers the constants it was built from;
* the ``.interp`` reference geometry is the natural-charge one (model 1), since
  that is what inference builds the ligand's bond graph from;
* end to end, interpolating at the natural geometry reproduces the
  natural-charge springs.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from xas_pipeline.chem import hessian, spring_hessian, spring_model, springs
from xas_pipeline.chem.units import BOHR_TO_ANG
from xas_pipeline.stages.interp_hessian import (
    RUN_LIGAND_DIR_NAME,
    SPRING_MODEL_NAME,
    build_interp_hessian,
    resolve_ligand_files,
)

SYMBOLS = ["Zn", "S", "C", "N", "O"]
MASSES = [65.38, 32.06, 12.011, 14.007, 15.999]
NATURAL_COORDS = np.array(
    [
        [0.00, 0.000, 0.0],
        [2.30, 0.000, 0.0],
        [3.21, 1.576, 0.0],
        [4.68, 1.576, 0.0],
        [5.38, 0.364, 0.3],
    ]
)
BONDS = {(0, 1), (1, 2), (2, 3), (3, 4)}

# The modified charge state: every bond 3% longer and every spring 20% softer.
STRETCH = 1.03
SOFTEN = 0.8

# Springs are written with 10 decimals; the synthetic .hess with 12.
K_TOL = 1e-9


def _natural_springs() -> dict[tuple[int, int], float]:
    """Stiff bonded springs, weak (one of them negative) non-bonded ones."""
    k = {}
    for i, j in itertools.combinations(range(len(SYMBOLS)), 2):
        k[(i, j)] = 0.1 + 0.05 * i if (i, j) in BONDS else 0.002 * (j - i) - 0.003
    return k


def _write_hess(path: Path, coords_ang: np.ndarray, k: dict, symbols=SYMBOLS) -> Path:
    pairs = [(i, j, kij) for (i, j), kij in k.items()]
    H = spring_hessian.build_hessian(coords_ang, pairs, len(symbols))
    coords_bohr = coords_ang / BOHR_TO_ANG
    lines = ["$atoms\n", f"{len(symbols)}\n"]
    for sym, mass, (x, y, z) in zip(symbols, MASSES, coords_bohr, strict=True):
        lines.append(f" {sym:<3} {mass:10.5f} {x:18.12f} {y:18.12f} {z:18.12f}\n")
    lines.append("\n$hessian\n")
    lines.extend(hessian.format_hessian_block(H, 5, 12))
    lines.append("\n$end\n")
    path.write_text("".join(lines))
    return path


def _write_xyz(path: Path, coords_ang: np.ndarray) -> Path:
    body = "".join(
        f"{sym} {x:.10f} {y:.10f} {z:.10f}\n" for sym, (x, y, z) in zip(SYMBOLS, coords_ang, strict=True)
    )
    path.write_text(f"{len(SYMBOLS)}\ntest ligand\n{body}")
    return path


@pytest.fixture
def charge_states(tmp_path):
    natural_k = _natural_springs()
    modified_k = {pair: SOFTEN * kij for pair, kij in natural_k.items()}
    natural = _write_hess(tmp_path / "natural.hess", NATURAL_COORDS, natural_k)
    modified = _write_hess(tmp_path / "modified.hess", NATURAL_COORDS * STRETCH, modified_k)
    return {"natural": natural, "modified": modified, "natural_k": natural_k,
            "modified_k": modified_k}


def test_springs_recover_the_constants_the_hessian_was_built_from(charge_states, tmp_path):
    result = spring_model.spring_model_from_hess(
        charge_states["natural"], tmp_path / "natural.springs"
    )
    model = springs.read_springs(result.springs_path)
    symbols, pairs = model.symbols, model.pairs

    assert symbols == SYMBOLS
    assert len(pairs) == len(SYMBOLS) * (len(SYMBOLS) - 1) // 2
    for i, j, k in pairs:
        assert k == pytest.approx(charge_states["natural_k"][(i, j)], abs=K_TOL)


def test_reconstruction_of_a_pure_spring_hessian_is_exact(charge_states, tmp_path):
    result = spring_model.spring_model_from_hess(
        charge_states["natural"], tmp_path / "natural.springs"
    )
    assert result.reconstruction["relative_mae"] < 1e-9


def test_pair_selection_options_are_honoured(charge_states, tmp_path):
    result = spring_model.spring_model_from_hess(
        charge_states["natural"],
        tmp_path / "bonded.springs",
        options=spring_model.SpringModelOptions(bonded_only=True),
        check=False,
    )
    pairs = springs.read_springs(result.springs_path).pairs
    assert {(i, j) for i, j, _k in pairs} == BONDS
    assert result.reconstruction is None


def test_hess_without_a_hessian_block_is_refused(tmp_path):
    path = tmp_path / "broken.hess"
    path.write_text("$atoms\n1\n Zn 65.38 0.0 0.0 0.0\n$end\n")
    with pytest.raises(ValueError, match=r"\$hessian"):
        hessian.read_hess(path)


def test_interp_reference_is_the_natural_charge_state(charge_states, tmp_path):
    result = spring_model.build_interp_from_hess(
        charge_states["natural"], charge_states["modified"], tmp_path / "lig.interp"
    )
    model = springs.read_interp(result.interp_path)
    symbols, coords, pairs = model.symbols, model.coords, model.pairs

    assert symbols == SYMBOLS
    # Reference coordinates are the natural geometry (written relative to atom 0).
    np.testing.assert_allclose(coords, NATURAL_COORDS - NATURAL_COORDS[0], atol=1e-7)
    for i, j, k1, k2, d1, d2 in pairs:
        d_natural = np.linalg.norm(NATURAL_COORDS[i] - NATURAL_COORDS[j])
        assert d1 == pytest.approx(d_natural, abs=1e-7)
        assert d2 == pytest.approx(STRETCH * d_natural, abs=1e-7)
        assert k1 == pytest.approx(charge_states["natural_k"][(i, j)], abs=K_TOL)
        assert k2 == pytest.approx(charge_states["modified_k"][(i, j)], abs=K_TOL)

    # Both intermediate spring models are kept, named in the .interp header.
    header = result.interp_path.read_text()
    assert "lig.natural.springs" in header and "lig.modified.springs" in header
    assert not result.bonds_only_natural and not result.bonds_only_modified


def test_atoms_in_a_different_order_are_refused(charge_states, tmp_path):
    k = charge_states["natural_k"]
    reordered = _write_hess(
        tmp_path / "reordered.hess", NATURAL_COORDS, k, symbols=["S", "Zn", "C", "N", "O"]
    )
    with pytest.raises(ValueError, match="same order"):
        spring_model.build_interp_from_hess(
            charge_states["natural"], reordered, tmp_path / "lig.interp"
        )


def test_a_bond_broken_in_one_charge_state_is_reported(charge_states, tmp_path):
    broken = NATURAL_COORDS.copy()
    broken[4] = broken[3] + 2.0 * (broken[4] - broken[3])  # N-O well past bonding
    modified = _write_hess(tmp_path / "broken.hess", broken, charge_states["modified_k"])

    result = spring_model.build_interp_from_hess(
        charge_states["natural"], modified, tmp_path / "lig.interp"
    )
    assert result.bonds_only_natural == [(3, 4)]
    assert result.bonds_only_modified == []
    assert result.interp_path.is_file()


def test_inference_at_the_natural_geometry_reproduces_the_natural_springs(
    charge_states, tmp_path
):
    db = tmp_path / "db"
    db.mkdir()
    spring_model.build_interp_from_hess(
        charge_states["natural"], charge_states["modified"], db / "lig.interp"
    )

    run_id = "lig-asis-spring"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    _write_xyz(run_dir / f"{run_id}.xyz", NATURAL_COORDS)
    build_interp_hessian(
        run_dir,
        run_id,
        ligand_files=sorted(db.glob("*.interp")),
        add_inter_ligand=0.0,
        freq_check=False,
    )

    pairs = springs.read_springs(run_dir / SPRING_MODEL_NAME).pairs
    assert len(pairs) == len(charge_states["natural_k"])
    for i, j, k in pairs:
        assert k == pytest.approx(charge_states["natural_k"][(i, j)], abs=1e-7)


def test_a_run_dir_ligand_database_replaces_the_packaged_one(charge_states, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / RUN_LIGAND_DIR_NAME).mkdir(parents=True)
    interp = run_dir / RUN_LIGAND_DIR_NAME / "lig.interp"
    spring_model.build_interp_from_hess(
        charge_states["natural"], charge_states["modified"], interp,
        springs_dir=tmp_path / "springs",
    )
    assert resolve_ligand_files(run_dir) == [interp]

    # An explicit selection still wins over the run's own database.
    explicit = [tmp_path / "other.interp"]
    assert resolve_ligand_files(run_dir, explicit) == explicit


def test_an_empty_run_dir_ligand_database_is_an_error_not_a_fallback(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / RUN_LIGAND_DIR_NAME).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_ligand_files(run_dir)


def test_build_interp_cli(charge_states, tmp_path):
    import subprocess
    import sys

    out = tmp_path / "db" / "lig.interp"
    result = subprocess.run(
        [sys.executable, "-m", "xas_pipeline.cli.build_interp",
         str(charge_states["natural"]), str(charge_states["modified"]), "-o", str(out),
         "--no-check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert (out.parent / "lig.natural.springs").is_file()


def test_a_model_of_the_structure_itself_maps_by_identity(charge_states, tmp_path, caplog):
    """Same atoms in the same order: no subgraph search, each pair keeps its own data."""
    spring_model.build_interp_from_hess(charge_states["natural"], charge_states["modified"], tmp_path / "s.interp")
    model = springs.read_interp(tmp_path / "s.interp")
    with caplog.at_level("INFO", logger="xas_pipeline"):
        pair_k = springs.interpolate(SYMBOLS, NATURAL_COORDS, [model], springs.InterpOptions(extrapolate=False))
    assert "identity mapping" in caplog.text
    for i, j, k1, k2, d1, d2 in model.pairs:
        d = np.linalg.norm(NATURAL_COORDS[i] - NATURAL_COORDS[j])
        expected = k1 if abs(d - d1) < abs(d - d2) else k2
        assert pair_k[(i, j)] == pytest.approx(expected, abs=1e-6)


# ── The bonding gate ──────────────────────────────────────────────────────────
#
# A whole-structure model (same atoms, same order) must not be interpolated onto a
# geometry whose bonding network differs from the model's reference geometry: the
# springs would describe a different molecule. These pin the refusal, its report,
# the override, and what the rest of the pipeline sees.


def _stretched():
    """Zn-S pulled from 2.30 to 2.90 A, past 1.2 x (r_Zn + r_S) = 2.72 A: bond 0-1 lost."""
    coords = NATURAL_COORDS.copy()
    coords[0, 0] -= 0.6
    assert (0, 1) not in springs.bond_set(SYMBOLS, coords)
    return coords


def _whole_structure_model(charge_states, tmp_path):
    spring_model.build_interp_from_hess(charge_states["natural"], charge_states["modified"], tmp_path / "s.interp")
    return tmp_path / "s.interp"


def test_a_bond_lost_is_refused_and_named(charge_states, tmp_path):
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    with pytest.raises(springs.BondingChangedError) as caught:
        springs.interpolate(SYMBOLS, _stretched(), [model])
    (change,) = caught.value.changes
    assert (change.i, change.j, change.kind) == (0, 1, "lost")
    assert change.d_model == pytest.approx(2.30, abs=1e-6)
    assert change.d_structure == pytest.approx(2.90, abs=1e-6)
    assert change.cutoff == pytest.approx(springs.bond_cutoff_for("Zn", "S"))
    assert "Zn0-S1 lost: 2.300 A in the model, 2.900 A in this structure" in change.describe()


def test_a_bond_formed_is_refused(charge_states, tmp_path):
    """Zn pulled onto C2 (2.0 A): a bond the model does not have."""
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    coords = NATURAL_COORDS.copy()
    coords[0] = coords[2] + np.array([0.0, -2.0, 0.0])
    with pytest.raises(springs.BondingChangedError) as caught:
        springs.interpolate(SYMBOLS, coords, [model])
    assert ("new" in {c.kind for c in caught.value.changes}
            and any((c.i, c.j) == (0, 2) for c in caught.value.changes))


def test_same_bonding_passes_the_gate(charge_states, tmp_path, caplog):
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    with caplog.at_level("INFO", logger="xas_pipeline"):
        springs.interpolate(SYMBOLS, NATURAL_COORDS * 1.01, [model])
    assert "Bonding gate: ligand 1 has this structure's bonding network" in caplog.text


def test_reordered_whole_structure_model_is_gated_too(charge_states, tmp_path):
    """The structure's own atoms in another order and a lost bond: no mapping, refused."""
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    order = [4, 3, 2, 1, 0]
    symbols = [SYMBOLS[i] for i in order]
    with pytest.raises(springs.BondingChangedError, match="another order"):
        springs.interpolate(symbols, _stretched()[order], [model])
    springs.interpolate(symbols, NATURAL_COORDS[order], [model])  # same bonds, reordered: fine


def test_fragment_models_are_not_gated(charge_states, tmp_path):
    """A model of part of the structure maps by subgraph, which already needs its bonds."""
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    symbols = [*SYMBOLS, "H"]
    coords = np.vstack([_stretched(), [[9.0, 9.0, 9.0]]])
    assert springs.interpolate(symbols, coords, [model]) == {}  # no match, nothing raised


def test_the_override_builds_it_anyway_with_a_warning(charge_states, tmp_path, caplog):
    """allow_bonding_change keeps the old behaviour: identity mapping, the lost bond stays log-space."""
    model = springs.read_interp(_whole_structure_model(charge_states, tmp_path))
    with caplog.at_level("INFO", logger="xas_pipeline"):
        pair_k = springs.interpolate(SYMBOLS, _stretched(), [model],
                                     springs.InterpOptions(allow_bonding_change=True))
    assert "WARNING: bonding network differs from spring model 1" in caplog.text
    assert "Zn0-S1 lost" in caplog.text
    assert len(pair_k) == len(model.pairs)
    _, _, k1, k2, d1, d2 = next(p for p in model.pairs if p[:2] == (0, 1))
    k, _, method, _ = springs.interp_pair(k1, k2, d1, d2, 2.90, "Zn", "S", True, True, "0,1")
    assert method == "log+"
    assert pair_k[(0, 1)] == pytest.approx(k)


def _gated_run_dir(charge_states, tmp_path, coords):
    run_dir = tmp_path / "RUN"
    (run_dir / RUN_LIGAND_DIR_NAME).mkdir(parents=True)
    spring_model.build_interp_from_hess(charge_states["natural"], charge_states["modified"],
                                        run_dir / RUN_LIGAND_DIR_NAME / "s.interp")
    _write_xyz(run_dir / "RUN.xyz", coords)
    return run_dir


def test_the_stage_builds_nothing_reports_and_exits_3(charge_states, tmp_path, monkeypatch, caplog):
    import sys

    from xas_pipeline.stages import interp_hessian

    run_dir = _gated_run_dir(charge_states, tmp_path, _stretched())
    monkeypatch.setattr(sys, "argv", ["interp_hessian", str(run_dir), "--run-id", "RUN"])
    with caplog.at_level("INFO", logger="xas_pipeline"):
        assert interp_hessian.main() == interp_hessian.BONDING_GATE_EXIT == 3
    assert not (run_dir / SPRING_MODEL_NAME).exists()
    assert not (run_dir / "RUN.hess").exists()
    marker = (run_dir / interp_hessian.BONDING_CHANGED_NAME).read_text()
    log = caplog.text
    for text in (marker, log):
        assert "BONDING GATE: no spring model built for RUN." in text
        assert "Zn0-S1 lost: 2.300 A in the model, 2.900 A in this structure" in text
        assert "s.interp" in text


def test_a_passing_rebuild_clears_an_old_verdict(charge_states, tmp_path):
    from xas_pipeline.stages import interp_hessian

    run_dir = _gated_run_dir(charge_states, tmp_path, NATURAL_COORDS)
    (run_dir / interp_hessian.BONDING_CHANGED_NAME).write_text("stale\n")
    build_interp_hessian(run_dir, "RUN")
    assert (run_dir / "RUN.hess").is_file()
    assert not (run_dir / interp_hessian.BONDING_CHANGED_NAME).exists()


def test_the_override_flag_reaches_the_stage(charge_states, tmp_path):
    run_dir = _gated_run_dir(charge_states, tmp_path, _stretched())
    with pytest.raises(springs.BondingChangedError):
        build_interp_hessian(run_dir, "RUN")
    build_interp_hessian(run_dir, "RUN", allow_bonding_change=True)
    assert (run_dir / "RUN.hess").is_file()


def test_a_gated_run_is_diagnosed_as_bonding_changed_and_never_rerun(charge_states, tmp_path):
    from xas_pipeline import corvus_diagnosis
    from xas_pipeline.stages import feff_process, interp_hessian

    run_dir = _gated_run_dir(charge_states, tmp_path, _stretched())
    with pytest.raises(springs.BondingChangedError):
        build_interp_hessian(run_dir, "RUN")
    diag = corvus_diagnosis.diagnose_run_dir(run_dir)
    assert diag.kind is corvus_diagnosis.CorvusFailureKind.BONDING_CHANGED
    assert not diag.ok and not diag.auto_remediable
    assert "Zn0-S1 lost" in diag.reason

    ok, reasons = feff_process.process_system_dir(run_dir)  # moves it into working-RUN/
    assert not ok and "bonding network differs" in reasons[0]
    assert (run_dir / "working-RUN" / interp_hessian.BONDING_CHANGED_NAME).is_file()
    assert corvus_diagnosis.diagnose_run_dir(run_dir).kind is corvus_diagnosis.CorvusFailureKind.BONDING_CHANGED


@pytest.mark.parametrize("scheduler", ["slurm-scripts", "pbs-scripts"])
def test_the_wrapper_stops_before_corvus_and_says_why(tmp_path, scheduler):
    """Run the wrapper template's Hessian step with a stand-in that exits 3."""
    import subprocess

    from xas_pipeline import resources

    template = (resources.template_root() / scheduler / "corvus-wrapper.script").read_text()
    start = template.index("[INTERP_HESS_CMD]")
    end = template.index('echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting prepare-corvus')
    script = (
        'RUN_ID=RUN; RUN_DIR=/runs/RUN; timing_file="$PWD/t"\n'
        'finalize_timing() { echo "exit_code=$1" >> "$timing_file"; }\n'
        + template[start:end].replace("[INTERP_HESS_CMD]", "(exit 3)")
        + 'echo "CORVUS STARTED"\n'
    )
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 3
    assert "CORVUS STARTED" not in result.stdout
    for stream in (result.stdout, result.stderr):
        assert "BONDING GATE: RUN not run." in stream
        assert "/runs/RUN/bonding-changed.txt" in stream
    assert (tmp_path / "t").read_text() == "exit_code=3\n"
