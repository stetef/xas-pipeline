"""Fit spring models to ORCA Hessians, and build ligand ``.interp`` files.

This is the database-building side of the spring route; the batch pipeline only
consumes the results (:mod:`xas_pipeline.stages.interp_hessian`).

Each pair's spring constant is the off-diagonal Hessian block projected on the
unit bond vector of the ``.hess`` file's own geometry::

    k_ij = -U_ij^T H_ij U_ij

(positive = restoring). The fit is checked by rebuilding the Hessian from the
springs (acoustic sum rule on the diagonal) and reporting the error against the
original.

A ligand's ``.interp`` combines two fits of the same atoms, one per charge
state (:func:`build_interp_from_hess`). The natural-charge Hessian must come
first: its geometry becomes the ``.interp`` reference, from which inference
builds the ligand bond graph that is matched against each cluster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import springs
from .hessian import HessFile, read_hess
from .units import BOHR_TO_ANG, HARTREE_BOHR2_TO_EV_ANG2, HARTREE_BOHR2_TO_NM

log = logging.getLogger(__name__)


def compute_spring_constants(hess: HessFile, cutoff=None, bonded_only=False, bond_factor=1.2):
    """``(i, j, k_au, dist_ang, sym_i, sym_j)`` for each selected pair, i < j."""
    coords_ang = hess.coords_bohr * BOHR_TO_ANG
    H = hess.matrix
    pairs = []
    n_atoms = len(hess.symbols)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            sym_i, sym_j = hess.symbols[i], hess.symbols[j]
            r_ij = coords_ang[j] - coords_ang[i]
            dist_ang = float(np.linalg.norm(r_ij))
            if dist_ang < 1e-6:
                continue
            if cutoff is not None and dist_ang > cutoff:
                continue
            if bonded_only and not springs.is_bonded(sym_i, sym_j, dist_ang, bond_factor):
                continue
            U_ij = r_ij / dist_ang
            k_au = -float(U_ij @ H[3 * i:3 * i + 3, 3 * j:3 * j + 3] @ U_ij)
            pairs.append((i, j, k_au, dist_ang, sym_i, sym_j))
    return pairs


def reconstruct_hessian(hess: HessFile, pairs) -> np.ndarray:
    """The Hessian implied by *pairs* alone, with the acoustic sum rule on the diagonal."""
    coords_ang = hess.coords_bohr * BOHR_TO_ANG
    n_atoms = len(hess.symbols)
    H_spring = np.zeros(hess.matrix.shape)
    for i, j, k_au, *_ in pairs:
        r_ij = coords_ang[j] - coords_ang[i]
        U = r_ij / np.linalg.norm(r_ij)
        outer = np.outer(U, U)
        H_spring[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= k_au * outer
        H_spring[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= k_au * outer
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                H_spring[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= H_spring[3 * i:3 * i + 3, 3 * j:3 * j + 3]
    return H_spring


def reconstruction_error(hess: HessFile, pairs) -> dict:
    """Error of the spring-model Hessian against the original (Ha/Bohr^2).

    ``relative_mae`` is MAE / mean|H|: < 5 % excellent, < 15 % good, < 30 % fair.
    """
    diff = np.abs(hess.matrix - reconstruct_hessian(hess, pairs))
    mae = float(diff.mean())
    scale = float(np.abs(hess.matrix).mean())
    stats = {
        "mae": mae,
        "rmse": float(np.sqrt((diff**2).mean())),
        "max_abs": float(diff.max()),
        "relative_mae": mae / scale if scale > 0 else float("nan"),
    }
    n_atoms = len(hess.symbols)
    log.info(
        "Reconstruction (%d of %d pairs): MAE %.3e Ha/Bohr^2 (%.3e eV/A^2, %.3e N/m), "
        "RMSE %.3e, max %.3e, relative MAE %.2f%%",
        len(pairs), n_atoms * (n_atoms - 1) // 2, mae, mae * HARTREE_BOHR2_TO_EV_ANG2,
        mae * HARTREE_BOHR2_TO_NM, stats["rmse"], stats["max_abs"], 100 * stats["relative_mae"],
    )
    return stats


def write_springs_file(path, pairs, hess: HessFile) -> None:
    """Write a fitted spring model (coordinates relative to atom 0)."""
    coords_ang = hess.coords_bohr * BOHR_TO_ANG
    lines = [
        "# Atom symbols and coordinates (index order):\n",
        f'# {"Index":>8}  {"Symbol":>6}  {"Mass (amu)":>12}  {"X (Ang)":>14}  {"Y (Ang)":>14}  {"Z (Ang)":>14}\n',
        "# " + "-" * 80 + "\n",
    ]
    for idx, (sym, mass) in enumerate(zip(hess.symbols, hess.masses, strict=True)):
        x, y, z = coords_ang[idx] - coords_ang[0]
        lines.append(f"# {idx:>8}  {sym:>6}  {mass:>12.6f}  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}\n")
    lines += [
        "#\n",
        "# Spring constants from ORCA Hessian spring model\n",
        "# Units: Ha/Bohr²\n",
        f'# {"atom1":>8}  {"atom2":>8}  {"k (Ha/Bohr²)":>22}\n',
        "# " + "-" * 44 + "\n",
    ]
    for i, j, k_au, *_ in sorted(pairs, key=lambda p: (p[0], p[1])):
        lines.append(f"  {i:>8}  {j:>8}  {k_au:>22.10f}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")
    log.info("Spring constants written to %s", path)


@dataclass(frozen=True)
class SpringModelOptions:
    """Which pairs a fitted model keeps.

    The default -- every pair -- is what the packaged ``.interp`` files were built
    with (they carry Zn-H springs at 5 Angstrom). ``bond_factor`` applies only
    with ``bonded_only``.
    """

    cutoff: float | None = None
    bonded_only: bool = False
    bond_factor: float = 1.2


@dataclass(frozen=True)
class SpringModelResult:
    springs_path: Path
    symbols: list[str]
    coords_ang: np.ndarray
    n_pairs: int
    reconstruction: dict | None  # reconstruction_error(), or None if skipped


def spring_model_from_hess(hess_path, springs_path, *, options: SpringModelOptions | None = None,
                           check: bool = True) -> SpringModelResult:
    """Fit the spring model of one ``.hess`` and write it to *springs_path*."""
    options = options or SpringModelOptions()
    hess = read_hess(hess_path)
    if not hess.symbols:
        raise ValueError(f"No $atoms block in {hess_path}")
    pairs = compute_spring_constants(hess, options.cutoff, options.bonded_only, options.bond_factor)
    log.info("Fitted %s: %d atoms, %d pairs", hess_path, len(hess.symbols), len(pairs))
    reconstruction = reconstruction_error(hess, pairs) if check else None
    springs_path = Path(springs_path)
    springs_path.parent.mkdir(parents=True, exist_ok=True)
    write_springs_file(springs_path, pairs, hess)
    return SpringModelResult(
        springs_path=springs_path,
        symbols=hess.symbols,
        coords_ang=hess.coords_bohr * BOHR_TO_ANG,
        n_pairs=len(pairs),
        reconstruction=reconstruction,
    )


@dataclass(frozen=True)
class InterpBuildResult:
    interp_path: Path
    natural: SpringModelResult
    modified: SpringModelResult
    # Bonds present in only one charge state (inference bond criterion). The
    # .interp is written either way -- inference matches the natural-charge graph
    # -- but a bond that breaks between charge states deserves a look.
    bonds_only_natural: list[tuple[int, int]] = field(default_factory=list)
    bonds_only_modified: list[tuple[int, int]] = field(default_factory=list)


def build_interp_from_hess(
    natural_hess,
    modified_hess,
    interp_path,
    *,
    springs_dir=None,
    options: SpringModelOptions | None = None,
    check: bool = True,
    bond_factor: float = 1.2,
    bond_cutoff: float | None = None,
) -> InterpBuildResult:
    """Build a ligand's ``.interp`` from its natural- and modified-charge Hessians.

    The two fitted models are kept as ``<stem>.natural.springs`` and
    ``<stem>.modified.springs`` in *springs_dir* (default: beside *interp_path*);
    the ``.interp`` header names them. *bond_factor* / *bond_cutoff* only drive
    the bond-graph comparison between the charge states and should match what
    inference uses.

    Raises :class:`ValueError` when the Hessians do not list the same atoms in
    the same order.
    """
    interp_path = Path(interp_path)
    springs_dir = Path(springs_dir) if springs_dir is not None else interp_path.parent
    natural = spring_model_from_hess(
        natural_hess, springs_dir / f"{interp_path.stem}.natural.springs", options=options, check=check
    )
    modified = spring_model_from_hess(
        modified_hess, springs_dir / f"{interp_path.stem}.modified.springs", options=options, check=check
    )
    if natural.symbols != modified.symbols:
        raise ValueError(
            f"{natural_hess} and {modified_hess} do not list the same atoms in the "
            "same order; the .interp pairs would be meaningless."
        )

    natural_bonds = springs.bond_set(natural.symbols, natural.coords_ang, bond_factor, bond_cutoff)
    modified_bonds = springs.bond_set(modified.symbols, modified.coords_ang, bond_factor, bond_cutoff)
    only_natural = sorted(natural_bonds - modified_bonds)
    only_modified = sorted(modified_bonds - natural_bonds)
    for label, bonds in (("natural", only_natural), ("modified", only_modified)):
        for i, j in bonds:
            log.warning("Bond %s%d-%s%d exists only in the %s-charge geometry",
                        natural.symbols[i], i, natural.symbols[j], j, label)

    interp_path.parent.mkdir(parents=True, exist_ok=True)
    springs.build_interp(natural.springs_path, modified.springs_path, interp_path)
    return InterpBuildResult(interp_path, natural, modified, only_natural, only_modified)
