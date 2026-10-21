"""Build an ORCA-format Hessian from a spring model and a geometry.

For every pair with spring constant k_ij (Ha/Bohr^2) and unit bond vector U_ij
taken from the *geometry* (never from the spring model's reference coordinates)::

    H_ij = -k_ij U_ij (x) U_ij          (off-diagonal 3x3 blocks)
    H_ii = -sum_{j != i} H_ij           (acoustic sum rule)

Frequency check: H is mass-weighted (F = H / sqrt(m_i m_j), SI) with the spring
model's masses and diagonalized; negative eigenvalues are imaginary modes. The
``skip_modes`` lowest-|eigenvalue| modes are treated as translation/rotation.

Spectrum repair (``min_freq_scale > 0``, needs the frequency check): every
eigenvalue below ``min_freq_scale * max(eigval) / 1000^2`` is raised to that
floor, the trans/rot modes are set to exactly zero, and H is rebuilt from the
repaired spectrum with the eigenvectors unchanged. A spring model interpolated
onto a cluster that was never optimized under it is not at a minimum, and its
imaginary or near-zero modes would wreck the Debye-Waller factors downstream.
Pinning six trans/rot modes is only right when the null space is six-dimensional;
``add_inter_ligand`` couples fragments a ligand-only model leaves floating.

The written ``.hess`` is read by :func:`xas_pipeline.chem.hessian.read_orca_hessian`,
the same parser as a real ORCA Hessian.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hessian import write_hess
from .springs import read_springs
from .units import AMU_TO_KG, ANG_TO_BOHR, BOHR_TO_M, C_CM_PER_S, HARTREE_TO_J
from .xyz import read_symbols_coords

log = logging.getLogger(__name__)

# 1.0: no mode left softer than 1/1000 of the stiffest one.
MIN_FREQ_SCALE_DEFAULT = 1.0

# A hydrogen bond's spring constant (Ha/Bohr^2): the unit of add_inter_ligand.
INTER_LIGAND_K_HYDROGEN = 0.005


def build_hessian(coords_ang: np.ndarray, spring_pairs, n_atoms: int) -> np.ndarray:
    """The 3N x 3N Hessian (Ha/Bohr^2) from ``(i, j, k)`` pairs and the geometry."""
    coords_bohr = coords_ang * ANG_TO_BOHR
    H = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i, j, k_au in spring_pairs:
        if i == j or k_au == 0.0:
            continue
        diff = coords_bohr[j] - coords_bohr[i]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            continue
        U = diff / dist
        block = -k_au * np.outer(U, U)
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] += block
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] += block.T
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= block
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] -= block
    return H


def diagonalize_and_check(H, masses_amu, n_skip_modes=6, fix_neg_freq=MIN_FREQ_SCALE_DEFAULT):
    """Mass-weight and diagonalize H; optionally repair the spectrum.

    Returns ``(wavenumbers, skip_mask, n_imaginary, H_fixed)``: wavenumbers in
    cm^-1 ascending (negative = imaginary), the trans/rot mask, the number of
    imaginary modes outside it, and H rebuilt from the floored spectrum (H itself
    when ``fix_neg_freq <= 0``).
    """
    n_coords = 3 * len(masses_amu)

    H_SI = H * HARTREE_TO_J / BOHR_TO_M**2  # kg/s^2
    m_vec = np.repeat(1.0 / np.sqrt(np.array(masses_amu) * AMU_TO_KG), 3)
    F = H_SI * np.outer(m_vec, m_vec)  # 1/s^2
    F = 0.5 * (F + F.T)

    eigvals, eigvecs = np.linalg.eigh(F)
    nu_hz = np.sqrt(np.abs(eigvals)) / (2.0 * np.pi)
    wavenumbers = np.sign(eigvals) * nu_hz / C_CM_PER_S

    skip_indices = np.argsort(np.abs(eigvals))[:n_skip_modes]
    skip_mask = np.zeros(n_coords, dtype=bool)
    skip_mask[skip_indices] = True
    n_imaginary = int(np.sum((wavenumbers < 0.0) & (~skip_mask)))

    if fix_neg_freq <= 0.0:
        return wavenumbers, skip_mask, n_imaginary, H

    min_positive = fix_neg_freq * np.max(eigvals) / 1000.0**2
    below = eigvals < min_positive
    eigvals_fixed = np.where(below, min_positive, eigvals)
    eigvals_fixed[skip_indices] = 0.0
    n_raised = int(np.sum(below & (~skip_mask)))
    if n_raised:
        log.info("Eigenvalue floor: %d mode(s) raised to %.6e 1/s^2", n_raised, min_positive)

    F_fixed = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
    F_fixed = 0.5 * (F_fixed + F_fixed.T)
    M_outer = np.outer(m_vec, m_vec)
    H_fixed = np.zeros_like(F_fixed, dtype=float)
    np.divide(F_fixed, M_outer, out=H_fixed, where=M_outer != 0)
    H_fixed = H_fixed * BOHR_TO_M**2 / HARTREE_TO_J
    return wavenumbers, skip_mask, n_imaginary, H_fixed


def frequency_report(wavenumbers, skip_mask, n_imaginary) -> str:
    """The wavenumber table, one mode per line, with trans/rot and imaginary marked."""
    lines = [f"  {'Mode':>6}   {'Wavenumber (cm-1)':>18}   Note", "  " + "-" * 44]
    for idx, wn in enumerate(wavenumbers):
        note = "(trans/rot)" if skip_mask[idx] else ("*** IMAGINARY ***" if wn < 0.0 else "")
        lines.append(f"  {idx:>6}   {wn:>18.4f}   {note}")
    n_skip = int(np.sum(skip_mask))
    lines.append(
        f"  WARNING: {n_imaginary} imaginary mode(s) beyond the {n_skip} trans/rot modes; "
        "not a stable minimum."
        if n_imaginary
        else f"  No imaginary frequencies (excluding {n_skip} trans/rot modes)."
    )
    return "\n".join(lines)


@dataclass
class HessianResult:
    """What :func:`build_hess_from_spring_model` produced."""

    output_path: Path
    n_atoms: int
    n_pairs: int
    symmetry_error: float
    wavenumbers: np.ndarray | None
    #: Imaginary modes in the Hessian written (after any repair); None without
    #: the frequency check.
    n_imaginary: int | None
    #: Imaginary modes in the raw spring Hessian, before repair: the number that
    #: says something about the spring model itself.
    n_imaginary_raw: int | None = None


def build_hess_from_spring_model(
    xyz_path,
    springs_path,
    output_path,
    *,
    zero_negative: bool = False,
    add_inter_ligand: float = 0.0,
    freq_check: bool = True,
    skip_modes: int = 6,
    min_freq_scale: float = MIN_FREQ_SCALE_DEFAULT,
) -> HessianResult:
    """Write the ``.hess`` for the geometry at *xyz_path* and the spring model.

    *zero_negative* clips negative constants to zero. *add_inter_ligand* > 0
    floors every pair at that many hydrogen bonds (:data:`INTER_LIGAND_K_HYDROGEN`),
    weakly coupling atoms the model says nothing about. *min_freq_scale* > 0
    repairs the spectrum (see the module docstring).

    Raises :class:`ValueError` when the geometry and spring model disagree on
    atoms: the model was built for a different structure.
    """
    symbols, coords, _comment = read_symbols_coords(xyz_path)
    model = read_springs(springs_path)
    if len(symbols) != len(model.symbols):
        raise ValueError(
            f"Atom count mismatch: {xyz_path} has {len(symbols)} atoms, "
            f"spring model {springs_path} has {len(model.symbols)}."
        )
    mismatches = [
        f"atom {idx}: xyz={xs}, spring model={ss}"
        for idx, (xs, ss) in enumerate(zip(symbols, model.symbols))
        if xs.capitalize() != ss.capitalize()
    ]
    if mismatches:
        raise ValueError(f"Atomic symbol mismatch between {xyz_path} and {springs_path}: {'; '.join(mismatches)}")

    n_atoms = len(symbols)
    pairs = model.pairs
    log.info("Spring Hessian: %s (%d atoms) + %s (%d pairs)", xyz_path, n_atoms, springs_path, len(pairs))
    if zero_negative:
        log.info("zero_negative: %d negative constant(s) set to zero", sum(1 for _, _, k in pairs if k < 0))
        pairs = [(i, j, max(k, 0.0)) for i, j, k in pairs]
    if add_inter_ligand > 0.0:
        k_floor = INTER_LIGAND_K_HYDROGEN * add_inter_ligand
        log.info("add_inter_ligand: %d pair(s) raised to k = %.6g Ha/Bohr^2",
                 sum(1 for _, _, k in pairs if k < k_floor), k_floor)
        pairs = [(i, j, max(k_floor, k)) for i, j, k in pairs]

    H = build_hessian(coords, pairs, n_atoms)
    sym_err = float(np.max(np.abs(H - H.T)))
    log.info("Symmetry check: max|H - H^T| = %.3e", sym_err)

    H_out, wavenumbers, n_imaginary, n_imaginary_raw = H, None, None, None
    if freq_check:
        raw_wn, raw_skip, n_imaginary_raw, H_fixed = diagonalize_and_check(
            H, model.masses, n_skip_modes=skip_modes, fix_neg_freq=min_freq_scale
        )
        wavenumbers, skip_mask, n_imaginary = raw_wn, raw_skip, n_imaginary_raw
        if min_freq_scale > 0.0:
            # Report the frequencies of the Hessian actually written; the second
            # repair is a no-op on an already-floored spectrum.
            wavenumbers, skip_mask, n_imaginary, _ = diagonalize_and_check(
                H_fixed, model.masses, n_skip_modes=skip_modes, fix_neg_freq=min_freq_scale
            )
            H_out = H_fixed
            if not np.allclose(H_fixed, H, rtol=1e-8, atol=1e-12):
                log.info("Frequencies of the raw spring Hessian:\n%s",
                         frequency_report(raw_wn, raw_skip, n_imaginary_raw))
                log.info("Frequencies after the eigenvalue floor:")
        log.info("%s", frequency_report(wavenumbers, skip_mask, n_imaginary))

    write_hess(output_path, H_out, model.symbols, model.masses, coords)
    log.info("Hessian written to %s", output_path)
    return HessianResult(
        output_path=Path(output_path),
        n_atoms=n_atoms,
        n_pairs=len(pairs),
        symmetry_error=sym_err,
        wavenumbers=wavenumbers,
        n_imaginary=n_imaginary,
        n_imaginary_raw=n_imaginary_raw,
    )
