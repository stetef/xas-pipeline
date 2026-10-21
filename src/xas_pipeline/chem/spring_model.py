"""
Spring Model from ORCA Hessian
==============================
Vendored from DW_Interpolation/scripts/orca_spring_model.py (2026-10-20). The
numerics (.hess parsing, pair projection, reconstruction check, springs-file
format) are kept identical to the upstream script so spring models built here
match ones built by hand; the argparse CLI was replaced by
:func:`spring_model_from_hess` and :func:`build_interp_from_hess`, and the
reconstruction check returns its statistics instead of only printing them.
Upstream stays the source of truth for the science -- re-vendor rather than
editing the numerics here.

This is the database-*building* side of the spring route. The pipeline itself
only does inference (:mod:`xas_pipeline.stages.interp_hessian`); nothing in the
batch DAG calls this module.

Reads an ORCA .hess file and computes spring constants for all atom pairs
by projecting the off-diagonal 3x3 Hessian block onto the unit bond vector:

    k_ij = -U_ij^T H_ij U_ij

where:
    H_ij : 3x3 submatrix of the full Hessian (rows 3i:3i+3, cols 3j:3j+3)
    U_ij : unit vector from atom i to atom j

The negative sign ensures that a positive spring constant corresponds to a
restoring force (the off-diagonal Hessian block has the opposite sign
convention from a standard spring constant).

The geometry is the one in the .hess file's own $atoms block, so no separate
XYZ file is needed.

A Hessian reconstruction check compares the original Hessian with the one
rebuilt from the spring model:

    H_ij^spring = -k_ij * U_ij (x) U_ij      (off-diagonal 3x3 blocks)
    H_ii^spring = -sum_{j!=i} H_ij^spring    (diagonal, acoustic sum rule)

    MAE = (1/n^2) * sum_ij |H_ij^original - H_ij^spring|

The spring constants are written by :func:`write_springs_file` in the format
read by :func:`xas_pipeline.chem.springs.parse_springs_file`: the atom symbols,
masses and coordinates (Angstrom, relative to atom 0) in the header, then one
``atom1 atom2 k(Ha/Bohr^2)`` line per pair.

Building one ligand's ``.interp`` takes two such models, one per charge state
(:func:`build_interp_from_hess`). The natural-charge model must be model 1: the
``.interp`` reference coordinates come from model 1, and at inference the
ligand's bond graph -- what the subgraph search matches against the cluster --
is built from those coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xas_pipeline.chem import springs


# ── Physical constants ────────────────────────────────────────────────────────
BOHR_TO_ANG              = 0.529177210903
HARTREE_TO_EV            = 27.211396132
HARTREE_TO_J             = 4.3597447222071e-18
BOHR_TO_M                = 5.29177210903e-11
HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / BOHR_TO_ANG**2
HARTREE_BOHR2_TO_NM      = HARTREE_TO_J / BOHR_TO_M**2   # N/m

# Covalent radii live in springs.py; the upstream script carried its own copy
# of the same Alvarez 2008 table.
COVALENT_RADII = springs.COVALENT_RADII
DEFAULT_RADIUS = springs.DEFAULT_RADIUS


# ── Column-header detection ───────────────────────────────────────────────────

def _is_col_header(tokens, n):
    """
    Return True if tokens form a Hessian column-header line.
    All tokens must be non-negative integers < n, with no decimal points,
    in strictly ascending order.
    """
    if not tokens:
        return False
    if any('.' in t or 'e' in t.lower() for t in tokens):
        return False
    try:
        indices = [int(t) for t in tokens]
    except ValueError:
        return False
    if any(c < 0 or c >= n for c in indices):
        return False
    if indices != sorted(indices):
        return False
    return True


# ── Hessian block parser ──────────────────────────────────────────────────────

def _parse_hessian_block(lines, data_start_idx, n):
    """
    Parse the numeric data of a $hessian block.

    Returns
    -------
    H          : (n, n) float64 ndarray
    block_size : number of columns per block (detected)
    decimals   : number of decimal places (detected)
    end_idx    : first line index after the hessian block
    """
    H           = np.zeros((n, n))
    block_size  = None
    decimals    = 8
    col_indices = None
    i           = data_start_idx

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue
        if stripped.startswith('$') and i != data_start_idx:
            break

        tokens = stripped.split()

        if _is_col_header(tokens, n):
            col_indices = [int(t) for t in tokens]
            if block_size is None:
                block_size = len(col_indices)
            i += 1
            continue

        if col_indices is None:
            i += 1
            continue

        try:
            row_idx = int(tokens[0])
        except ValueError:
            i += 1
            continue

        if row_idx < 0 or row_idx >= n:
            i += 1
            continue

        try:
            values = [float(v) for v in tokens[1:]]
        except ValueError:
            i += 1
            continue

        for k, col in enumerate(col_indices):
            if k < len(values):
                H[row_idx, col] = values[k]

        # Detect decimal places from the first float token
        for val_str in tokens[1:]:
            clean = val_str.lstrip('+-')
            if '.' in clean:
                if 'e' in clean.lower():
                    coeff = clean.lower().split('e')[0]
                    if '.' in coeff:
                        decimals = len(coeff.split('.')[1].rstrip())
                else:
                    decimals = len(clean.split('.')[-1].rstrip())
                break

        i += 1

    return H, block_size or 6, decimals, i


# ── ORCA .hess parser ─────────────────────────────────────────────────────────

def parse_hess_file(filepath) -> dict:
    """
    Parse an ORCA .hess file.

    Returns a dict with keys:
        'n_atoms'         : int
        'n_coords'        : int
        'masses'          : np.ndarray (n_atoms,)
        'atoms'           : list of (symbol, mass, x_bohr, y_bohr, z_bohr)
        'hessian'         : np.ndarray (n_coords, n_coords) in a.u.
        'frequencies'     : np.ndarray (n_coords,) in cm-1 (if present)
        'raw_lines'       : list of str — every line in the file
        'hess_start'      : int — line index of '$hessian' keyword
        'hess_end'        : int — first line index after hessian data
        'hess_block_size' : int — columns per block
        'hess_decimals'   : int — decimal places in values

    Raises :class:`ValueError` when the file has no ``$atoms`` or no
    ``$hessian`` block (upstream carried on and failed later on a ``None``).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {
        'raw_lines'       : lines,
        'n_atoms'         : 0,
        'n_coords'        : 0,
        'masses'          : None,
        'atoms'           : [],
        'hessian'         : None,
        'frequencies'     : None,
        'hess_start'      : None,
        'hess_end'        : None,
        'hess_block_size' : 6,
        'hess_decimals'   : 8,
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '$act_atoms':
            i += 1
            data['n_atoms'] = int(lines[i].strip())

        elif line == '$act_coord':
            i += 1
            data['n_coords'] = int(lines[i].strip())

        elif line == '$atoms':
            i += 1
            n = int(lines[i].strip())
            if data['n_atoms'] == 0:
                data['n_atoms'] = n
            i += 1
            atoms = []
            for _ in range(n):
                parts = lines[i].split()
                symbol = parts[0].capitalize()
                mass   = float(parts[1])
                x      = float(parts[2])
                y      = float(parts[3])
                z      = float(parts[4])
                atoms.append((symbol, mass, x, y, z))
                i += 1
            data['atoms'] = atoms
            data['masses'] = np.array([a[1] for a in atoms])
            continue

        elif line == '$hessian':
            data['hess_start'] = i
            i += 1
            n = int(lines[i].strip())
            if data['n_coords'] == 0:
                data['n_coords'] = n
            i += 1
            H, bs, dec, end = _parse_hessian_block(lines, i, n)
            data['hessian']         = H
            data['hess_block_size'] = bs
            data['hess_decimals']   = dec
            data['hess_end']        = end
            i = end
            continue

        elif line == '$vibrational_frequencies':
            i += 1
            n = int(lines[i].strip())
            i += 1
            freqs = []
            for _ in range(n):
                parts = lines[i].split()
                freqs.append(float(parts[1]))
                i += 1
            data['frequencies'] = np.array(freqs)
            continue

        i += 1

    if not data['atoms']:
        raise ValueError(f"No $atoms block found in '{filepath}'.")
    if data['hessian'] is None:
        raise ValueError(f"No $hessian block found in '{filepath}'.")

    return data


# ── Bond utilities ────────────────────────────────────────────────────────────

def get_coords_ang(data: dict) -> np.ndarray:
    """Return atom coordinates in Ångström as (n_atoms, 3) array."""
    return np.array([[a[2], a[3], a[4]] for a in data['atoms']]) * BOHR_TO_ANG


def is_bonded(sym_i, sym_j, dist_ang, factor=1.2):
    ri = COVALENT_RADII.get(sym_i, DEFAULT_RADIUS)
    rj = COVALENT_RADII.get(sym_j, DEFAULT_RADIUS)
    return dist_ang <= factor * (ri + rj)


# ── Spring model ──────────────────────────────────────────────────────────────

def compute_spring_constants(data: dict,
                             cutoff: float = None,
                             bonded_only: bool = False,
                             bond_factor: float = 1.2) -> list:
    """
    Compute spring constants for all atom pairs.

    Returns list of tuples:
        (i, j, k_au, k_ev, k_nm, dist_ang, sym_i, sym_j)
    """
    atoms     = data['atoms']
    H         = data['hessian']
    n_atoms   = data['n_atoms']
    coords_ang = get_coords_ang(data)

    pairs = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            sym_i = atoms[i][0]
            sym_j = atoms[j][0]

            r_ij     = coords_ang[j] - coords_ang[i]
            dist_ang = float(np.linalg.norm(r_ij))

            if dist_ang < 1e-6:
                continue
            if cutoff is not None and dist_ang > cutoff:
                continue
            if bonded_only and not is_bonded(sym_i, sym_j, dist_ang, bond_factor):
                continue

            U_ij = r_ij / dist_ang

            # Extract 3x3 off-diagonal Hessian block
            H_ij = H[3*i:3*i+3, 3*j:3*j+3]

            k_au = -float(U_ij @ H_ij @ U_ij)
            k_ev = k_au * HARTREE_BOHR2_TO_EV_ANG2
            k_nm = k_au * HARTREE_BOHR2_TO_NM

            pairs.append((i, j, k_au, k_ev, k_nm, dist_ang, sym_i, sym_j))

    return pairs


# ── Hessian reconstruction ────────────────────────────────────────────────────

def reconstruct_hessian(data: dict, pairs: list) -> np.ndarray:
    """
    Reconstruct a Hessian from spring constants using ALL pairs.
    Uses the acoustic sum rule for diagonal blocks.
    """
    n         = data['n_coords']
    n_atoms   = data['n_atoms']
    coords_ang = get_coords_ang(data)

    H_spring = np.zeros((n, n))

    # Build all-pairs dict for reconstruction (ignore cutoff filter)
    all_pairs = compute_spring_constants(data, cutoff=None,
                                         bonded_only=False)

    for (i, j, k_au, k_ev, k_nm, dist_ang, sym_i, sym_j) in all_pairs:
        r_ij = coords_ang[j] - coords_ang[i]
        U    = r_ij / np.linalg.norm(r_ij)
        outer = np.outer(U, U)

        H_spring[3*i:3*i+3, 3*j:3*j+3] -= k_au * outer
        H_spring[3*j:3*j+3, 3*i:3*i+3] -= k_au * outer

    # Acoustic sum rule for diagonal blocks
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                H_spring[3*i:3*i+3, 3*i:3*i+3] -= H_spring[3*i:3*i+3, 3*j:3*j+3]

    return H_spring


# ── MAE check ────────────────────────────────────────────────────────────────

def mae_check(data: dict, pairs: list) -> dict:
    """Print the reconstruction report and return its statistics (Ha/Bohr²)."""
    H_orig   = data['hessian']
    H_spring = reconstruct_hessian(data, pairs)
    n        = data['n_coords']

    diff     = np.abs(H_orig - H_spring)
    mae_au   = diff.mean()
    rmse_au  = float(np.sqrt((diff**2).mean()))
    max_au   = diff.max()
    h_scale  = np.abs(H_orig).mean()
    rel_mae  = mae_au / h_scale if h_scale > 0 else float('nan')

    mae_ev   = mae_au  * HARTREE_BOHR2_TO_EV_ANG2
    rmse_ev  = rmse_au * HARTREE_BOHR2_TO_EV_ANG2
    mae_nm   = mae_au  * HARTREE_BOHR2_TO_NM
    rmse_nm  = rmse_au * HARTREE_BOHR2_TO_NM

    if rel_mae < 0.05:
        assessment = "Excellent — spring model reproduces Hessian well."
    elif rel_mae < 0.15:
        assessment = "Good."
    elif rel_mae < 0.30:
        assessment = "Fair — consider including more pairs."
    else:
        assessment = "Poor — spring model misses significant Hessian content."

    print()
    print("=" * 62)
    print("  Hessian Reconstruction Check (Spring Model vs Original)")
    print("=" * 62)
    print(f"  Matrix size            : {n:3d} × {n:3d}  ({data['n_atoms']} atoms)")
    print(f"  Pairs used in model    : {len(pairs)} / {data['n_atoms']*(data['n_atoms']-1)//2} total")
    print("-" * 62)
    print(f"  MAE  (Ha/Bohr²)        : {mae_au:>18.6e}")
    print(f"  MAE  (eV/Å²)           : {mae_ev:>18.6e}")
    print(f"  MAE  (N/m)             : {mae_nm:>18.6e}")
    print("-" * 62)
    print(f"  RMSE (Ha/Bohr²)        : {rmse_au:>18.6e}")
    print(f"  RMSE (eV/Å²)           : {rmse_ev:>18.6e}")
    print(f"  RMSE (N/m)             : {rmse_nm:>18.6e}")
    print("-" * 62)
    print(f"  Max  |ΔH| (Ha/Bohr²)  : {max_au:>18.6e}")
    print(f"  Relative MAE           : {rel_mae:>18.2e}  ({rel_mae*100:.2f}%)")
    print("-" * 62)
    print(f"  Assessment : {assessment}")
    print("=" * 62)

    return {
        'mae': float(mae_au),
        'rmse': rmse_au,
        'max_abs': float(max_au),
        'relative_mae': float(rel_mae),
        'assessment': assessment,
    }


# ── Spring constants writer ───────────────────────────────────────────────────

def write_springs_file(filepath, pairs: list, data: dict) -> None:
    """
    Write spring constants to a plain-text file.

    Header contains atom symbols, masses, and coordinates (Ångström, relative
    to atom 0) so the file is fully self-contained for
    :func:`xas_pipeline.chem.springs.parse_springs_file`.

    Format:
        # Atom symbols (index order):
        #    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)
        #        0  Zn        65.380000     -3.610007      -1.884123      -0.535550
        #  ...
        #
        # Spring constants (Ha/Bohr²):
        #   atom1   atom2       k (Ha/Bohr²)
             0       1    0.0123456789
             ...
    """
    atoms      = data['atoms']
    coords_ang = get_coords_ang(data)
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    with open(filepath, 'w') as f:
        # ── Atom symbol / mass / coordinate table ─────────────────────────
        f.write('# Atom symbols and coordinates (index order):\n')
        f.write(f'# {"Index":>8}  {"Symbol":>6}  {"Mass (amu)":>12}'
                f'  {"X (Ang)":>14}  {"Y (Ang)":>14}  {"Z (Ang)":>14}\n')
        f.write('# ' + '-' * 80 + '\n')
        for idx, (sym, mass, xb, yb, zb) in enumerate(atoms):
            x, y, z = coords_ang[idx] - coords_ang[0]
            f.write(f'# {idx:>8}  {sym:>6}  {mass:>12.6f}'
                    f'  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}\n')
        f.write('#\n')

        # ── Spring constant table ─────────────────────────────────────────
        f.write('# Spring constants from ORCA Hessian spring model\n')
        f.write('# Units: Ha/Bohr²\n')
        f.write(f'# {"atom1":>8}  {"atom2":>8}  {"k (Ha/Bohr²)":>22}\n')
        f.write('# ' + '-' * 44 + '\n')

        for (i, j, k_au, k_ev, k_nm, dist_ang, sym_i, sym_j) in sorted_pairs:
            f.write(f'  {i:>8}  {j:>8}  {k_au:>22.10f}\n')

    print(f"  Spring constants written to    : {filepath}")


# ── Library entry points ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpringModelOptions:
    """Which pairs go into a spring model.

    The default (every pair, no cutoff) is what the packaged ``.interp`` files
    were built with -- they carry Zn-H springs at 5 Angstrom. ``bond_factor``
    only matters with ``bonded_only``.
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
    # Reconstruction statistics from mae_check, or None when the check was skipped.
    reconstruction: dict | None


def spring_model_from_hess(
    hess_path,
    springs_path,
    *,
    options: SpringModelOptions | None = None,
    check: bool = True,
) -> SpringModelResult:
    """Compute the spring model of one ORCA ``.hess`` and write it to *springs_path*."""
    options = options or SpringModelOptions()
    hess_path = Path(hess_path)
    springs_path = Path(springs_path)

    print(f"\n  Reading: {hess_path}")
    data = parse_hess_file(hess_path)
    print(f"  Atoms   : {data['n_atoms']}")
    print(f"  Coords  : {data['n_coords']}")

    pairs = compute_spring_constants(
        data,
        cutoff=options.cutoff,
        bonded_only=options.bonded_only,
        bond_factor=options.bond_factor,
    )
    reconstruction = mae_check(data, pairs) if check else None

    springs_path.parent.mkdir(parents=True, exist_ok=True)
    write_springs_file(springs_path, pairs, data)

    return SpringModelResult(
        springs_path=springs_path,
        symbols=[atom[0] for atom in data['atoms']],
        coords_ang=get_coords_ang(data),
        n_pairs=len(pairs),
        reconstruction=reconstruction,
    )


@dataclass(frozen=True)
class InterpBuildResult:
    interp_path: Path
    natural: SpringModelResult
    modified: SpringModelResult
    # Bonds (i, j) present in only one charge state, at the inference bond
    # criterion. Empty in the expected case. The .interp is written either way:
    # inference matches the natural-charge graph, so a bond that only the
    # modified state has still gets interpolated constants, but one that breaks
    # in it is worth a look before the ligand goes into a database.
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
    """Build one ligand's ``.interp`` from its natural- and modified-charge Hessians.

    Writes the two intermediate spring models to *springs_dir* (default: next to
    *interp_path*) as ``<stem>.natural.springs`` and ``<stem>.modified.springs``
    -- the ``.interp`` header names them, so they are its provenance -- then
    combines them with the natural-charge model as model 1.

    *bond_factor* / *bond_cutoff* are only for the bond-graph comparison
    between the two charge states, and should match what inference will use.

    Raises :class:`ValueError` when the two Hessians do not describe the same
    atoms in the same order.
    """
    interp_path = Path(interp_path)
    springs_dir = Path(springs_dir) if springs_dir is not None else interp_path.parent
    stem = interp_path.stem

    natural = spring_model_from_hess(
        natural_hess, springs_dir / f"{stem}.natural.springs", options=options, check=check
    )
    modified = spring_model_from_hess(
        modified_hess, springs_dir / f"{stem}.modified.springs", options=options, check=check
    )

    if natural.symbols != modified.symbols:
        raise ValueError(
            f"{natural_hess} and {modified_hess} do not list the same atoms in the "
            "same order; the .interp pairs would be meaningless."
        )

    natural_bonds = springs.build_bond_set(
        natural.symbols, natural.coords_ang, bond_factor, bond_cutoff
    )
    modified_bonds = springs.build_bond_set(
        modified.symbols, modified.coords_ang, bond_factor, bond_cutoff
    )
    only_natural = sorted(natural_bonds - modified_bonds)
    only_modified = sorted(modified_bonds - natural_bonds)
    for label, bonds in (("natural", only_natural), ("modified", only_modified)):
        for i, j in bonds:
            print(
                f"  WARNING: bond {natural.symbols[i]}{i}-{natural.symbols[j]}{j} "
                f"exists only in the {label}-charge geometry."
            )

    interp_path.parent.mkdir(parents=True, exist_ok=True)
    springs.build_and_save_interp(
        str(natural.springs_path), str(modified.springs_path), str(interp_path)
    )

    return InterpBuildResult(
        interp_path=interp_path,
        natural=natural,
        modified=modified,
        bonds_only_natural=only_natural,
        bonds_only_modified=only_modified,
    )
