"""
Hessian from Spring Model and XYZ Structure
============================================
Vendored from DW_Interpolation/scripts/orca_hess_from_springs_warn.py
(re-vendored 2026-08-20: the eigenvalue floor / spectrum repair and the optional
inter-ligand spring floor). The numerics (Hessian construction, acoustic sum
rule, mass weighting, frequency check, spectrum repair) are kept byte-identical
to upstream; only the argparse CLI was replaced by
:func:`build_hess_from_spring_model`, and upstream's per-eigenvalue print was
collapsed to a count. Upstream stays the source of truth for the science --
re-vendor rather than editing the numerics here.

Reads an XYZ file and a spring model file (produced by spring_model.py),
verifies that the atomic species match, and constructs a new Hessian matrix
using the bond vectors from the XYZ structure and the spring constants from
the model:

    H_ij = -k_ij * U_ij ⊗ U_ij      (off-diagonal 3×3 blocks)
    H_ii = -sum_{j≠i} H_ij           (diagonal, acoustic sum rule)

where:
    k_ij : spring constant for pair (i, j) in Ha/Bohr²
    U_ij : unit vector from atom i to atom j in the NEW structure
            *** always computed from the input XYZ file ***

The resulting Hessian is written to a .hess file. A template .hess file
may optionally be provided to preserve all non-Hessian blocks ($act_atoms,
$act_coord, $atoms, $vibrational_freq, $normal_modes, etc.); otherwise a
minimal .hess file is written containing only the $hessian block.

When a template .hess file is provided, the $atoms block is written using:
    - symbol and mass from the template .hess file
    - x, y, z coordinates from the input XYZ file (converted to Bohr)

Spring constants are read in Ha/Bohr² (the native unit of the spring model
file). XYZ coordinates are in Ångström and are converted to Bohr internally
before computing unit bond vectors.

The spring model file is expected to be in the format produced by
spring_model.py --write-springs, which includes atom symbols, masses,
and coordinates in the header:

    # Atom symbols (index order):
    #    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)
    #        0  Zn        65.380000     -3.61000671    -1.88412310    -0.53555001
    #        1  S         32.060000     ...
    # Spring constants ...
    #   atom1   atom2       k (Ha/Bohr²)
       0       1    0.0123456789

═══════════════════════════════════════════════════════════════════════
FREQUENCY / IMAGINARY-MODE CHECK
═══════════════════════════════════════════════════════════════════════

After the Hessian is built, it is mass-weighted using the atomic masses
from the spring model file header:

    F_ij = H_ij / sqrt(m_i * m_j)          (SI units, 1/s²)

F is diagonalized. Eigenvalues < 0 correspond to imaginary vibrational
frequencies (reported as negative wavenumbers, cm⁻¹). The lowest
--skip-modes (default 6) modes by |eigenvalue| are excluded from the
imaginary-mode warning, since these correspond to the three overall
translations and three overall rotations, which are only approximately
zero due to the acoustic sum rule and finite numerical precision.

A full frequency table is printed (wavenumbers in cm⁻¹, sorted ascending),
with skipped (trans/rot) modes marked, and a clear WARNING is printed if
any non-skipped mode has a negative (imaginary) frequency. Use
``freq_check=False`` to skip this step entirely (useful for very large systems).

═══════════════════════════════════════════════════════════════════════
SPECTRUM REPAIR  (``min_freq_scale``)
═══════════════════════════════════════════════════════════════════════

A spring model interpolated onto a cluster that was never optimized under that
model does not sit at a minimum, so the spectrum can carry imaginary and
near-zero modes -- which the Debye-Waller step downstream turns into garbage or
divergences. When ``min_freq_scale > 0`` (the default), every eigenvalue below

    min_positive = min_freq_scale * max(eigval) / 1000²

is raised to that floor, the six trans/rot modes are pinned to exactly zero, and
the Hessian is rebuilt from the repaired spectrum with its eigenvectors
untouched. The rebuilt Hessian is what gets written. Because the repair rides on
the diagonalization, it does not happen when ``freq_check=False``.

Note that pinning the trans/rot modes to zero preserves the acoustic sum rule
only when the null space really is six-dimensional. A spring model that leaves
the cluster in disconnected fragments has more zero modes than that, and the
extra ones get floored -- see ``add_inter_ligand``, which couples the fragments.

Usage:
    from xas_pipeline.chem import spring_hessian

    result = spring_hessian.build_hess_from_spring_model(
        "cluster.xyz", "spring.model", "cluster.hess"
    )
    print(result.n_imaginary)

The emitted ``.hess`` is read back by
:func:`xas_pipeline.chem.hessian.read_orca_hessian`, the same parser used for a
real ORCA Hessian, so the CORVUS/FEFF stage cannot tell the two apart.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ── Physical constants ────────────────────────────────────────────────────────
BOHR_TO_ANG              = 0.529177210903
ANG_TO_BOHR              = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV            = 27.211396132
HARTREE_TO_J             = 4.3597447222071e-18
BOHR_TO_M                = 5.29177210903e-11
AMU_TO_KG                = 1.66053906660e-27
C_CM_PER_S               = 2.99792458e10          # speed of light, cm/s
HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / BOHR_TO_ANG**2
HARTREE_BOHR2_TO_NM      = HARTREE_TO_J / BOHR_TO_M**2

# Default scale on the eigenvalue floor applied after diagonalization; see
# diagonalize_and_check. 1.0 means "no mode softer than 1/1000 of the stiffest".
MIN_FREQ_SCALE_DEFAULT = 1.0

# Spring constant of a hydrogen bond (Ha/Bohr²), the unit in which the optional
# inter-ligand floor is expressed: add_inter_ligand=1.0 gives every pair at
# least a hydrogen bond's stiffness. Without it the ligand spring models leave
# nothing at all between ligands, so the cluster's spring graph can fall apart
# into independently floating fragments.
INTER_LIGAND_K_HYDROGEN = 0.005


# ── XYZ parser ────────────────────────────────────────────────────────────────

def parse_xyz(filepath: str) -> tuple:
    """
    Parse a standard XYZ file.

    Returns
    -------
    symbols : list of str
    coords  : np.ndarray (N, 3) in Angstrom
    comment : str
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    comment = lines[1].rstrip('\n')
    symbols = []
    coords  = []

    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0].capitalize())
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(symbols) != n_atoms:
        raise ValueError(
            f"XYZ file declares {n_atoms} atoms but only {len(symbols)} "
            f"coordinate lines were found."
        )

    return symbols, np.array(coords), comment


# ── Spring model parser ───────────────────────────────────────────────────────

def parse_springs_file(filepath: str) -> tuple:
    """
    Parse a spring model file produced by spring_model.py --write-springs.

    The file header contains atom symbols, masses, and coordinates (Å),
    so no separate reference XYZ file is needed.

    Returns
    -------
    spring_symbols : list of str    — atom symbols in index order
    spring_masses  : list of float  — atom masses in amu
    spring_coords  : np.ndarray (N, 3) in Angstrom — reference coordinates
    spring_pairs   : list of (i, j, k_au)
                     i, j : int   atom indices (0-based)
                     k_au : float spring constant in Ha/Bohr²
    """
    spring_symbols = []
    spring_masses  = []
    spring_coords  = []
    spring_pairs   = []

    in_atom_block   = False
    in_spring_block = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()

            # ── Detect atom symbol/coordinate table ──────────────────────────
            if 'Atom symbols' in stripped or 'Index  Symbol' in stripped or \
               ('Index' in stripped and 'Symbol' in stripped and 'Mass' in stripped):
                in_atom_block   = True
                in_spring_block = False
                continue

            # ── Detect spring constant block ─────────────────────────────────
            if 'Spring constants' in stripped or \
               ('atom1' in stripped.lower() and 'atom2' in stripped.lower()):
                in_spring_block = True
                in_atom_block   = False
                continue

            # ── Skip blank lines ─────────────────────────────────────────────
            if stripped == '':
                continue

            # ── Parse comment lines (atom table) ─────────────────────────────
            if stripped.startswith('#'):
                content = stripped.lstrip('#').strip()
                parts   = content.split()

                if in_atom_block and len(parts) >= 3:
                    try:
                        idx    = int(parts[0])
                        symbol = parts[1].capitalize()
                        mass   = float(parts[2])

                        # Extend lists if needed
                        while len(spring_symbols) <= idx:
                            spring_symbols.append('')
                            spring_masses.append(0.0)
                            spring_coords.append([0.0, 0.0, 0.0])

                        spring_symbols[idx] = symbol
                        spring_masses[idx]  = mass

                        # Parse coordinates if present (X, Y, Z in Angstrom)
                        if len(parts) >= 6:
                            spring_coords[idx] = [float(parts[3]),
                                                   float(parts[4]),
                                                   float(parts[5])]
                    except (ValueError, IndexError):
                        pass
                continue

            # ── Parse spring constant data line ──────────────────────────────
            if in_spring_block:
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        i    = int(parts[0])
                        j    = int(parts[1])
                        k_au = float(parts[2])
                        spring_pairs.append((i, j, k_au))
                    except ValueError:
                        continue

    if not spring_symbols:
        raise ValueError(
            "No atom symbol table found in the spring model file.\n"
            "Make sure the file was produced by spring_model.py with "
            "the --write-springs flag."
        )

    if not spring_pairs:
        raise ValueError(
            "No spring constant pairs found in the spring model file."
        )

    return spring_symbols, spring_masses, np.array(spring_coords), spring_pairs


# ── Atom validation ────────────────────────────────────────────────────────────

def validate_atoms(xyz_symbols: list, spring_symbols: list) -> list:
    """
    Check that the XYZ structure and spring model have matching atomic
    species, line by line.

    Returns
    -------
    mismatches : list of (index, xyz_symbol, spring_symbol)
                 Empty list if everything matches.
    """
    mismatches = []

    if len(xyz_symbols) != len(spring_symbols):
        raise ValueError(
            f"Atom count mismatch: XYZ file has {len(xyz_symbols)} atoms, "
            f"spring model has {len(spring_symbols)} atoms."
        )

    for idx, (xs, ss) in enumerate(zip(xyz_symbols, spring_symbols)):
        if xs.capitalize() != ss.capitalize():
            mismatches.append((idx, xs, ss))

    return mismatches


# ── Column-header detection (for .hess template parsing) ──────────────────────

def _is_col_header(tokens, n):
    """
    Return True if tokens form a Hessian column-header line: all tokens are
    non-negative integers < n, strictly ascending, no decimal points.
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


# ── .hess template parser ──────────────────────────────────────────────────────

def parse_hess_template(filepath: str) -> dict:
    """
    Parse an ORCA .hess template file, extracting enough information to
    reproduce its exact format when writing a new Hessian, while keeping
    all other blocks verbatim.

    Returns a dict with keys:
        'raw_lines'       : list of str — every line in the file
        'n_coords'        : int
        'n_atoms'         : int
        'atoms'           : list of (symbol, mass, x_bohr, y_bohr, z_bohr)
        'atoms_start'     : int — line index of '$atoms'
        'atoms_end'       : int — first line index after the atoms block
        'hess_start'      : int — line index of '$hessian'
        'hess_end'        : int — first line index after the hessian block
        'hess_block_size' : int — columns per block (detected)
        'hess_decimals'   : int — decimal places in values (detected)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {
        'raw_lines'       : lines,
        'n_coords'        : 0,
        'n_atoms'         : 0,
        'atoms'           : [],
        'atoms_start'     : None,
        'atoms_end'       : None,
        'hess_start'      : None,
        'hess_end'        : None,
        'hess_block_size' : 6,
        'hess_decimals'   : 8,
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '$act_coord':
            i += 1
            data['n_coords'] = int(lines[i].strip())
            i += 1
            continue

        if line == '$atoms':
            data['atoms_start'] = i
            i += 1
            data['n_atoms'] = int(lines[i].strip())
            i += 1
            atom_count = 0
            while i < len(lines) and atom_count < data['n_atoms']:
                parts = lines[i].split()
                if len(parts) >= 5:
                    sym  = parts[0]
                    mass = float(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    data['atoms'].append((sym, mass, x, y, z))
                    atom_count += 1
                i += 1
            data['atoms_end'] = i
            continue

        if line == '$hessian':
            data['hess_start'] = i
            i += 1
            n = int(lines[i].strip())
            if data['n_coords'] == 0:
                data['n_coords'] = n
            i += 1
            data_start_idx = i

            block_size  = None
            decimals    = 8
            col_indices = None

            while i < len(lines):
                l2 = lines[i]
                s2 = l2.strip()

                if not s2:
                    i += 1
                    continue

                if s2.startswith('$') and i != data_start_idx:
                    break

                tokens = s2.split()

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
                    int(tokens[0])
                except ValueError:
                    i += 1
                    continue

                for val_str in tokens[1:]:
                    clean = val_str.lstrip('+-')
                    if '.' in clean and 'e' not in clean.lower():
                        decimals = len(clean.split('.')[-1].rstrip())
                        break
                    elif '.' in clean and 'e' in clean.lower():
                        decimals = len(clean.split('.')[-1].split('e')[0].split('E')[0])
                        break

                i += 1

            data['hess_block_size'] = block_size or 6
            data['hess_decimals']   = decimals
            data['hess_end']        = i
            continue

        i += 1

    return data


# ── Hessian block writer (scientific notation, matches detected format) ───────

def _format_hessian_block(H: np.ndarray, block_size: int, decimals: int) -> list:
    """Format an (n,n) Hessian matrix into ORCA-style $hessian block lines."""
    n = H.shape[0]
    val_width = decimals + 8   # sign + digit + '.' + decimals + 'e±XX'
    lines = [f'{n}\n']

    for col_start in range(0, n, block_size):
        col_end = min(col_start + block_size, n)
        cols    = list(range(col_start, col_end))

        header = '  ' + '   '.join(f'{c:>{val_width - 3}d}' for c in cols)
        lines.append(header + '\n')

        for row in range(n):
            row_str = f'{row:>6d}'
            for c in cols:
                row_str += f' {H[row, c]:>{val_width}.{decimals}e}'
            lines.append(row_str + '\n')

    return lines


# ── Hessian construction from spring model ─────────────────────────────────────

def build_hessian(coords_ang: np.ndarray, spring_pairs: list, n_atoms: int) -> np.ndarray:
    """
    Build the 3N x 3N Hessian from spring constants and bond vectors.

    Bond vectors are computed EXCLUSIVELY from coords_ang (the input XYZ
    structure) — this is the only place bond vectors are computed, and
    spring model reference coordinates are never used here.

    Parameters
    ----------
    coords_ang   : (N, 3) ndarray, Angstrom — input XYZ structure
    spring_pairs : list of (i, j, k_au)     — k in Ha/Bohr²
    n_atoms      : int

    Returns
    -------
    H : (3N, 3N) ndarray, Hartree/Bohr²
    """
    coords_bohr = coords_ang * ANG_TO_BOHR
    n_coords    = 3 * n_atoms
    H           = np.zeros((n_coords, n_coords))

    for (i, j, k_au) in spring_pairs:
        if i == j or k_au == 0.0:
            continue

        # ── Bond vector from input XYZ structure ─────────────────────────────
        # This is the ONLY place bond vectors are computed.
        # coords_bohr comes directly from the input XYZ file.
        r_i  = coords_bohr[i]
        r_j  = coords_bohr[j]
        diff = r_j - r_i
        dist = np.linalg.norm(diff)

        if dist < 1e-10:
            continue

        U      = diff / dist
        block  = -k_au * np.outer(U, U)

        H[3*i:3*i+3, 3*j:3*j+3] += block
        H[3*j:3*j+3, 3*i:3*i+3] += block.T
        H[3*i:3*i+3, 3*i:3*i+3] -= block
        H[3*j:3*j+3, 3*j:3*j+3] -= block

    return H


# ── Frequency / imaginary-mode check ───────────────────────────────────────────

def diagonalize_and_check(H: np.ndarray, masses_amu: list,
                          n_skip_modes: int = 6,
                          fix_neg_freq: float = MIN_FREQ_SCALE_DEFAULT) -> tuple:
    """
    Mass-weight and diagonalize the Hessian; report vibrational
    wavenumbers (cm⁻¹) and flag imaginary (negative eigenvalue) modes.

    F_ij = H_ij / sqrt(m_i * m_j)     (SI units, 1/s²)

    Parameters
    ----------
    H            : (3N, 3N) ndarray, Hartree/Bohr²
    masses_amu   : list of float, length N — atomic masses in amu
    n_skip_modes : int — number of lowest-|freq| modes excluded from the
                   imaginary-frequency warning (translation + rotation)
    fix_neg_freq : float — scale on the eigenvalue floor used to repair
                   imaginary/near-zero modes. Every eigenvalue below
                   ``fix_neg_freq * max(eigval) / 1000²`` is raised to that
                   floor (i.e. no mode is left softer than 1/1000 of the
                   stiffest one, times this scale), the n_skip_modes trans/rot
                   modes are set to exactly zero, and the Hessian is rebuilt
                   from the repaired spectrum. Pass 0 (or less) to leave the
                   spectrum alone.

    Returns
    -------
    wavenumbers    : (3N,) ndarray, cm⁻¹, sorted ascending
                     (negative values denote imaginary frequencies)
    skip_mask      : (3N,) boolean ndarray — True for modes excluded as
                     translation/rotation
    n_imaginary    : int — number of non-skipped modes with negative freq
    H_fixed        : (3N, 3N) ndarray, Hartree/Bohr² — H rebuilt from the
                     repaired eigenvalues, or H itself when fix_neg_freq <= 0
    """
    n_atoms = len(masses_amu)
    n_coords = 3 * n_atoms

    H_SI = H * HARTREE_TO_J / BOHR_TO_M**2       # J/m² = kg/s²
    m_kg = np.array(masses_amu) * AMU_TO_KG

    m_sqrt_inv = 1.0 / np.sqrt(m_kg)
    m_vec      = np.repeat(m_sqrt_inv, 3)         # length 3N, per-coordinate

    F = H_SI * np.outer(m_vec, m_vec)             # SI, units 1/s²
    F = 0.5 * (F + F.T)                           # enforce exact symmetry

    eigvals, eigvecs = np.linalg.eigh(F)          # ascending order, 1/s²

    omega       = np.sqrt(np.abs(eigvals))        # rad/s
    nu_hz       = omega / (2.0 * np.pi)           # Hz
    wavenumbers = np.sign(eigvals) * nu_hz / C_CM_PER_S   # cm⁻¹

    # Identify translation/rotation modes: lowest |eigenvalue|
    order_by_abs = np.argsort(np.abs(eigvals))
    skip_indices = order_by_abs[:n_skip_modes]
    skip_mask    = np.zeros(n_coords, dtype=bool)
    skip_mask[skip_indices] = True

    n_imaginary = int(np.sum((wavenumbers < 0.0) & (~skip_mask)))

    # ── Repair the spectrum ───────────────────────────────────────────────────
    # A spring model interpolated onto an unoptimized cluster is not at a
    # minimum, so it can carry imaginary and near-zero modes. Those give
    # nonsensical (or divergent) Debye-Waller factors downstream, so every
    # eigenvalue is floored and the trans/rot modes pinned to exactly zero;
    # rebuilding H from the repaired spectrum keeps the eigenvectors untouched.
    if fix_neg_freq > 0.0:
        min_positive  = fix_neg_freq * np.max(eigvals) / 1000.0**2
        below         = eigvals < min_positive
        eigvals_fixed = np.where(below, min_positive, eigvals)
        eigvals_fixed[skip_indices] = 0.0     # trans/rot: exactly zero
        n_raised = int(np.sum(below & (~skip_mask)))
        if n_raised:
            # Upstream printed one line per replaced eigenvalue; the count and
            # the floor say the same thing without 3N lines in a job log.
            print(f"  Eigenvalue floor       : {n_raised} mode(s) raised to "
                  f"{min_positive:.6e} 1/s^2")

        F_fixed = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
        F_fixed = 0.5 * (F_fixed + F_fixed.T)     # undo reconstruction drift

        # Un-mass-weight and return to Hartree/Bohr².
        M_outer = np.outer(m_vec, m_vec)
        H_fixed = np.zeros_like(F_fixed, dtype=float)
        np.divide(F_fixed, M_outer, out=H_fixed, where=M_outer != 0)
        H_fixed = H_fixed * BOHR_TO_M**2 / HARTREE_TO_J
    else:
        H_fixed = H

    return wavenumbers, skip_mask, n_imaginary, H_fixed


def print_frequency_report(wavenumbers: np.ndarray, skip_mask: np.ndarray,
                           n_imaginary: int) -> None:
    """Print a formatted table of vibrational wavenumbers with warnings."""
    n = len(wavenumbers)

    print()
    print("=" * 60)
    print("  Hessian Diagonalization — Frequency Check")
    print("=" * 60)
    print(f"  {'Mode':>6}   {'Wavenumber (cm-1)':>18}   Note")
    print("  " + "-" * 44)

    for idx in range(n):
        note = ''
        if skip_mask[idx]:
            note = '(trans/rot)'
        elif wavenumbers[idx] < 0.0:
            note = '*** IMAGINARY ***'
        print(f"  {idx:>6}   {wavenumbers[idx]:>18.4f}   {note}")

    print("  " + "-" * 44)

    if n_imaginary > 0:
        print(f"  *** WARNING: {n_imaginary} imaginary frequency mode(s) "
              f"detected (excluding {int(np.sum(skip_mask))} trans/rot "
              f"modes) ***")
        print("  This Hessian does NOT correspond to a stable minimum.")
    else:
        print(f"  No imaginary frequencies detected "
              f"(excluding {int(np.sum(skip_mask))} trans/rot modes).")

    print("=" * 60)
    print()


# ── Output writer ──────────────────────────────────────────────────────────────

def write_hess_file(output_path: str, coords_ang: np.ndarray, H: np.ndarray,
                    n_atoms: int, template: dict = None,
                    fallback_symbols: list = None,
                    fallback_masses: list = None) -> None:
    """
    Write the reconstructed Hessian to a .hess file.

    If a template is provided, all non-Hessian blocks are copied verbatim,
    and the $atoms block is written using symbol/mass from the TEMPLATE
    and x,y,z coordinates from coords_ang (input XYZ — always).

    If no template is provided, a minimal .hess file is written using
    fallback_symbols / fallback_masses (typically from the spring model
    file) together with coords_ang for the $atoms block.
    """
    coords_bohr = coords_ang * ANG_TO_BOHR
    n_coords    = 3 * n_atoms

    if template is not None:
        lines = template['raw_lines']
        block_size = template['hess_block_size']
        decimals   = template['hess_decimals']

        out_lines = []
        i = 0
        while i < len(lines):
            if template['atoms_start'] is not None and i == template['atoms_start']:
                out_lines.append('$atoms\n')
                out_lines.append(f'{n_atoms}\n')
                for k in range(n_atoms):
                    sym  = template['atoms'][k][0]   # symbol ← always from .hess
                    mass = template['atoms'][k][1]   # mass   ← always from .hess
                    x, y, z = coords_bohr[k]          # xyz    ← always from XYZ
                    out_lines.append(
                        f'  {sym:<4s}  {mass:>12.6f}  '
                        f'{x:>18.10f}  {y:>18.10f}  {z:>18.10f}\n'
                    )
                i = template['atoms_end']
                continue

            if template['hess_start'] is not None and i == template['hess_start']:
                out_lines.append('$hessian\n')
                out_lines.extend(_format_hessian_block(H, block_size, decimals))
                i = template['hess_end']
                continue

            out_lines.append(lines[i])
            i += 1

        with open(output_path, 'w') as f:
            f.writelines(out_lines)

    else:
        with open(output_path, 'w') as f:
            f.write('\n$orca_hessian_file\n\n')
            f.write('$act_atoms\n')
            f.write(f'{n_atoms}\n\n')
            f.write('$act_coord\n')
            f.write(f'{n_coords}\n\n')
            f.write('$hessian\n')
            for line in _format_hessian_block(H, 5, 8):
                f.write(line)
            f.write('\n$atoms\n')
            f.write(f'{n_atoms}\n')
            for k in range(n_atoms):
                sym  = fallback_symbols[k] if fallback_symbols else 'X'
                mass = fallback_masses[k]  if fallback_masses  else 0.0
                x, y, z = coords_bohr[k]
                f.write(
                    f'  {sym:<4s}  {mass:>12.6f}  '
                    f'{x:>18.10f}  {y:>18.10f}  {z:>18.10f}\n'
                )


# ── Reporting helpers ───────────────────────────────────────────────────────────

def print_springs_summary(coords_ang: np.ndarray, symbols: list,
                          spring_pairs: list) -> None:
    """Print spring constants with bond lengths from the input structure."""
    print()
    print("=" * 70)
    print("  Spring Constants and Bond Lengths (from input XYZ structure)")
    print("=" * 70)
    print(f"  {'i':>4} {'j':>4}  {'Sym_i':>5} {'Sym_j':>5}  "
          f"{'k (Ha/Bohr2)':>16}  {'d (Ang)':>10}")
    print("  " + "-" * 64)

    for (i, j, k_au) in spring_pairs:
        d = np.linalg.norm(coords_ang[j] - coords_ang[i])
        si = symbols[i] if i < len(symbols) else '?'
        sj = symbols[j] if j < len(symbols) else '?'
        print(f"  {i:>4} {j:>4}  {si:>5} {sj:>5}  {k_au:>16.8f}  {d:>10.4f}")

    print("=" * 70)
    print()


def print_hessian_matrix(H: np.ndarray) -> None:
    print()
    print("=" * 70)
    print("  Reconstructed Hessian (Hartree/Bohr²)")
    print("=" * 70)
    with np.printoptions(precision=6, suppress=False, linewidth=200):
        print(H)
    print("=" * 70)
    print()


# ── Library entry point ───────────────────────────────────────────────────────
# Replaces the upstream argparse CLI. The atom-symbol mismatch that upstream
# handled with sys.exit(1) raises ValueError here so the calling stage can fail
# the job with a diagnostic instead of killing the interpreter.


@dataclass
class HessianResult:
    """What :func:`build_hess_from_spring_model` produced, for logging/asserts."""

    output_path: Path
    n_atoms: int
    n_pairs: int
    symmetry_error: float
    wavenumbers: "np.ndarray | None"
    #: Imaginary modes left in the Hessian that was written, i.e. after the
    #: eigenvalue repair when ``min_freq_scale > 0``. None when freq_check=False.
    n_imaginary: int | None
    #: Imaginary modes in the raw spring Hessian, before any repair. This is the
    #: number that says something about the spring model; ``n_imaginary`` says
    #: whether the repair worked.
    n_imaginary_raw: int | None = None


def build_hess_from_spring_model(
    xyz_path,
    springs_path,
    output_path,
    *,
    template_path=None,
    zero_negative: bool = False,
    add_inter_ligand: float = 0.0,
    freq_check: bool = True,
    skip_modes: int = 6,
    min_freq_scale: float = MIN_FREQ_SCALE_DEFAULT,
) -> HessianResult:
    """Build an ORCA-format ``.hess`` from a geometry and a spring model.

    Bond vectors always come from *xyz_path*; the spring model supplies only the
    force constants (and the symbols/masses used for the ``$atoms`` block when
    no *template_path* is given). Returns a :class:`HessianResult`;
    ``n_imaginary`` is ``None`` when ``freq_check`` is False.

    *add_inter_ligand*, when positive, floors every pair constant at
    ``add_inter_ligand * INTER_LIGAND_K_HYDROGEN``, so pairs the ligand spring
    models say nothing about (in particular, atoms in different ligands) are
    still weakly coupled.

    *min_freq_scale*, when positive, repairs the spectrum after the frequency
    check and writes the repaired Hessian -- see :func:`diagonalize_and_check`.
    It only applies when *freq_check* is on, since the repair is a by-product of
    the diagonalization.

    Raises :class:`ValueError` if the geometry and the spring model disagree on
    atomic symbols -- that means the spring model was built for a different
    structure and the resulting Hessian would be meaningless.
    """
    xyz_symbols, xyz_coords, _comment = parse_xyz(str(xyz_path))
    spring_symbols, spring_masses, _spring_coords, spring_pairs = parse_springs_file(
        str(springs_path)
    )

    n_atoms = len(xyz_symbols)

    mismatches = validate_atoms(xyz_symbols, spring_symbols)
    if mismatches:
        detail = "; ".join(
            f"atom {idx}: xyz={xs}, spring model={ss}" for idx, xs, ss in mismatches
        )
        raise ValueError(
            f"Atomic symbol mismatch between {xyz_path} and {springs_path}: {detail}"
        )

    print(f"  Input structure       : {xyz_path}  ({n_atoms} atoms)")
    print(f"  Spring model file     : {springs_path}  ({len(spring_pairs)} pairs)")
    print(f"  Atom validation       : OK - all {n_atoms} symbols match")

    if zero_negative:
        n_neg = sum(1 for (_, _, k) in spring_pairs if k < 0)
        spring_pairs = [(i, j, max(k, 0.0)) for (i, j, k) in spring_pairs]
        print(f"  zero_negative         : {n_neg} negative spring constant(s) set to zero.")

    if add_inter_ligand > 0.0:
        k_floor = INTER_LIGAND_K_HYDROGEN * add_inter_ligand
        n_floored = sum(1 for (_, _, k) in spring_pairs if k < k_floor)
        spring_pairs = [(i, j, max(k_floor, k)) for (i, j, k) in spring_pairs]
        print(f"  add_inter_ligand      : {n_floored} pair(s) raised to "
              f"k = {k_floor:.6g} Ha/Bohr^2.")

    H = build_hessian(coords_ang=xyz_coords, spring_pairs=spring_pairs, n_atoms=n_atoms)

    sym_err = float(np.max(np.abs(H - H.T)))
    print(f"  Hessian symmetry check : max|H - H^T| = {sym_err:.3e}")

    H_out = H
    wavenumbers = None
    n_imaginary = None
    n_imaginary_raw = None
    if freq_check:
        raw_wavenumbers, raw_skip_mask, n_imaginary_raw, H_fixed = diagonalize_and_check(
            H, spring_masses, n_skip_modes=skip_modes, fix_neg_freq=min_freq_scale
        )
        wavenumbers, skip_mask, n_imaginary = raw_wavenumbers, raw_skip_mask, n_imaginary_raw

        if min_freq_scale > 0.0:
            # Re-diagonalize the repaired Hessian: the frequencies reported (and
            # n_imaginary) must describe the Hessian actually written, not the
            # one it was derived from. The second repair is a no-op on a
            # spectrum that is already floored.
            wavenumbers, skip_mask, n_imaginary, _H2 = diagonalize_and_check(
                H_fixed, spring_masses, n_skip_modes=skip_modes,
                fix_neg_freq=min_freq_scale
            )
            H_out = H_fixed

            # A well-conditioned spring model needs no repair, and printing the
            # same 3N-line table twice only buries the interesting case. Show
            # the before/after pair only when the floor actually moved something.
            # atol is well below the weakest interpolated constant (~1e-4) and
            # well above the reconstruction's round-trip noise (~1e-15).
            if not np.allclose(H_fixed, H, rtol=1e-8, atol=1e-12):
                print("  Frequencies of the raw spring Hessian:")
                print_frequency_report(raw_wavenumbers, raw_skip_mask, n_imaginary_raw)
                print("  Frequencies after the eigenvalue floor:")

        print_frequency_report(wavenumbers, skip_mask, n_imaginary)

    template = None
    if template_path is not None:
        template = parse_hess_template(str(template_path))
        print(f"  Template .hess file   : {template_path}")
        if template["n_atoms"] != n_atoms:
            print(
                f"  WARNING: template has {template['n_atoms']} atoms, "
                f"input structure has {n_atoms} atoms."
            )

    write_hess_file(
        output_path=str(output_path),
        coords_ang=xyz_coords,
        H=H_out,
        n_atoms=n_atoms,
        template=template,
        fallback_symbols=spring_symbols,
        fallback_masses=spring_masses,
    )
    print(f"  Hessian written to    : {output_path}")

    return HessianResult(
        output_path=Path(output_path),
        n_atoms=n_atoms,
        n_pairs=len(spring_pairs),
        symmetry_error=sym_err,
        wavenumbers=wavenumbers,
        n_imaginary=n_imaginary,
        n_imaginary_raw=n_imaginary_raw,
    )
