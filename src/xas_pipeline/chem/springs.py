
"""
Spring Model Interpolator
=========================
Vendored from DW_Interpolation/scripts/orca_interpolate_spring_ligands2.py.
The numerics (bonding criteria, log/linear method selection, subgraph
isomorphism, merge rules) are kept byte-identical to the upstream script so the
interpolated spring constants are reproducible; only the argparse CLI was
replaced by :func:`interpolate_to_spring_model` and one misplaced per-pair
progress ``print`` was moved out of its loop. Upstream stays the source of
truth for the science -- re-vendor rather than editing the numerics here.

Reads two spring model files (produced by spring_model.py) and an input
XYZ structure. Each spring model file contains both the spring constants
AND the reference coordinates in its header, so no separate reference
XYZ files are needed.

For each atom pair (i, j), the spring constant is interpolated based on
the bond length in the input structure. The interpolation method is chosen
per bond:

Log-space interpolation is used when the spring constant DECREASES as
bond length increases (physically consistent with bond weakening) AND
both spring constants are positive AND the pair is covalently bonded
in the input structure AND neither atom is hydrogen:

    dk/dd < 0  AND  k1 > 0  AND  k2 > 0  AND  bonded(i,j)  AND  no H  →  log-space interpolation

    alpha_ij  = (d_ij - d1_ij) / (d2_ij - d1_ij)
    log_k_ij  = (1 - alpha_ij) * log(k1_ij) + alpha_ij * log(k2_ij)
    k_ij      = exp(log_k_ij)

Linear interpolation is used in all other cases:

    k_ij = (1 - alpha_ij) * k1_ij + alpha_ij * k2_ij

This includes:
  - dk/dd >= 0 (spring constant grows with bond length)
  - Both k < 0 (both negative — linear is used)
  - Mixed signs (one positive, one negative)
  - Either zero
  - Pair is not covalently bonded in the input structure
  - Either atom is hydrogen

where:
    d1_ij : bond length from reference coordinates in spring model 1 (Å)
    d2_ij : bond length from reference coordinates in spring model 2 (Å)
    d_ij  : bond length in input structure                            (Å)
    k1_ij : spring constant from spring model 1                (Ha/Bohr²)
    k2_ij : spring constant from spring model 2                (Ha/Bohr²)

Each bond has its own interpolation parameter alpha_ij, so bonds that
change length by different amounts are each interpolated correctly.

Pairs that exist in only one spring model are skipped with a warning.
Extrapolation (alpha < 0 or alpha > 1) is allowed by default but reported.

ALL pairs (i < j) for all N atoms are written to the output file.
Pairs with no interpolation data are written with k = 0.

═══════════════════════════════════════════════════════════════════════
MULTI-LIGAND MODE  (subgraph isomorphism search)  -- what the pipeline uses
═══════════════════════════════════════════════════════════════════════

Rather than matching a whole cluster to a single spring model, each ligand's
bond network is located as an induced subgraph of the input structure, once per
occurrence, and its pair constants are interpolated to the observed bond
lengths. Pairs claimed by more than one match keep the first assignment.

Bonding criterion (see :class:`InterpOptions`):
    bond_factor F  : pair (i,j) is bonded if d <= F * (r_i + r_j), with r the
                     covalent radii (default F = 1.2)
    bond_cutoff C  : pair (i,j) is bonded if d <= C (Angstrom), regardless of
                     atom type; overrides bond_factor when set

Usage:
    from xas_pipeline.chem import springs

    springs.interpolate_to_spring_model(
        "cluster.xyz",
        ["ZnHis.interp", "ZnHis_2.interp", "ZnCys.interp"],
        "spring.model",
    )

The ``.interp`` constants files are pre-built upstream (the script's
``--build-interp`` mode) from two spring models per ligand; the pipeline ships
them as package data under ``data/interp-ligands/`` and does not rebuild them.
The upstream single-ligand and ``--ligand`` modes are not exposed here.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ── Physical constants ────────────────────────────────────────────────────────
BOHR_TO_ANG              = 0.529177210903
ANG_TO_BOHR              = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV            = 27.211396132
HARTREE_TO_J             = 4.3597447222071e-18
BOHR_TO_M                = 5.29177210903e-11
HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / BOHR_TO_ANG**2
HARTREE_BOHR2_TO_NM      = HARTREE_TO_J / BOHR_TO_M**2


# ── Covalent radii (Å) — Alvarez 2008, DOI: 10.1039/b801115j ─────────────────
COVALENT_RADII = {
    'H' : 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B' : 0.84, 'C' : 0.76, 'N' : 0.71,
    'O' : 0.66, 'F' : 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P' : 1.07,
    'S' : 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K' : 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V' : 1.53,
    'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24,
    'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
    'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y' : 1.90, 'Zr': 1.75, 'Nb': 1.64,
    'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
    'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39,
    'Te': 1.38, 'I' : 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03,
    'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90,
    'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W' : 1.62,
    'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36,
    'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48,
    'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U' : 1.96, 'Np': 1.90,
    'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69,
}
DEFAULT_RADIUS = 1.50


# ── Atomic masses (amu) ───────────────────────────────────────────────────────
ATOMIC_MASSES = {
    'H' : 1.008,   'He': 4.003,
    'Li': 6.941,   'Be': 9.012,   'B' : 10.811,  'C' : 12.011,
    'N' : 14.007,  'O' : 15.999,  'F' : 18.998,  'Ne': 20.180,
    'Na': 22.990,  'Mg': 24.305,  'Al': 26.982,  'Si': 28.086,
    'P' : 30.974,  'S' : 32.06,   'Cl': 35.45,   'Ar': 39.948,
    'K' : 39.098,  'Ca': 40.078,  'Sc': 44.956,  'Ti': 47.867,
    'V' : 50.942,  'Cr': 51.996,  'Mn': 54.938,  'Fe': 55.845,
    'Co': 58.933,  'Ni': 58.693,  'Cu': 63.546,  'Zn': 65.38,
    'Ga': 69.723,  'Ge': 72.63,   'As': 74.922,  'Se': 78.971,
    'Br': 79.904,  'Kr': 83.798,  'Rb': 85.468,  'Sr': 87.62,
    'Y' : 88.906,  'Zr': 91.224,  'Nb': 92.906,  'Mo': 95.96,
    'Tc': 98.0,    'Ru': 101.07,  'Rh': 102.906, 'Pd': 106.42,
    'Ag': 107.868, 'Cd': 112.414, 'In': 114.818, 'Sn': 118.710,
    'Sb': 121.760, 'Te': 127.60,  'I' : 126.904, 'Xe': 131.293,
    'Cs': 132.905, 'Ba': 137.327, 'La': 138.905,
    'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242,
    'Sm': 150.36,  'Eu': 151.964, 'Gd': 157.25,
    'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930,
    'Er': 167.259, 'Tm': 168.934, 'Yb': 173.045, 'Lu': 174.967,
    'Hf': 178.49,  'Ta': 180.948, 'W' : 183.84,
    'Re': 186.207, 'Os': 190.23,  'Ir': 192.217,
    'Pt': 195.084, 'Au': 196.967, 'Hg': 200.592,
    'Tl': 204.38,  'Pb': 207.2,   'Bi': 208.980,
}


# ── XYZ parser ────────────────────────────────────────────────────────────────

def parse_xyz(filepath):
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
        raise ValueError(f"XYZ declares {n_atoms} atoms but found {len(symbols)}")
    return symbols, np.array(coords), comment


# ── Spring model parser ───────────────────────────────────────────────────────

def parse_springs_file(filepath):
    """
    Parse a spring model file produced by spring_model.py --write-springs.

    Returns
    -------
    spring_symbols : list of str
    spring_masses  : list of float  (amu)
    spring_coords  : np.ndarray (N, 3)  Angstrom
    spring_pairs   : list of (i, j, k_au)
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

            if 'Atom symbols' in stripped or 'Index  Symbol' in stripped or \
               ('Index' in stripped and 'Symbol' in stripped and 'Mass' in stripped):
                in_atom_block   = True
                in_spring_block = False
                continue

            if 'Spring constants' in stripped or \
               ('atom1' in stripped.lower() and 'atom2' in stripped.lower()):
                in_spring_block = True
                in_atom_block   = False
                continue

            if stripped == '':
                continue

            if stripped.startswith('#'):
                content = stripped.lstrip('#').strip()
                parts   = content.split()

                if in_atom_block and len(parts) >= 3:
                    try:
                        idx    = int(parts[0])
                        symbol = parts[1].capitalize()
                        mass   = float(parts[2])

                        while len(spring_symbols) <= idx:
                            spring_symbols.append('')
                            spring_masses.append(0.0)
                            spring_coords.append([0.0, 0.0, 0.0])

                        spring_symbols[idx] = symbol
                        spring_masses[idx]  = mass

                        if len(parts) >= 6:
                            spring_coords[idx] = [float(parts[3]),
                                                   float(parts[4]),
                                                   float(parts[5])]
                    except (ValueError, IndexError):
                        pass
                continue

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
            f"No atom symbol table found in '{filepath}'.\n"
            "Make sure the file was produced by the updated spring_model.py "
            "that writes atom symbols and coordinates in the header."
        )

    if not any(c != [0.0, 0.0, 0.0] for c in spring_coords):
        raise ValueError(
            f"No coordinates found in '{filepath}'.\n"
            "Make sure the file was produced by the updated spring_model.py "
            "that writes coordinates in the header."
        )

    return spring_symbols, spring_masses, np.array(spring_coords), spring_pairs


# ── Bonding utilities ─────────────────────────────────────────────────────────

def is_bonded(sym_i, sym_j, dist, bond_factor=1.2, bond_cutoff=None):
    if bond_cutoff is not None:
        return dist <= bond_cutoff
    ri = COVALENT_RADII.get(sym_i, DEFAULT_RADIUS)
    rj = COVALENT_RADII.get(sym_j, DEFAULT_RADIUS)
    return dist <= bond_factor * (ri + rj)


def build_bond_set(symbols, coords, bond_factor=1.2, bond_cutoff=None):
    n = len(symbols)
    bonds = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if is_bonded(symbols[i], symbols[j], d, bond_factor, bond_cutoff):
                bonds.add((i, j))
    return bonds


def build_adjacency(n, bonds):
    adj = dict()
    for i in range(n):
        adj[i] = set()

    for (i, j) in bonds:
        adj[i].add(j) 
        adj[j].add(i)
    return adj


# ── Interpolation ─────────────────────────────────────────────────────────────

def interp_pair(k1, k2, d1, d2, d_inp, sym_i, sym_j,
                bonded, allow_extrap, verbose_info):
    """
    Interpolate (or extrapolate) the spring constant for one pair.

    Returns (k_interp, alpha, method_label, warning_str or None)
    """
    if abs(d2 - d1) < 1e-12:
        return 0.5 * (k1 + k2), 0.5, 'linear(d1=d2)', None

    alpha = (d_inp - d1) / (d2 - d1)

    warn = None
    if not allow_extrap and (alpha < 0.0 or alpha > 1.0):
        warn = (f"  Extrapolation blocked for pair ({verbose_info}): "
                f"alpha={alpha:.4f}")
        alpha = max(0.0, min(1.0, alpha))

    # ── Determine interpolation method ────────────────────────────────────────
    has_H      = (sym_i == 'H' or sym_j == 'H')
    dk_dd_neg  = ((k2 - k1) * (d2 - d1) < 0)   # k decreases as d increases
    both_pos   = (k1 > 0 and k2 > 0)
    long_bond  = (alpha > 1.0) 

    #use_log = (both_pos and dk_dd_neg and bonded and not has_H)
    use_log = (both_pos and dk_dd_neg)

    if use_log:
        log_k = (1.0 - alpha) * np.log(k1) + alpha * np.log(k2)
        return np.exp(log_k), alpha, 'log+', warn

    # Linear fallback
    if has_H:
        method = 'linear(H)'
    elif not bonded:
        method = 'linear(non-bonded)'
    elif not dk_dd_neg:
        method = 'linear(dk/dd>=0)'
    elif not both_pos:
        if k1 < 0 and k2 < 0:
            method = 'linear(both-)'
        elif k1 == 0 or k2 == 0:
            method = 'linear(zero)'
        else:
            method = 'linear(mixed)'
    else:
        method = 'linear'

    k_interp = (1.0 - alpha) * k1 + alpha * k2
    return k_interp, alpha, method, warn


# ── Graph isomorphism (bond network matching) ─────────────────────────────────

def find_matching_ordering(ref_symbols, ref_adj, inp_symbols, inp_adj):
    """
    Find a permutation of inp atoms such that the bond network matches ref.
    Returns the permutation as a list p where p[ref_idx] = inp_idx,
    or None if no match found.
    Uses backtracking VF2-style search constrained to same-element swaps.
    """
    n = len(ref_symbols)
    if len(inp_symbols) != n:
        return None
    if Counter(ref_symbols) != Counter(inp_symbols):
        return None

    # Degree sequences per element must match
    ref_deg = [len(ref_adj[i]) for i in range(n)]
    inp_deg = [len(inp_adj[i]) for i in range(n)]
    for elem in set(ref_symbols):
        ref_d = sorted(ref_deg[i] for i in range(n) if ref_symbols[i] == elem)
        inp_d = sorted(inp_deg[i] for i in range(n) if inp_symbols[i] == elem)
        if ref_d != inp_d:
            return None

    mapping = {}   # ref_idx → inp_idx
    used    = set()

    def backtrack(ref_idx):
        if ref_idx == n:
            return True
        sym  = ref_symbols[ref_idx]
        rdeg = len(ref_adj[ref_idx])
        for inp_idx in range(n):
            if inp_idx in used:
                continue
            if inp_symbols[inp_idx] != sym:
                continue
            if len(inp_adj[inp_idx]) != rdeg:
                continue
            # Check neighbor consistency
            ok = True
            for prev_ref, prev_inp in mapping.items():
                if prev_ref in ref_adj[ref_idx]:
                    if inp_idx not in inp_adj[prev_inp]:
                        ok = False
                        break
                else:
                    if inp_idx in inp_adj[prev_inp]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[ref_idx] = inp_idx
            used.add(inp_idx)
            if backtrack(ref_idx + 1):
                return True
            del mapping[ref_idx]
            used.discard(inp_idx)
        return False

    if backtrack(0):
        return [mapping[i] for i in range(n)]
    return None


# ── Subgraph isomorphism ──────────────────────────────────────────────────────

def find_subgraph_isomorphisms(ref_graph: dict, ref_symbols: list,
                                inp_graph: dict, inp_symbols: list) -> list:
    """
    Find ALL induced-subgraph isomorphisms of ref_graph into inp_graph.

    An induced subgraph match preserves both edges AND non-edges within
    the matched atom set: if ref atoms a,b are NOT bonded, their images
    in the input structure must also NOT be bonded.

    IMPORTANT: does NOT exclude previously matched atoms. It finds
    every valid mapping independently. Overlap is resolved afterward,
    at the BOND level, by merge_pair_assignments() -- not here.

    Parameters
    ----------
    ref_graph    : dict[int, set[int]]  adjacency list of the ligand
    ref_symbols  : list[str]            element symbols, ligand index order
    inp_graph    : dict[int, set[int]]  adjacency list of the full structure
    inp_symbols  : list[str]            element symbols, input index order

    Returns
    -------
    matches : list of dict[int, int]    ref_index -> input_index
              One dict per valid embedding (includes symmetry-equivalent
              automorphisms of the ligand -- these are expected and are
              resolved by merge_pair_assignments, not filtered out here).
    """
    n_ref = len(ref_symbols)
    # Search most-constrained (highest-degree) ref atoms first -- pure
    # efficiency heuristic, does not affect correctness or completeness.
    ref_order = sorted(range(n_ref), key=lambda a: -len(ref_graph.get(a, ())))

    matches = []
    mapping = {}
    reverse = {}

    def backtrack(pos):
        if pos == len(ref_order):
            matches.append(dict(mapping))
            return

        ref_atom = ref_order[pos]

        for inp_atom in range(len(inp_symbols)):
            if inp_atom in reverse:
                continue
            if inp_symbols[inp_atom].capitalize() != ref_symbols[ref_atom].capitalize():
                continue
            if len(inp_graph.get(inp_atom, ())) < len(ref_graph.get(ref_atom, ())):
                continue

            # Check consistency with all already-assigned ref atoms
            ok = True
            for other_ref, other_inp in mapping.items():
                ref_bonded = other_ref in ref_graph.get(ref_atom, ())
                inp_bonded = other_inp in inp_graph.get(inp_atom, ())
                if ref_bonded != inp_bonded:          # induced-subgraph check
                    ok = False
                    break
            if not ok:
                continue

            mapping[ref_atom] = inp_atom
            reverse[inp_atom] = ref_atom
            backtrack(pos + 1)
            del mapping[ref_atom]
            del reverse[inp_atom]

    backtrack(0)
    return matches

def merge_pair_assignments(all_matches: list,
                            ref_pairs: list,
                            tol: float = 1e-9) -> tuple:
    """
    Merge per-bond data from ALL subgraph matches into a single,
    order-independent mapping of physical pairs -> interpolation data.

    No atoms are excluded and no subset of matches is ever chosen --
    every match contributes, and results are combined by physical
    pair (i, j), which is the natural unit for spring-constant
    interpolation. This is why the result cannot depend on the order
    atoms appear in the input file: set/dict union over an unordered
    collection of matches does not depend on iteration order, and
    every match that contains a given ref-pair must (for a physically
    sensible ligand definition) compute identical data for the
    physical pair it maps to.

    Parameters
    ----------
    all_matches : list of dict[int,int]
                  ref_idx -> inp_idx, one dict per valid embedding,
                  as returned by find_subgraph_isomorphisms(). Must
                  NOT have been pre-filtered/selected into a disjoint
                  subset -- pass every match found.
    ref_pairs   : list of tuples (ref_i, ref_j, *data)
                  The ligand's own per-pair data (e.g. k1, k2, d1, d2
                  or any other payload used later for interpolation).
                  ref_i, ref_j are indices into the ligand's own
                  numbering (0..n_ref-1).
    tol         : float
                  Numerical tolerance used when comparing whether two
                  matches agree on the data for the same physical pair.

    Returns
    -------
    pair_data : dict[(int,int), tuple]
                Physical pair (i,j) with i<j -> the agreed-upon data
                tuple (k1, k2, d1, d2, ...) for that bond.
    conflicts : list of (int, int, set)
                Physical pairs for which different matches produced
                DIFFERENT data (a genuine ambiguity in the ligand
                definition or a symmetry-equivalence bug). For each
                such pair, pair_data holds the elementwise AVERAGE of
                the conflicting values as a safe fallback, but this
                should be investigated -- it means the ligand pattern
                is not being applied consistently.
    """
    contributions = {}   # (i,j) -> list of data tuples
    refs_dict = {}

    for match in all_matches:
        for entry in ref_pairs:
            ref_i, ref_j = entry[0], entry[1]
            data = tuple(entry[2:])

            if ref_i not in match or ref_j not in match:
                continue   # shouldn't happen for a complete match, but be safe

            i, j = match[ref_i], match[ref_j]
            if i > j:
                i, j = j, i

            contributions.setdefault((i, j), []).append(data)
            refs_dict.setdefault((i, j), []).append((ref_i,ref_j))

    pair_data = {}
    conflicts = []

    for (i, j), values in contributions.items():
        arr = np.array(values, dtype=float)
        first = arr[0]

        if np.all(np.abs(arr - first) <= tol):
            # All matches agree (as expected) -- order-independent by
            # construction, since this check does not care which
            # match came first.
            pair_data[(i, j)] = tuple(first)
        else:
            # Genuine disagreement between matches -- report it rather
            # than silently picking whichever was found first.
            unique_vals = {tuple(row) for row in arr}
            conflicts.append((i, j, unique_vals,refs_dict[(i,j)]))
            pair_data[(i, j)] = tuple(arr.mean(axis=0))

    return pair_data, conflicts

# ── Spring model writer ───────────────────────────────────────────────────────

def write_spring_model(filepath, symbols, masses, coords, pair_k_dict, n_atoms):
    """
    Write a spring model file with ALL (i<j) pairs.
    Pairs not in pair_k_dict are written with k = 0.

    Parameters
    ----------
    filepath    : output file path
    symbols     : list of str — atom symbols for ALL n_atoms atoms
    masses      : list of float — atom masses for ALL n_atoms atoms
    coords      : np.ndarray (n_atoms, 3) — coordinates in Angstrom
    pair_k_dict : dict (i,j) -> k_au  — only interpolated pairs
    n_atoms     : int — total number of atoms in the input structure
    """
    n_interp  = len(pair_k_dict)
    n_total   = n_atoms * (n_atoms - 1) // 2
    n_zero    = n_total - n_interp

    with open(filepath, 'w') as f:
        f.write('# Spring model — interpolated from two reference spring models\n')
        f.write('# Pairs with no interpolation data have k = 0.\n')
        f.write('#\n')
        f.write('# Atom symbols (index order):\n')
        f.write('#    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)\n')
        for idx in range(n_atoms):
            sym  = symbols[idx] if idx < len(symbols) else '?'
            mass = masses[idx]  if idx < len(masses)  else 0.0
            if coords is not None and idx < len(coords):
                x, y, z = coords[idx]
            else:
                x, y, z = 0.0, 0.0, 0.0
            f.write(f'#  {idx:>7}  {sym:<6}  {mass:>12.6f}'
                    f'  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}\n')
        f.write('#\n')
        f.write('# Spring constants (Ha/Bohr²):\n')
        f.write('#   atom1   atom2       k (Ha/Bohr²)\n')

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                key   = (i, j)
                k_val = pair_k_dict.get(key, 0.0)
                f.write(f'  {i:>6}  {j:>6}  {k_val:>22.10f}\n')

    print(f'\n  Spring model written to  : {filepath}')
    print(f'  Total pairs written      : {n_total}')
    print(f'  Interpolated pairs       : {n_interp}')
    print(f'  Zero-filled pairs        : {n_zero}  (no interpolation data)')


# ── Interpolation constants file ──────────────────────────────────────────────

def build_and_save_interp(springs1_file, springs2_file, out_file):
    """
    Build and save per-pair interpolation constants from two spring model files.
    No input structure is required.
    """
    sym1, mas1, crd1, pairs1 = parse_springs_file(springs1_file)
    sym2, mas2, crd2, pairs2 = parse_springs_file(springs2_file)

    if sym1 != sym2:
        raise ValueError("Atom symbols differ between the two spring model files.")

    n = len(sym1)

    # Build pair dictionaries
    dict1 = {(i, j): k for (i, j, k) in pairs1}
    dict2 = {(i, j): k for (i, j, k) in pairs2}

    # Compute reference distances
    def pair_dist(coords, i, j):
        return float(np.linalg.norm(coords[i] - coords[j]))

    common_pairs = sorted(set(dict1.keys()) & set(dict2.keys()))

    print(f'\n  Building interpolation constants from:')
    print(f'    Spring model 1 : {springs1_file}  ({len(dict1)} pairs)')
    print(f'    Spring model 2 : {springs2_file}  ({len(dict2)} pairs)')
    print(f'    Common pairs   : {len(common_pairs)}')

    with open(out_file, 'w') as f:
        f.write('# Interpolation constants built from two spring model files\n')
        f.write(f'# Spring model 1 : {springs1_file}\n')
        f.write(f'# Spring model 2 : {springs2_file}\n')
        f.write('#\n')
        f.write('# Atom symbols, masses, and reference coordinates (Ang):\n')
        f.write('#    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)\n')
        for idx in range(n):
            f.write(f'#  {idx:>7}  {sym1[idx]:<6}  {mas1[idx]:>12.6f}'
                    f'  {crd1[idx,0]:>14.8f}  {crd1[idx,1]:>14.8f}  {crd1[idx,2]:>14.8f}\n')
        f.write('#\n')
        f.write('# Per-pair interpolation constants:\n')
        f.write('#   atom1   atom2         k1 (Ha/Bohr²)         k2 (Ha/Bohr²)'
                '        d1 (Ang)        d2 (Ang)\n')
        for (i, j) in common_pairs:
            k1 = dict1[(i, j)]
            k2 = dict2[(i, j)]
            d1 = pair_dist(crd1, i, j)
            d2 = pair_dist(crd2, i, j)
            f.write(f'  {i:>6}  {j:>6}  {k1:>22.10f}  {k2:>22.10f}'
                    f'  {d1:>16.8f}  {d2:>16.8f}\n')

    print(f'  Interpolation constants saved to : {out_file}')


def load_interp_constants(filepath):
    """
    Load pre-built interpolation constants.

    Returns
    -------
    symbols : list of str
    masses  : list of float
    coords  : np.ndarray (N, 3)
    pairs   : list of (i, j, k1, k2, d1, d2)
    """
    symbols = []
    masses  = []
    coords  = []
    pairs   = []

    in_atom_block  = False
    in_pair_block  = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if 'Atom symbols' in stripped or \
               ('Index' in stripped and 'Symbol' in stripped and 'Mass' in stripped):
                in_atom_block = True
                in_pair_block = False
                continue

            if 'Per-pair interpolation' in stripped or \
               ('atom1' in stripped.lower() and 'k1' in stripped.lower()):
                in_pair_block = True
                in_atom_block = False
                continue

            if stripped.startswith('#'):
                content = stripped.lstrip('#').strip()
                parts   = content.split()
                if in_atom_block and len(parts) >= 6:
                    try:
                        idx  = int(parts[0])
                        sym  = parts[1].capitalize()
                        mass = float(parts[2])
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        while len(symbols) <= idx:
                            symbols.append('')
                            masses.append(0.0)
                            coords.append([0.0, 0.0, 0.0])
                        symbols[idx] = sym
                        masses[idx]  = mass
                        coords[idx]  = [x, y, z]
                    except (ValueError, IndexError):
                        pass
                continue

            if in_pair_block:
                parts = stripped.split()
                if len(parts) >= 6:
                    try:
                        i  = int(parts[0])
                        j  = int(parts[1])
                        k1 = float(parts[2])
                        k2 = float(parts[3])
                        d1 = float(parts[4])
                        d2 = float(parts[5])
                        if d2 < d1:
                            d1,d2 = d2,d1
                            k1,k2 = k2,k1
                        pairs.append((i, j, k1, k2, d1, d2))
                    except ValueError:
                        continue

    return symbols, masses, np.array(coords), pairs


# ── Single-ligand interpolation ───────────────────────────────────────────────

def run_single_ligand(sym1, mas1, crd1, pairs1,
                      sym2, mas2, crd2, pairs2,
                      inp_symbols, inp_coords,
                      args, out_file):
    n = len(sym1)

    # Validate atom symbols
    if sym1 != inp_symbols:
        print("  Warning: atom symbols in spring model 1 differ from input XYZ.")
    if sym2 != inp_symbols:
        print("  Warning: atom symbols in spring model 2 differ from input XYZ.")

    # Bond network matching / reordering
    if not args.no_reorder:
        ref_bonds_for_match = build_bond_set(sym1, crd1, args.bond_factor,
                                              getattr(args, 'bond_cutoff', None))
        ref_adj = build_adjacency(n, ref_bonds_for_match)
        inp_bonds_for_match = build_bond_set(inp_symbols, inp_coords,
                                              args.bond_factor,
                                              getattr(args, 'bond_cutoff', None))
        inp_adj = build_adjacency(len(inp_symbols), inp_bonds_for_match)
        perm = find_matching_ordering(sym1, ref_adj, inp_symbols, inp_adj)
        if perm is not None:
            print(f'  Bond network match found — reordering input atoms.')
            inp_symbols = [inp_symbols[perm[i]] for i in range(n)]
            inp_coords  = inp_coords[perm]
        else:
            print('  Warning: no bond network match found — using original atom order.')

    # Build pair dictionaries
    dict1 = {(i, j): k for (i, j, k) in pairs1}
    dict2 = {(i, j): k for (i, j, k) in pairs2}

    # Bond set for input structure
    inp_bond_set = build_bond_set(inp_symbols, inp_coords,
                                   args.bond_factor,
                                   getattr(args, 'bond_cutoff', None))

    # Reference distances
    def pdist(crd, i, j):
        return float(np.linalg.norm(crd[i] - crd[j]))

    common = sorted(set(dict1.keys()) & set(dict2.keys()))

    pair_k_dict = {}
    n_log = n_lin = n_extrap = 0
    verbose_lines = []

    for (i, j) in common:
        k1  = dict1[(i, j)]
        k2  = dict2[(i, j)]
        d1  = pdist(crd1, i, j)
        d2  = pdist(crd2, i, j)
        d   = pdist(inp_coords, i, j)
        si  = inp_symbols[i]
        sj  = inp_symbols[j]
        bon = (i, j) in inp_bond_set or (j, i) in inp_bond_set

        k_interp, alpha, method, warn = interp_pair(
            k1, k2, d1, d2, d, si, sj, bon,
            not args.no_extrapolate, f'{i},{j}')

        if warn:
            print(warn)
            n_extrap += 1

        pair_k_dict[(i, j)] = k_interp

        if method.startswith('log'):
            n_log += 1
        else:
            n_lin += 1

        verbose_lines.append(
            f'  {i:>4}({si:<2}) {j:>4}({sj:<2})  '
            f'd1={d1:8.4f}  d={d:8.4f}  d2={d2:8.4f}  '
            f'alpha={alpha:7.4f}  k1={k1:14.8f}  k2={k2:14.8f}  '
            f'k={k_interp:14.8f}  [{method}]'
        )

    if args.verbose:
        print(f'\n  {"i":>4}      {"j":>4}    '
              f'{"d1(Å)":>10}  {"d(Å)":>10}  {"d2(Å)":>10}  '
              f'{"alpha":>8}  {"k1":>16}  {"k2":>16}  {"k":>16}  method')
        print('  ' + '-' * 120)
        for ln in verbose_lines:
            print(ln)

    print(f'\n  Pairs interpolated : {len(common)}')
    print(f'    log+             : {n_log}')
    print(f'    linear           : {n_lin}')
    if n_extrap:
        print(f'    extrapolated     : {n_extrap}')

    # Build full symbol/mass lists for output
    out_masses = [ATOMIC_MASSES.get(s, 0.0) for s in inp_symbols]

    if out_file is None:
        out_file = 'interpolated_springs.txt'

    write_spring_model(out_file, inp_symbols, out_masses,
                       inp_coords, pair_k_dict, len(inp_symbols))


# ── Use-interp single ligand ──────────────────────────────────────────────────

def run_use_interp_single(interp_symbols, interp_masses, interp_coords, interp_pairs,
                           inp_symbols, inp_coords, args, out_file):
    n = len(interp_symbols)

    # Bond network matching
    if not args.no_reorder:
        ref_bonds = build_bond_set(interp_symbols, interp_coords,
                                    args.bond_factor,
                                    getattr(args, 'bond_cutoff', None))
        ref_adj   = build_adjacency(n, ref_bonds)
        inp_bonds = build_bond_set(inp_symbols, inp_coords,
                                    args.bond_factor,
                                    getattr(args, 'bond_cutoff', None))
        inp_adj   = build_adjacency(len(inp_symbols), inp_bonds)
        perm = find_matching_ordering(interp_symbols, ref_adj, inp_symbols, inp_adj)
        if perm is not None:
            print(f'  Bond network match found — reordering input atoms.')
            inp_symbols = [inp_symbols[perm[i]] for i in range(n)]
            inp_coords  = inp_coords[perm]
        else:
            print('  Warning: no bond network match found — using original atom order.')

    inp_bond_set = build_bond_set(inp_symbols, inp_coords,
                                   args.bond_factor,
                                   getattr(args, 'bond_cutoff', None))

    def pdist(i, j):
        return float(np.linalg.norm(inp_coords[i] - inp_coords[j]))

    pair_k_dict = {}
    n_log = n_lin = n_extrap = 0

    for (i, j, k1, k2, d1, d2) in interp_pairs:
        if i >= len(inp_symbols) or j >= len(inp_symbols):
            continue
        si  = inp_symbols[i]
        sj  = inp_symbols[j]
        d   = pdist(i, j)
        bon = (i, j) in inp_bond_set or (j, i) in inp_bond_set

        k_interp, alpha, method, warn = interp_pair(
            k1, k2, d1, d2, d, si, sj, bon,
            not args.no_extrapolate, f'{i},{j}')

        if warn:
            print(warn)
            n_extrap += 1

        pair_k_dict[(i, j)] = k_interp

        if method.startswith('log'):
            n_log += 1
        else:
            n_lin += 1

    print(f'\n  Pairs interpolated : {len(pair_k_dict)}')
    print(f'    log+             : {n_log}')
    print(f'    linear           : {n_lin}')
    if n_extrap:
        print(f'    extrapolated     : {n_extrap}')

    out_masses = [ATOMIC_MASSES.get(s, 0.0) for s in inp_symbols]

    write_spring_model(out_file, inp_symbols, out_masses,
                       inp_coords, pair_k_dict, len(inp_symbols))


# ── Conflict classification (pipeline addition, not upstream) ────────────────

def _h_blind_pair_signature(ref_pair, ref_symbols):
    """A ref pair's identity with the *which* H forgotten, only the *that* kept."""
    return tuple(
        sorted(
            "H" if ref_symbols[r].capitalize() == "H" else str(r)
            for r in ref_pair
        )
    )


def partition_h_swap_conflicts(conflicts, ref_symbols):
    """Split merge conflicts into (genuine, hydrogen-degenerate).

    A ligand with interchangeable hydrogens has graph automorphisms, so the
    subgraph search returns one match per H permutation. Those matches map
    *different* reference H atoms onto the *same* physical pair, and the
    reference constants for chemically equivalent H differ in the last digits --
    so merge_pair_assignments sees a numerical disagreement and flags a conflict
    for what is really the same bond described several equivalent ways.

    A conflict is hydrogen-degenerate when every contributing reference pair is
    identical once you stop distinguishing *which* H it involved: e.g. ref pairs
    (0,4) and (0,6) with 4 and 6 both H both reduce to (0, H). Anything else --
    (0,1) vs (0,2) with 1=N, 2=C -- is a real ambiguity in the ligand definition
    and is returned as genuine.

    Both kinds keep the averaged value merge_pair_assignments already computed;
    this only decides what is worth a warning.
    """
    genuine, hydrogen_degenerate = [], []
    for conflict in conflicts:
        _i, _j, _values, refs = conflict
        signatures = {_h_blind_pair_signature(ref_pair, ref_symbols) for ref_pair in refs}
        target = hydrogen_degenerate if len(signatures) == 1 else genuine
        target.append(conflict)
    return genuine, hydrogen_degenerate


# ── Multi-ligand subgraph interpolation ───────────────────────────────────────

def run_multi_ligand_subgraph(ligand_interp_list,
                               inp_symbols, inp_coords,
                               args, out_file):
    """
    For each ligand in ligand_interp_list, find ALL subgraph isomorphisms
    in the input structure and interpolate the corresponding spring pairs.

    ligand_interp_list : list of (symbols, masses, coords, pairs)
      where pairs is either:
        - list of (i, j, k_au)              for --ligand mode
        - list of (i, j, k1, k2, d1, d2)   for --use-interp mode
    """
    n_inp       = len(inp_symbols)
    pair_k_dict = {}   # (i,j) in input indexing → k_interp
    conflicts   = []

    inp_bond_set = build_bond_set(inp_symbols, inp_coords,
                                   args.bond_factor,
                                   getattr(args, 'bond_cutoff', None))
    inp_adj      = build_adjacency(n_inp, inp_bond_set)

    for lig_idx, (lig_sym, lig_mas, lig_crd, lig_pairs) in enumerate(ligand_interp_list):
        n_lig = len(lig_sym)
        print(f'\n  ── Ligand {lig_idx + 1} : {n_lig} atoms, {len(lig_pairs)} pairs ──')

        ref_bond_set = build_bond_set(lig_sym, lig_crd,
                                       args.bond_factor,
                                       getattr(args, 'bond_cutoff', None))
        ref_adj      = build_adjacency(n_lig, ref_bond_set)

        matches = find_subgraph_isomorphisms(ref_adj, lig_sym, inp_adj, inp_symbols)
        print(f'  Subgraph matches found : {len(matches)}')

        if not matches:
            print(f'  Warning: no subgraph match found for ligand {lig_idx + 1}.')
            continue

        pair_data, ligand_conflicts = merge_pair_assignments(matches, lig_pairs)

        # Conflicts that are only the ligand's interchangeable hydrogens being
        # labelled differently by different matches are expected and physically
        # meaningless, so they are counted rather than listed -- otherwise a
        # ligand with a methyl group buries the log in warnings about nothing.
        ligand_genuine, hydrogen_degenerate = partition_h_swap_conflicts(
            ligand_conflicts, lig_sym
        )
        if hydrogen_degenerate:
            print(f'  Hydrogen-swap degeneracies (expected, averaged) : '
                  f'{len(hydrogen_degenerate)} pair(s)')
        for (i, j, values, _refs) in ligand_genuine:
            print(f"  WARNING: pair ({i},{j}) got inconsistent data "
                  f"from different matches: {values}")
        conflicts.extend(ligand_genuine)

        # Determine if pairs are (i,j,k) or (i,j,k1,k2,d1,d2)
        use_interp_mode = (len(lig_pairs[0]) == 6) if lig_pairs else False

        n_log = n_lin = n_extrap = 0
        for (inp_i, inp_j), (k1, k2, d1, d2) in pair_data.items():
                if use_interp_mode:
                    si    = inp_symbols[inp_i]
                    sj    = inp_symbols[inp_j]
                    d     = float(np.linalg.norm(inp_coords[inp_i] - inp_coords[inp_j]))
                    bon   = (min(inp_i, inp_j), max(inp_i, inp_j)) in inp_bond_set

                    k_interp, alpha, method, warn = interp_pair(
                        k1, k2, d1, d2, d, si, sj, bon,
                        not args.no_extrapolate, f'{inp_i},{inp_j}')
                    
                    pair_k_dict[(inp_i,inp_j)] = k_interp

                    if warn:
                        print(warn)
                        n_extrap += 1
                else:
                    # --ligand mode: pairs are (i, j, k_au) from spring model 1
                    # We do not have k2 here — just assign k1 directly
                    k_interp = k_au
                    method   = 'direct'

                if method.startswith('log'):
                    n_log += 1
                else:
                    n_lin += 1

        # Per-ligand tally. Upstream printed this inside the per-pair loop, which
        # emitted one line per atom pair (thousands of lines per cluster in a job
        # log); the counts it reported were partial. Same numbers, once per ligand.
        print(f'  log={n_log}  linear={n_lin}'
              + (f'  extrap={n_extrap}' if n_extrap else ''))

    if conflicts:
        # Upstream said "first assignment is kept"; merge_pair_assignments has
        # always averaged instead (see its final branch). Report what it does.
        print(f'\n  Warning: {len(conflicts)} pair(s) got genuinely inconsistent data '
              f'from different ligand matches.')
        print(f'  The elementwise average is used for each; the ligand definitions '
              f'are worth checking.')

    # ── Build full symbol and mass lists for ALL input atoms ─────────────────
    # This is the fix: always build from inp_symbols so length == n_inp
    out_symbols = list(inp_symbols)
    out_masses  = [ATOMIC_MASSES.get(s, 0.0) for s in inp_symbols]

    write_spring_model(out_file, out_symbols, out_masses,
                       inp_coords, pair_k_dict, n_inp)



# ── Library entry point ───────────────────────────────────────────────────────
# Replaces the upstream argparse CLI. Only the ``--use-interp`` path is exposed:
# that is the mode the pipeline uses (pre-built per-ligand constants + subgraph
# search against the cluster), and it is the mode upstream's own main() always
# took, since its "single use-interp" branch was guarded by an unreachable
# ``len(args.use_interp) == 0`` inside a block that requires len >= 1.


@dataclass
class InterpOptions:
    """Bonding/extrapolation knobs, mirroring the upstream CLI flags.

    The interpolation helpers read these off an ``args``-shaped object, so this
    stands in for the argparse namespace they were written against.
    """

    bond_factor: float = 1.2
    bond_cutoff: float | None = None
    no_extrapolate: bool = False
    verbose: bool = False


def interpolate_to_spring_model(
    xyz_path,
    interp_files,
    out_path,
    *,
    options: InterpOptions | None = None,
):
    """Interpolate per-ligand spring constants onto a cluster geometry.

    Each file in *interp_files* is a pre-built ``.interp`` constants file (one
    ligand type). Every occurrence of that ligand's bond network is located in
    the geometry at *xyz_path* by subgraph isomorphism, and its pair constants
    are interpolated to the observed bond lengths. The merged spring model for
    all atoms is written to *out_path*, which is returned as a
    :class:`~pathlib.Path`.

    Raises :class:`ValueError` when no ligand files are given.
    """
    options = options or InterpOptions()
    interp_files = [str(path) for path in interp_files]
    if not interp_files:
        raise ValueError("At least one .interp ligand constants file is required.")

    inp_symbols, inp_coords, _comment = parse_xyz(str(xyz_path))
    print(f'\n  Input structure : {xyz_path}  ({len(inp_symbols)} atoms)')
    print(f'  Ligand types    : {len(interp_files)}')

    ligand_interp_list = []
    for path in interp_files:
        sym, mas, crd, pairs = load_interp_constants(path)
        print(f'    {path} : {len(sym)} atoms, {len(pairs)} pairs')
        ligand_interp_list.append((sym, mas, crd, pairs))

    run_multi_ligand_subgraph(
        ligand_interp_list, inp_symbols, inp_coords, options, str(out_path)
    )
    return Path(out_path)
