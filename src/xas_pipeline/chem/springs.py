"""Ligand spring models: file formats, and interpolation onto a cluster.

A *spring model* assigns a force constant k_ij (Ha/Bohr^2) to atom pairs. A
ligand's ``.interp`` file holds two of them -- the ligand at two charge states,
with reference bond lengths d1 and d2 -- and :func:`interpolate` transfers them
onto a cluster:

1. Bond graphs are built for the ligand (from the ``.interp`` reference
   coordinates, i.e. model 1) and for the cluster (:func:`bond_set`).
2. Every induced-subgraph embedding of the ligand in the cluster is found
   (:func:`find_subgraph_isomorphisms`); symmetry-equivalent embeddings are all
   kept and merged per physical pair (:func:`merge_pair_assignments`).
3. Each pair's k is interpolated to the cluster's bond length d
   (:func:`interp_pair`), with alpha = (d - d1) / (d2 - d1) clamped to
   ``extrap_limits``:

   - log space, k = exp((1-alpha) ln k1 + alpha ln k2), when k falls as d grows,
     both k > 0, the pair is bonded in the cluster and neither atom is H;
   - linear, k = (1-alpha) k1 + alpha k2, otherwise.

Pairs no ligand covers get k = 0 in the written model.

Two conventions to know when reading results:

- :func:`read_interp` orders each pair so d1 < d2 (swapping k1/k2 with it), so
  alpha = 0 is the *shorter* reference bond, not necessarily model 1.
- When two ligand files match the same pair, the later file's value is used
  (a warning is logged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .periodic import COVALENT_RADII, DEFAULT_COVALENT_RADIUS, SPRING_MASSES_AMU
from .xyz import read_symbols_coords

log = logging.getLogger(__name__)

# Window alpha is clamped to: alpha = 0 is reference bond length d1, alpha = 1
# is d2, so +/-2 allows one reference span of extrapolation beyond either end.
EXTRAP_LIMITS_DEFAULT = (-2.0, 2.0)


# ── File formats ──────────────────────────────────────────────────────────────


@dataclass
class SpringModel:
    """A spring-model file: atoms (Angstrom) and ``(i, j, k)`` pair constants."""

    symbols: list[str]
    masses: list[float]
    coords: np.ndarray
    pairs: list[tuple[int, int, float]]


@dataclass
class InterpModel:
    """A ligand ``.interp`` file: model-1 atoms and ``(i, j, k1, k2, d1, d2)`` pairs."""

    symbols: list[str]
    masses: list[float]
    coords: np.ndarray
    pairs: list[tuple[int, int, float, float, float, float]]


def _parse_blocks(path, *, atom_markers, pair_markers, min_atom_fields, min_pair_fields):
    """Shared reader for the commented-header formats.

    The atom table is ``# idx symbol mass [x y z]`` lines after a header naming
    the atoms; the pair table is plain numeric lines after a header naming the
    pairs. Returns (symbols, masses, coords, pair rows as float lists).
    """
    symbols: list[str] = []
    masses: list[float] = []
    coords: list[list[float]] = []
    rows: list[list[float]] = []
    section = None
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker(stripped, lowered) for marker in atom_markers):
            section = "atoms"
            continue
        if any(marker(stripped, lowered) for marker in pair_markers):
            section = "pairs"
            continue
        if stripped.startswith("#"):
            parts = stripped.lstrip("#").split()
            if section == "atoms" and len(parts) >= min_atom_fields:
                try:
                    idx, symbol, mass = int(parts[0]), parts[1].capitalize(), float(parts[2])
                    xyz = [float(v) for v in parts[3:6]] if len(parts) >= 6 else [0.0, 0.0, 0.0]
                except ValueError:
                    continue
                while len(symbols) <= idx:
                    symbols.append("")
                    masses.append(0.0)
                    coords.append([0.0, 0.0, 0.0])
                symbols[idx], masses[idx], coords[idx] = symbol, mass, xyz
            continue
        if section == "pairs":
            parts = stripped.split()
            if len(parts) >= min_pair_fields:
                try:
                    rows.append([int(parts[0]), int(parts[1]), *map(float, parts[2:min_pair_fields])])
                except ValueError:
                    continue
    return symbols, masses, np.array(coords), rows


_ATOM_HEADER = (
    lambda s, _l: "Atom symbols" in s,
    lambda s, _l: "Index" in s and "Symbol" in s and "Mass" in s,
)


def read_springs(path) -> SpringModel:
    """Read a spring-model file (fitted or interpolated format)."""
    symbols, masses, coords, rows = _parse_blocks(
        path,
        atom_markers=_ATOM_HEADER,
        pair_markers=(
            lambda s, _l: "Spring constants" in s,
            lambda _s, line: "atom1" in line and "atom2" in line,
        ),
        min_atom_fields=3,
        min_pair_fields=3,
    )
    if not symbols:
        raise ValueError(f"No atom table in spring model {path}")
    if not coords.any():
        raise ValueError(f"No coordinates in spring model {path}")
    if not rows:
        raise ValueError(f"No spring constants in spring model {path}")
    return SpringModel(symbols, masses, coords, [(int(i), int(j), k) for i, j, k in rows])


def read_interp(path) -> InterpModel:
    """Read a ligand ``.interp`` file, ordering each pair so that d1 < d2."""
    symbols, masses, coords, rows = _parse_blocks(
        path,
        atom_markers=_ATOM_HEADER,
        pair_markers=(
            lambda s, _l: "Per-pair interpolation" in s,
            lambda _s, line: "atom1" in line and "k1" in line,
        ),
        min_atom_fields=6,
        min_pair_fields=6,
    )
    pairs = []
    for i, j, k1, k2, d1, d2 in rows:
        if d2 < d1:
            d1, d2 = d2, d1
            k1, k2 = k2, k1
        pairs.append((int(i), int(j), k1, k2, d1, d2))
    return InterpModel(symbols, masses, coords, pairs)


def write_spring_model(path, symbols, masses, coords, pair_k: dict) -> None:
    """Write an interpolated spring model: every i < j pair, k = 0 where unset."""
    n_atoms = len(symbols)
    lines = [
        "# Spring model — interpolated from two reference spring models\n",
        "# Pairs with no interpolation data have k = 0.\n",
        "#\n",
        "# Atom symbols (index order):\n",
        "#    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)\n",
    ]
    for idx, (sym, mass, (x, y, z)) in enumerate(zip(symbols, masses, coords, strict=True)):
        lines.append(f"#  {idx:>7}  {sym:<6}  {mass:>12.6f}  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}\n")
    lines += ["#\n", "# Spring constants (Ha/Bohr²):\n", "#   atom1   atom2       k (Ha/Bohr²)\n"]
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            lines.append(f"  {i:>6}  {j:>6}  {pair_k.get((i, j), 0.0):>22.10f}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")
    n_total = n_atoms * (n_atoms - 1) // 2
    log.info("Spring model written to %s: %d pairs, %d interpolated", path, n_total, len(pair_k))


def build_interp(springs1_path, springs2_path, out_path, *, pair_sources: dict | None = None) -> None:
    """Combine two spring models of one ligand into an ``.interp`` file.

    Model 1's coordinates become the reference geometry (and so the ligand bond
    graph at inference). Only pairs present in both models are kept.

    *pair_sources* maps a pair ``(i, j)`` (i < j) to another spring-model file of
    the same atoms that supplies that pair's model 2 -- its k2, and d2 from its
    own geometry -- instead of *springs2_path*. Each pair interpolates
    independently, so one file can carry, say, a different second charge state
    per metal-ligand bond.
    """
    pair_sources = {tuple(sorted(pair)): Path(src) for pair, src in (pair_sources or {}).items()}
    m1, m2 = read_springs(springs1_path), read_springs(springs2_path)
    alternates = {src: read_springs(src) for src in sorted(set(pair_sources.values()))}
    for model in (m2, *alternates.values()):
        if model.symbols != m1.symbols:
            raise ValueError("Atom symbols differ between the spring model files.")
    k1s = {(i, j): k for i, j, k in m1.pairs}
    k2s = {(i, j): k for i, j, k in m2.pairs}
    alt_k = {src: {(i, j): k for i, j, k in model.pairs} for src, model in alternates.items()}
    missing = [pair for pair, src in pair_sources.items() if pair not in alt_k[src] or pair not in k1s]
    if missing:
        raise ValueError(f"pair(s) {missing} have no constant in their designated spring model")
    common = sorted(set(k1s) & set(k2s))
    crd1 = m1.coords
    lines = [
        "# Interpolation constants built from two spring model files\n",
        f"# Spring model 1 : {springs1_path}\n",
        f"# Spring model 2 : {springs2_path}\n",
        *(f"# Spring model 2 for pair {i}-{j} : {src}\n" for (i, j), src in sorted(pair_sources.items())),
        "#\n",
        "# Atom symbols, masses, and reference coordinates (Ang):\n",
        "#    Index  Symbol    Mass (amu)       X (Ang)        Y (Ang)        Z (Ang)\n",
    ]
    for idx, (sym, mass) in enumerate(zip(m1.symbols, m1.masses, strict=True)):
        lines.append(
            f"#  {idx:>7}  {sym:<6}  {mass:>12.6f}"
            f"  {crd1[idx, 0]:>14.8f}  {crd1[idx, 1]:>14.8f}  {crd1[idx, 2]:>14.8f}\n"
        )
    lines += [
        "#\n",
        "# Per-pair interpolation constants:\n",
        "#   atom1   atom2         k1 (Ha/Bohr²)         k2 (Ha/Bohr²)        d1 (Ang)        d2 (Ang)\n",
    ]
    for i, j in sorted(set(common) | set(pair_sources)):
        src = pair_sources.get((i, j))
        crd2 = m2.coords if src is None else alternates[src].coords
        k2 = k2s[(i, j)] if src is None else alt_k[src][(i, j)]
        d1 = float(np.linalg.norm(crd1[i] - crd1[j]))
        d2 = float(np.linalg.norm(crd2[i] - crd2[j]))
        lines.append(
            f"  {i:>6}  {j:>6}  {k1s[(i, j)]:>22.10f}  {k2:>22.10f}  {d1:>16.8f}  {d2:>16.8f}\n"
        )
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    log.info(
        "Interpolation constants: %s + %s -> %s (%d common pairs)",
        springs1_path, springs2_path, out_path, len(common),
    )


# ── Bonds ─────────────────────────────────────────────────────────────────────


def is_bonded(sym_i, sym_j, dist, bond_factor=1.2, bond_cutoff=None) -> bool:
    """d <= cutoff when given, else d <= factor * (r_i + r_j) with covalent radii."""
    if bond_cutoff is not None:
        return dist <= bond_cutoff
    ri = COVALENT_RADII.get(sym_i, DEFAULT_COVALENT_RADIUS)
    rj = COVALENT_RADII.get(sym_j, DEFAULT_COVALENT_RADIUS)
    return dist <= bond_factor * (ri + rj)


def bond_set(symbols, coords, bond_factor=1.2, bond_cutoff=None) -> set[tuple[int, int]]:
    """All bonded pairs ``(i, j)``, i < j."""
    n = len(symbols)
    return {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if is_bonded(symbols[i], symbols[j], np.linalg.norm(coords[i] - coords[j]), bond_factor, bond_cutoff)
    }


def adjacency(n, bonds) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)
    return adj


# ── Interpolation ─────────────────────────────────────────────────────────────


def interp_pair(k1, k2, d1, d2, d_inp, sym_i, sym_j, bonded, allow_extrap, label,
                extrap_limits=EXTRAP_LIMITS_DEFAULT):
    """Interpolate one pair's constant to bond length *d_inp*.

    Returns ``(k, alpha, method, warning or None)``; *method* names the branch
    taken (``log+`` or ``linear(...)``).
    """
    if abs(d2 - d1) < 1e-12:
        return 0.5 * (k1 + k2), 0.5, "linear(d1=d2)", None

    alpha = (d_inp - d1) / (d2 - d1)
    # One pair extrapolated far outside [d1, d2] -- exponentially, in log space --
    # is enough to dominate the Hessian, hence the clamp.
    alpha = max(extrap_limits[0], min(extrap_limits[1], alpha))

    warn = None
    if not allow_extrap and (alpha < 0.0 or alpha > 1.0):
        warn = f"Extrapolation blocked for pair ({label}): alpha={alpha:.4f}"
        alpha = max(0.0, min(1.0, alpha))

    has_h = sym_i == "H" or sym_j == "H"
    dk_dd_neg = (k2 - k1) * (d2 - d1) < 0  # k decreases as d increases
    both_pos = k1 > 0 and k2 > 0

    if both_pos and dk_dd_neg and bonded and not has_h:
        log_k = (1.0 - alpha) * np.log(k1) + alpha * np.log(k2)
        return np.exp(log_k), alpha, "log+", warn

    if has_h:
        method = "linear(H)"
    elif not bonded:
        method = "linear(non-bonded)"
    elif not dk_dd_neg:
        method = "linear(dk/dd>=0)"
    elif k1 < 0 and k2 < 0:
        method = "linear(both-)"
    elif k1 == 0 or k2 == 0:
        method = "linear(zero)"
    else:
        method = "linear(mixed)"
    return (1.0 - alpha) * k1 + alpha * k2, alpha, method, warn


def find_subgraph_isomorphisms(ref_graph, ref_symbols, inp_graph, inp_symbols) -> list[dict[int, int]]:
    """Every induced-subgraph embedding of the ligand graph in the cluster graph.

    Induced: ligand atoms that are not bonded must map to cluster atoms that are
    not bonded either. Previously matched atoms are not excluded, and
    automorphisms (equivalent H, ring flips) each give their own match;
    :func:`merge_pair_assignments` combines them per physical pair.
    Returns one ``{ligand index: cluster index}`` dict per embedding.
    """
    # Most-constrained (highest-degree) ligand atoms first: speed only.
    ref_order = sorted(range(len(ref_symbols)), key=lambda a: -len(ref_graph.get(a, ())))
    matches: list[dict[int, int]] = []
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(pos):
        if pos == len(ref_order):
            matches.append(dict(mapping))
            return
        ref_atom = ref_order[pos]
        for inp_atom in range(len(inp_symbols)):
            if inp_atom in used:
                continue
            if inp_symbols[inp_atom].capitalize() != ref_symbols[ref_atom].capitalize():
                continue
            if len(inp_graph.get(inp_atom, ())) < len(ref_graph.get(ref_atom, ())):
                continue
            if any(
                (other_ref in ref_graph.get(ref_atom, ())) != (other_inp in inp_graph.get(inp_atom, ()))
                for other_ref, other_inp in mapping.items()
            ):
                continue
            mapping[ref_atom] = inp_atom
            used.add(inp_atom)
            backtrack(pos + 1)
            del mapping[ref_atom]
            used.discard(inp_atom)

    backtrack(0)
    return matches


def merge_pair_assignments(all_matches, ref_pairs, tol=1e-9):
    """Combine every match's per-pair data into one value per physical pair.

    *ref_pairs* are ``(ref_i, ref_j, *data)`` in ligand numbering. Returns
    ``(pair_data, conflicts)``: ``pair_data[(i, j)]`` (i < j, cluster numbering) is
    the data tuple all contributing matches agree on to *tol*, or their
    elementwise mean when they do not; each such disagreement is listed in
    *conflicts* as ``(i, j, distinct values, contributing ligand pairs)``.
    """
    contributions: dict[tuple[int, int], list] = {}
    refs: dict[tuple[int, int], list] = {}
    for match in all_matches:
        for entry in ref_pairs:
            ref_i, ref_j = entry[0], entry[1]
            if ref_i not in match or ref_j not in match:
                continue
            i, j = sorted((match[ref_i], match[ref_j]))
            contributions.setdefault((i, j), []).append(tuple(entry[2:]))
            refs.setdefault((i, j), []).append((ref_i, ref_j))

    pair_data, conflicts = {}, []
    for key, values in contributions.items():
        arr = np.array(values, dtype=float)
        if np.all(np.abs(arr - arr[0]) <= tol):
            pair_data[key] = tuple(arr[0])
        else:
            conflicts.append((*key, {tuple(row) for row in arr}, refs[key]))
            pair_data[key] = tuple(arr.mean(axis=0))
    return pair_data, conflicts


def partition_h_swap_conflicts(conflicts, ref_symbols):
    """Split merge conflicts into ``(genuine, hydrogen_degenerate)``.

    Equivalent hydrogens give one match per permutation, mapping *different*
    reference H onto the same physical pair; their constants differ in the last
    digits, so the merge flags them. A conflict is hydrogen-degenerate when its
    contributing reference pairs are identical once *which* H is forgotten.
    Both kinds keep the averaged value; this only decides what to warn about.
    """

    def signature(ref_pair):
        return tuple(sorted("H" if ref_symbols[r].capitalize() == "H" else str(r) for r in ref_pair))

    genuine, degenerate = [], []
    for conflict in conflicts:
        same = len({signature(ref_pair) for ref_pair in conflict[3]}) == 1
        (degenerate if same else genuine).append(conflict)
    return genuine, degenerate


@dataclass(frozen=True)
class InterpOptions:
    """Bond criterion and extrapolation window for :func:`interpolate`."""

    bond_factor: float = 1.2
    bond_cutoff: float | None = None
    extrapolate: bool = True
    extrap_limits: tuple[float, float] = EXTRAP_LIMITS_DEFAULT


def interpolate(symbols, coords, ligands: list[InterpModel], options: InterpOptions | None = None):
    """Interpolated constants ``{(i, j): k}`` for a cluster, from ligand models."""
    options = options or InterpOptions()
    cluster_bonds = bond_set(symbols, coords, options.bond_factor, options.bond_cutoff)
    cluster_graph = adjacency(len(symbols), cluster_bonds)
    pair_k: dict[tuple[int, int], float] = {}
    n_genuine = 0

    for number, ligand in enumerate(ligands, start=1):
        ligand_graph = adjacency(
            len(ligand.symbols),
            bond_set(ligand.symbols, ligand.coords, options.bond_factor, options.bond_cutoff),
        )
        matches = find_subgraph_isomorphisms(ligand_graph, ligand.symbols, cluster_graph, symbols)
        log.info("Ligand %d (%d atoms, %d pairs): %d subgraph match(es)",
                 number, len(ligand.symbols), len(ligand.pairs), len(matches))
        if not matches:
            continue

        pair_data, conflicts = merge_pair_assignments(matches, ligand.pairs)
        genuine, degenerate = partition_h_swap_conflicts(conflicts, ligand.symbols)
        if degenerate:
            log.info("  hydrogen-swap degeneracies (expected, averaged): %d pair(s)", len(degenerate))
        for i, j, values, _refs in genuine:
            log.warning("  pair (%d,%d) got inconsistent data from different matches: %s", i, j, values)
        n_genuine += len(genuine)

        n_log = n_lin = n_blocked = 0
        for (i, j), (k1, k2, d1, d2) in pair_data.items():
            d = float(np.linalg.norm(coords[i] - coords[j]))
            k, _alpha, method, warn = interp_pair(
                k1, k2, d1, d2, d, symbols[i], symbols[j], (i, j) in cluster_bonds,
                options.extrapolate, f"{i},{j}", options.extrap_limits,
            )
            if (i, j) in pair_k and pair_k[(i, j)] != k:
                log.warning("  pair (%d,%d) also matched an earlier ligand; using ligand %d's value",
                            i, j, number)
            pair_k[(i, j)] = k
            if warn:
                log.info("  %s", warn)
                n_blocked += 1
            if method.startswith("log"):
                n_log += 1
            else:
                n_lin += 1
        log.info("  log=%d  linear=%d%s", n_log, n_lin, f"  extrap-blocked={n_blocked}" if n_blocked else "")

    if n_genuine:
        log.warning("%d pair(s) got inconsistent data from different matches; the elementwise "
                    "average is used for each, and the ligand definitions are worth checking.",
                    n_genuine)
    return pair_k


def interpolate_to_spring_model(xyz_path, interp_files, out_path, *, options: InterpOptions | None = None) -> Path:
    """Interpolate ligand ``.interp`` files onto the geometry at *xyz_path*.

    Writes the full spring model (every pair; uncovered pairs k = 0, masses from
    :data:`~xas_pipeline.chem.periodic.SPRING_MASSES_AMU`) to *out_path*.
    """
    interp_files = list(interp_files)
    if not interp_files:
        raise ValueError("At least one .interp ligand constants file is required.")
    symbols, coords, _comment = read_symbols_coords(xyz_path)
    unknown = sorted(set(symbols) - set(SPRING_MASSES_AMU))
    if unknown:
        raise ValueError(f"No spring-model mass for element(s) {unknown} in {xyz_path}")
    log.info("Input structure %s (%d atoms), %d ligand file(s)", xyz_path, len(symbols), len(interp_files))
    ligands = [read_interp(path) for path in interp_files]
    pair_k = interpolate(symbols, coords, ligands, options)
    write_spring_model(out_path, symbols, [SPRING_MASSES_AMU[s] for s in symbols], coords, pair_k)
    return Path(out_path)
