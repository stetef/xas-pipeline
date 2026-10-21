"""ORCA ``.hess`` files: read the Hessian and geometry, write a minimal one.

The reader serves both routes: ORCA's own analytic-frequency Hessians and the
spring-model Hessians :mod:`~xas_pipeline.chem.spring_hessian` writes, so the
CORVUS stage cannot tell the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .units import ANG_TO_BOHR


@dataclass
class HessFile:
    """The parts of an ORCA ``.hess`` file the pipeline uses."""

    matrix: np.ndarray  # (3N, 3N), Hartree/Bohr^2
    symbols: list[str]  # from $atoms, capitalized; empty if the block is absent
    masses: np.ndarray  # amu
    coords_bohr: np.ndarray  # (N, 3)
    frequencies: np.ndarray | None  # cm^-1, from $vibrational_frequencies

    @property
    def n_atoms(self) -> int:
        return self.matrix.shape[0] // 3


def _is_col_header(tokens: list[str], n: int) -> bool:
    """A Hessian block's column-index line: ascending integers in [0, n)."""
    if not tokens or any("." in t or "e" in t.lower() for t in tokens):
        return False
    try:
        indices = [int(t) for t in tokens]
    except ValueError:
        return False
    return all(0 <= c < n for c in indices) and indices == sorted(indices)


def _parse_matrix(lines: list[str], start: int, n: int) -> tuple[np.ndarray, int]:
    """Parse the blocked ``$hessian`` data from *start*; returns (H, next line index)."""
    H = np.zeros((n, n))
    cols = None
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("$") and i != start:
            break
        tokens = stripped.split()
        if _is_col_header(tokens, n):
            cols = [int(t) for t in tokens]
        elif cols is not None:
            try:
                row = int(tokens[0])
                values = [float(v) for v in tokens[1:]]
            except ValueError:
                values = None
            if values is not None and 0 <= row < n:
                for col, value in zip(cols, values):
                    H[row, col] = value
        i += 1
    return H, i


def read_hess(path) -> HessFile:
    """Parse an ORCA ``.hess`` file. Raises :class:`ValueError` without ``$hessian``."""
    lines = Path(path).read_text().splitlines()
    matrix = None
    atoms: list[tuple[str, float, float, float, float]] = []
    frequencies = None
    i = 0
    while i < len(lines):
        key = lines[i].strip().lower()
        if key == "$atoms":
            n = int(lines[i + 1])
            for raw in lines[i + 2 : i + 2 + n]:
                parts = raw.split()
                atoms.append((parts[0].capitalize(), *map(float, parts[1:5])))
            i += 2 + n
            continue
        if key == "$hessian":
            n = int(lines[i + 1])
            matrix, i = _parse_matrix(lines, i + 2, n)
            continue
        if key == "$vibrational_frequencies":
            n = int(lines[i + 1])
            frequencies = np.array([float(raw.split()[1]) for raw in lines[i + 2 : i + 2 + n]])
            i += 2 + n
            continue
        i += 1
    if matrix is None:
        raise ValueError(f"No $hessian block in {path}")
    return HessFile(
        matrix=matrix,
        symbols=[a[0] for a in atoms],
        masses=np.array([a[1] for a in atoms]),
        coords_bohr=np.array([a[2:] for a in atoms]).reshape(-1, 3),
        frequencies=frequencies,
    )


def read_orca_hessian(path) -> tuple[np.ndarray, int]:
    """The Cartesian Hessian (Hartree/Bohr^2) and atom count of a ``.hess`` file."""
    hess = read_hess(path)
    return hess.matrix, hess.n_atoms


def format_hessian_block(H: np.ndarray, block_size: int = 5, decimals: int = 8) -> list[str]:
    """ORCA-style ``$hessian`` body lines (size line, then column blocks)."""
    n = H.shape[0]
    val_width = decimals + 8  # sign + digit + '.' + decimals + 'e±XX'
    lines = [f"{n}\n"]
    for col_start in range(0, n, block_size):
        cols = range(col_start, min(col_start + block_size, n))
        lines.append("  " + "   ".join(f"{c:>{val_width - 3}d}" for c in cols) + "\n")
        for row in range(n):
            lines.append(
                f"{row:>6d}" + "".join(f" {H[row, c]:>{val_width}.{decimals}e}" for c in cols) + "\n"
            )
    return lines


def write_hess(path, H: np.ndarray, symbols, masses, coords_ang: np.ndarray) -> Path:
    """Write a minimal ORCA ``.hess``: ``$act_atoms``, ``$hessian`` and ``$atoms``."""
    n_atoms = len(symbols)
    coords_bohr = coords_ang * ANG_TO_BOHR
    atoms = "".join(
        f"  {sym:<4s}  {mass:>12.6f}  {x:>18.10f}  {y:>18.10f}  {z:>18.10f}\n"
        for sym, mass, (x, y, z) in zip(symbols, masses, coords_bohr)
    )
    text = (
        "\n$orca_hessian_file\n\n"
        f"$act_atoms\n{n_atoms}\n\n"
        f"$act_coord\n{3 * n_atoms}\n\n"
        "$hessian\n" + "".join(format_hessian_block(H)) + f"\n$atoms\n{n_atoms}\n{atoms}"
    )
    path = Path(path)
    path.write_text(text)
    return path
