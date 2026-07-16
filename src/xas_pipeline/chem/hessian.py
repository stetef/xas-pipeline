"""ORCA Hessian (``.hess``) parsing.

Pure reader extracted from prepare-corvus.py. Returns the mass-weighted Cartesian
Hessian matrix and atom count; the ``.dym`` *writer* that consumes it is an I/O
shell that stays in the stage code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_orca_hessian(filename: Path) -> tuple[np.ndarray, int]:
    """Parse the ``$HESSIAN`` block of an ORCA ``.hess`` file -> (matrix, natoms)."""
    with open(filename, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "$HESSIAN" in line.upper():
            start = i + 1
            break

    if start is None:
        raise ValueError("HESSIAN section not found.")

    size = int(lines[start].strip())
    natoms = int(size / 3)
    hess_start = start + 1

    hessian = np.zeros((size, size))
    row = 0
    i = hess_start

    while row < size:
        header = lines[i].split()
        ncols = len(header)
        cols = [int(x) for x in header]
        i += 1
        for r in range(size):
            parts = lines[i + r].split()
            values = list(map(float, parts[1:1 + ncols]))
            for c, val in zip(cols, values):
                hessian[r, c] = val
        if c >= size - 1:
            break
        i += size
        row += 1

    return hessian, natoms
