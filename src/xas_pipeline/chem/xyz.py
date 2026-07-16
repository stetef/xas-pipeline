"""XYZ geometry parsing (ORCA and Corvus conventions).

Pure readers extracted from prepare-orca.py (``count_atoms_xyz``,
``extract_charge_multiplicity``) and prepare-corvus.py (``read_xyz``,
``read_last_xyz_frame``). Each reads a passed-in path and returns data with no
other side effects. The file-*writing* xyz cleaners stay in the stage code.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import periodic
from .periodic import ANGSTROM_PER_BOHR


def count_atoms_xyz(xyz_path):
    """Return the atom count for an XYZ file (header count, else counted rows)."""
    try:
        lines = Path(xyz_path).read_text().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    first = lines[0].strip().split()
    if first and first[0].lstrip("+-").isdigit():
        return int(first[0])
    # Fallback: count lines that look like "<element> x y z".
    natoms = sum(1 for raw in lines[2:] if len(raw.split()) >= 4)
    return natoms or None


def extract_charge_multiplicity(xyz_file):
    """Extract charge and multiplicity from XYZ header line 2."""
    try:
        lines = Path(xyz_file).read_text().splitlines()
    except FileNotFoundError:
        return None, None

    if len(lines) < 2:
        return None, None

    header = lines[1]
    charge_match = re.search(r"\b(?:CHARGE_ROUNDED|ROUNDED_CHARGE|CHARGE)=([-+]?\d+)\b", header)
    mult_match = re.search(r"\b(?:MULTIPLICITY|MULT)=(\d+)\b", header)

    if not charge_match or not mult_match:
        return None, None

    return int(charge_match.group(1)), int(mult_match.group(1))


def read_xyz(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an XYZ file into (atomic_numbers, masses_amu, coords_bohr)."""
    with open(filename, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    lines = [ln for ln in raw_lines if ln]
    if not lines:
        raise ValueError("Empty XYZ file")

    atom_lines = None
    try:
        n = int(lines[0].split()[0])
        if len(lines) < 2 + n:
            raise ValueError("XYZ file too short for declared atom count")
        atom_lines = lines[2:2 + n]
    except ValueError:
        atom_lines = [ln for ln in lines if len(ln.split()) >= 4]

    atomic_numbers = []
    masses_amu = []
    coords_bohr = []

    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        z = periodic.atomic_number_from_token(parts[0])
        x_a, y_a, z_a = map(float, parts[1:4])
        atomic_numbers.append(z)
        masses_amu.append(periodic.atomic_mass_amu(z))
        coords_bohr.append(
            [x_a / ANGSTROM_PER_BOHR, y_a / ANGSTROM_PER_BOHR, z_a / ANGSTROM_PER_BOHR]
        )

    if not atomic_numbers:
        raise ValueError("No atom records found in XYZ")

    return (
        np.array(atomic_numbers, dtype=int),
        np.array(masses_amu, dtype=float),
        np.array(coords_bohr, dtype=float),
    )


def read_last_xyz_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the final XYZ frame as (atomic_numbers, coords_angstrom)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    last_atomic_numbers = None
    last_coords = None

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            natoms = int(line.split()[0])
        except (IndexError, ValueError):
            i += 1
            continue

        if natoms <= 0:
            raise ValueError(f"Invalid atom count in XYZ frame ({natoms}) for {path}")
        if i + 2 + natoms > len(lines):
            raise ValueError(f"Truncated XYZ frame in {path} near line {i + 1}")

        atom_lines = lines[i + 2 : i + 2 + natoms]
        atomic_numbers = []
        coords = []
        for raw in atom_lines:
            main_part = raw.split("#", 1)[0]
            tokens = main_part.split()
            if len(tokens) < 4:
                raise ValueError(f"Invalid atom line in {path}: {raw!r}")
            z = periodic.atomic_number_from_token(tokens[0])
            x_a, y_a, z_a = map(float, tokens[1:4])
            atomic_numbers.append(z)
            coords.append([x_a, y_a, z_a])

        last_atomic_numbers = np.array(atomic_numbers, dtype=int)
        last_coords = np.array(coords, dtype=float)
        i = i + 2 + natoms

    if last_atomic_numbers is None or last_coords is None:
        raise ValueError(f"No XYZ frames parsed from {path}")

    return last_atomic_numbers, last_coords
