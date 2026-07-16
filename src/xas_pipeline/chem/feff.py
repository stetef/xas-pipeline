"""FEFF/Corvus output tables and the chi(k)->chi(R) FFT.

Pure readers/transforms extracted from script-process-feff-output.py. ``xftf_larch``
imports larch lazily (heavy optional dep) but is otherwise a deterministic numeric
transform. The plotting/copy orchestration stays in the stage code.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_feff_table(path: Path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns in {path}, got {data.shape[1]}")
    omega = data[:, 0]
    energy = data[:, 1]
    k = data[:, 2]
    mu = data[:, 3]
    mu0 = data[:, 4]
    chi = data[:, 5]
    return omega, energy, k, mu, mu0, chi


def load_xmu_columns(path: Path):
    # FEFF xmu.dat uses the same 6-column numeric layout as other FEFF tables.
    return load_feff_table(path)


def xmu_reports_zero_paths(path: Path) -> bool:
    pattern = re.compile(r"^#\s*0\s*/\s*0\s+paths\s+used\b", re.IGNORECASE)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if pattern.match(raw_line.strip()):
                return True
    return False


def load_chi_dat(path: Path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {data.shape[1]}")
    k = data[:, 0]
    chi = data[:, 1]
    return k, chi


def xftf_larch(k, chi, kmin, kmax, dk, kweight, kstep, rmax_out, window):
    from larch import Group
    from larch.xafs import xftf

    grp = Group()
    grp.k = k
    grp.chi = chi
    xftf(
        grp.k,
        grp.chi,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        kweight=kweight,
        kstep=kstep,
        rmax_out=rmax_out,
        window=window,
        group=grp,
    )
    return grp.r, grp.chir


def parse_cfavg_mode_from_input(input_path: Path) -> str | None:
    pattern = re.compile(r"cfavg_target\s*\{\s*(xas|xanes|exafs)\s*\}", re.IGNORECASE)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).lower()
