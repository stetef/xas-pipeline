"""FEFF/Corvus output tables, the chi(k)->chi(R) FFT, and chi-column health.

Pure readers/transforms extracted from script-process-feff-output.py. ``xftf_larch``
imports larch lazily (heavy optional dep) but is otherwise a deterministic numeric
transform. The plotting/copy orchestration stays in the stage code.

:func:`scan_chi_column` is the numeric health check on a FEFF 6-column table's
chi column, used by the postprocess gate (see :mod:`xas_pipeline.corvus_diagnosis`)
to catch a XANES leg that produced no fine structure at all. Ported from
calculations/scan-xanes-zero-chi.py, which surveys finished batches with the same
rules; keep the two in agreement.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
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


# ── chi-column health ────────────────────────────────────────────────────────
#
# FEFF's xmu.dat is a 6-column table -- omega, e, k, mu, mu0, chi -- where
# chi = mu - mu0 is the fine structure. A leg that produced no scattering
# contribution writes mu == mu0 exactly on every row, so chi is identically
# 0.0000000000E+00 and the spectrum carries no structural information.
#
# For the XANES leg that failure is invisible to a header check: the "0/0 paths
# used" line only means something for EXAFS (XANES is FMS, not a path expansion,
# so it legitimately reports zero paths). The only way to see it is to look at
# the numbers.

CHI_OK = "ok"
CHI_ALL_ZERO = "all-zero"
CHI_PARTIAL_ZERO = "partial-zero"
CHI_NON_FINITE = "non-finite"
CHI_UNREADABLE = "unreadable"


@dataclass
class ChiScan:
    """The verdict on one table's chi column, worst finding first.

    ``status`` is one of the ``CHI_*`` constants; ``detail`` is a human-readable
    sentence naming the evidence. Precedence, when a file trips more than one
    rule: unreadable (no parseable rows) > all-zero > partial-zero > non-finite >
    malformed-but-usable. Same order as the standalone scan script's buckets.
    """

    status: str
    detail: str
    n_rows: int = 0
    n_zero: int = 0
    n_nonfinite: int = 0
    n_malformed: int = 0
    # Photon-energy (column 1) span covered by the zero rows, for partial hits.
    zero_energy_range: tuple[float, float] | None = None

    @property
    def is_clean(self) -> bool:
        return self.status == CHI_OK

    @property
    def is_all_zero(self) -> bool:
        return self.status == CHI_ALL_ZERO


def scan_chi_column(path: Path) -> ChiScan:
    """Classify the chi column (last of 6) of a FEFF table.

    Parsed with ``str.split``/``float`` rather than ``numpy.genfromtxt`` so a
    Fortran-mangled row ("****" on overflow, a dropped E in a narrow field) is
    counted as malformed instead of poisoning the whole column with NaN, and so
    the counts in the returned :class:`ChiScan` are exact.
    """
    zero_energies: list[float] = []
    n_rows = 0
    n_zero = 0
    n_nonfinite = 0
    n_malformed = 0

    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 6:
                    n_malformed += 1
                    continue
                try:
                    omega = float(fields[0])
                    chi = float(fields[-1])
                except ValueError:
                    n_malformed += 1
                    continue
                n_rows += 1
                if not math.isfinite(chi):
                    n_nonfinite += 1
                elif chi == 0.0:
                    n_zero += 1
                    zero_energies.append(omega)
    except OSError as exc:
        return ChiScan(CHI_UNREADABLE, f"could not read {Path(path).name}: {exc}")

    zero_range = (min(zero_energies), max(zero_energies)) if zero_energies else None
    counts = {
        "n_rows": n_rows,
        "n_zero": n_zero,
        "n_nonfinite": n_nonfinite,
        "n_malformed": n_malformed,
        "zero_energy_range": zero_range,
    }

    if n_rows == 0:
        why = f"no parseable data rows ({n_malformed} malformed)" if n_malformed else "no data rows"
        return ChiScan(CHI_UNREADABLE, why, **counts)

    if n_zero == n_rows:
        return ChiScan(
            CHI_ALL_ZERO,
            f"all {n_rows} chi rows are exactly 0 (mu == mu0 throughout; no fine structure)",
            **counts,
        )

    if n_zero:
        lo, hi = zero_range
        return ChiScan(
            CHI_PARTIAL_ZERO,
            f"{n_zero}/{n_rows} chi rows are exactly 0, omega {lo:.2f}-{hi:.2f} eV",
            **counts,
        )

    if n_nonfinite:
        return ChiScan(
            CHI_NON_FINITE, f"{n_nonfinite}/{n_rows} chi rows are NaN/inf", **counts
        )

    if n_malformed:
        # Usable data, but the file is not pristine; worth surfacing.
        return ChiScan(
            CHI_UNREADABLE,
            f"{n_malformed} malformed rows alongside {n_rows} good ones",
            **counts,
        )

    return ChiScan(CHI_OK, f"{n_rows} chi rows, none zero or non-finite", **counts)


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
