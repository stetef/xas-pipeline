"""FEFF/CORVUS output tables, the chi(k) -> chi(R) transform, and chi-column health.

:func:`xftf_larch` imports larch (the optional ``exafs`` extra) on first use.

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

# --- gg.bin: FMS's output, checked for two FEFF 10 bugs ---------------------------------------------------------------
# gg.bin holds one Green's-function matrix per energy point, one ``#SN#`` section each, written by FMS
# (src/FMS/fmstot.f90:420 -> WriteComplex2D in src/IOMODS/m_iomod.f90). Two uninitialised variables in FEFF 10 leave
# their mark here and nowhere else (audits/odd-xanes/feff-bugs-report.html, 2026-11-20):
#
# * ``FlType`` (m_iomod.f90:2465) is never set, so each section's ``#DF# This section written in ????.`` line carries 4
#   leftover bytes, ``XX 7f 00 00`` with XX random per run. XX = CR (0x0D): MKGTR misreads the file and gtr.dat is wrong
#   at every energy (odd-XANES group A: edge ~4 eV early, ringing). XX = LF (0x0A): MKGTR stops with "Unexpected end of
#   record" and the spectrum is the path sum only. 1 run in 256 each.
# * ``map`` (fmstot.f90:110) is never cleared, so rank 0 can file a worker's results one slot early; the worker's last
#   slot is never filled and stays all zeros (group B: spikes, end-of-range spikes). Seen at 16 MPI ranks only.
#
# Both are sporadic: the same inputs rerun usually come back clean, so they are auto-remediable.

GG_OK = "ok"
GG_HEADER_CR = "header-cr"
GG_HEADER_LF = "header-lf"
GG_ZERO_SECTION = "zero-section"
GG_UNREADABLE = "unreadable"
# FMS does no work on this leg: fms.inp has mfms = 0 (MKGTR then never reads gg.bin, mkgtr.f90:42) or an FMS radius
# rfms2 <= 0 (an empty cluster: every gg matrix is zero by design, so even a misread returns zeros). The combined-xas
# EXAFS leg is always the second kind (rfms2 = -1, path expansion only). Not scanned; recorded so the table says why.
GG_FMS_OFF = "fms-off"

GG_HEADER_PREFIX = b"#DF# This section written in "
_GG_BAD_HEADER_BYTES = {0x0D: GG_HEADER_CR, 0x0A: GG_HEADER_LF}


@dataclass(frozen=True)
class GGScan:
    """Verdict on one gg.bin. ``status`` is a ``GG_*`` constant; ``detail`` names the evidence."""

    status: str
    detail: str
    n_sections: int = 0
    zero_sections: tuple[int, ...] = ()
    header_byte: int | None = None

    @property
    def is_clean(self) -> bool:
        return self.status == GG_OK

    @property
    def is_artifact(self) -> bool:
        """One of the two known FEFF bugs (rerunning is the remedy)."""
        return self.status in (GG_HEADER_CR, GG_HEADER_LF, GG_ZERO_SECTION)


def fms_active(fms_inp: Path) -> bool | None:
    """Whether FMS computes anything on a leg, from its fms.inp (FEFF's fms_write layout): ``mfms`` is the first integer
    of line 2 (under ``mfms, idwopt, minv``), ``rfms2`` the first number of line 4 (under ``rfms2, rdirec, ...``).
    True when mfms = 1 and rfms2 > 0; None when the file is missing or not in that layout."""
    try:
        lines = Path(fms_inp).read_text(encoding="utf-8", errors="replace").splitlines()
        if not (lines[0].strip().lower().startswith("mfms") and lines[2].strip().lower().startswith("rfms2")):
            return None
        return int(lines[1].split()[0]) == 1 and float(lines[3].split()[0]) > 0
    except (OSError, IndexError, ValueError):
        return None


def _gg_section_values(chunk: bytes) -> np.ndarray:
    """The numbers in one gg.bin section (header lines and anything unparseable skipped)."""
    rows = [line for line in chunk.split(b"\n")[1:] if line.strip() and not line.lstrip().startswith(b"#")]
    tokens = b" ".join(rows).split()
    try:
        return np.array(tokens).astype(float)
    except ValueError:
        values = []
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError:
                pass
        return np.array(values, dtype=float)


def scan_gg_bin(path: Path) -> GGScan:
    """Check a FEFF gg.bin for the header-byte bug (group A) and an all-zero section (group B).

    Read as bytes: the junk byte can be a CR or LF, which text-mode reading would swallow or split on. The header byte
    is the first character after ``written in `` in each ``#DF#`` line (byte 30 of the line); all sections of one file
    carry the same junk, but every one is checked.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return GGScan(GG_UNREADABLE, f"could not read {Path(path).name}: {exc}")

    chunks = data.split(b"#SN#")[1:]
    if not chunks:
        return GGScan(GG_UNREADABLE, "gg.bin has no '#SN#' sections")
    zero = tuple(n for n, chunk in enumerate(chunks, 1) if not np.any(_gg_section_values(chunk)))
    # Every section zero: FMS computed nothing on this leg (empty cluster, e.g. the EXAFS leg's rfms2 = -1). Not the
    # misfiling bug, which leaves one empty slot per affected rank, and a misread of zeros is still zeros.
    if len(zero) == len(chunks):
        return GGScan(GG_FMS_OFF, f"all {len(chunks)} sections zero: FMS computed nothing on this leg; not checked",
                      n_sections=len(chunks))

    pos = data.find(GG_HEADER_PREFIX)
    while pos != -1:
        at = pos + len(GG_HEADER_PREFIX)
        byte = data[at] if at < len(data) else None
        if byte in _GG_BAD_HEADER_BYTES:
            status = _GG_BAD_HEADER_BYTES[byte]
            return GGScan(status, f"gg.bin '#DF#' header byte is 0x{byte:02X} ({'CR' if byte == 0x0D else 'LF'}): "
                                  f"MKGTR {'misreads the file' if byte == 0x0D else 'stops reading it'} "
                                  "(FEFF uninitialised FlType)", header_byte=byte,
                          n_sections=data.count(b"#SN#"))
        pos = data.find(GG_HEADER_PREFIX, at)

    if zero:
        shown = ", ".join(map(str, zero[:10])) + (" ..." if len(zero) > 10 else "")
        return GGScan(GG_ZERO_SECTION, f"gg.bin section(s) {shown} of {len(chunks)} are all zero: FMS never filled "
                                       "them (FEFF uninitialised map, results misfiled)",
                      n_sections=len(chunks), zero_sections=zero)
    return GGScan(GG_OK, f"{len(chunks)} sections, header byte fine, none all zero", n_sections=len(chunks))


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
