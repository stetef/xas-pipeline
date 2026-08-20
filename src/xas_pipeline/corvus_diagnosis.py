#!/usr/bin/env python3
"""Machine-readable verdict on a finished CORVUS run.

This is the CORVUS-side sibling of :mod:`xas_pipeline.diagnosis` (which does the
same job for ORCA logs): :func:`diagnose` looks at what a combined
``cfavg_target{xas}`` run left on disk and returns a :class:`CorvusDiagnosis`
carrying a :class:`CorvusFailureKind`, so two callers can share one source of
truth:

* the postprocess gate (:func:`xas_pipeline.stages.feff_process.xas_is_valid`),
  which decides whether an id counts as a CORVUS failure;
* the auto-rerun policy (:mod:`xas_pipeline.cli.auto_rerun_corvus`), which decides
  whether that failure is worth recomputing automatically.

Only one kind is auto-remediable, and for a specific reason:

``XANES_ZERO_CHI``
    The xanes component's ``xmu.dat`` has ``chi`` identically 0 on every row --
    ``mu == mu0`` throughout, i.e. the XANES leg produced no fine structure at
    all and the spectrum carries no structural information. The FEFF header
    cannot show this: the "0/0 paths used" line is meaningless for XANES (an FMS
    calculation, not a path expansion), which is why the gate used to check only
    the exafs component and let a dead XANES leg through. The failure is
    *sporadic* rather than structural -- the same geometry recomputes fine -- so
    the remedy is simply to archive the dead output and recompute, no input
    changes. That is what makes it safe to automate.

Everything else (no spectrum written at all, a malformed table, an EXAFS leg with
no scattering paths) points at the inputs, the Hessian, or the run being killed,
and recomputing the same thing would just fail the same way -- so those escalate
to a human exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from xas_pipeline import layout
from xas_pipeline.chem import feff as _chem_feff

# CORVUS's combined configurationally-averaged spectrum (6-col xmu-like table),
# written at the run root. The deliverable of a cfavg_target{xas} run.
CFAVG_OUTPUT_TEMPLATE = "Corvus.cfavg_{mode}.out"
# Per-component FEFF outputs live in subdirs of the FEFF dir.
XAS_COMPONENTS = ("xanes", "exafs")


class CorvusFailureKind(str, Enum):
    """What went wrong with a CORVUS run (or OK)."""

    OK = "ok"
    MISSING_SPECTRUM = "missing_spectrum"
    MALFORMED_SPECTRUM = "malformed_spectrum"
    NO_EXAFS_PATHS = "no_exafs_paths"
    XANES_ZERO_CHI = "xanes_zero_chi"


# Kinds the pipeline is allowed to recompute on its own. A dead XANES leg is
# sporadic and reproduces clean, so a plain recompute is a real remedy; every
# other kind means the inputs (or the job) were wrong, and rerunning them
# unchanged would only burn queue time.
AUTO_REMEDIABLE: frozenset[CorvusFailureKind] = frozenset({CorvusFailureKind.XANES_ZERO_CHI})


@dataclass
class CorvusDiagnosis:
    ok: bool
    kind: CorvusFailureKind
    reason: str
    # Non-fatal findings worth printing next to the verdict: a partly-zero or
    # NaN-carrying chi column is suspicious but not necessarily dead output, so
    # it is surfaced rather than failing the run.
    warnings: list[str] = field(default_factory=list)

    @property
    def auto_remediable(self) -> bool:
        return self.kind in AUTO_REMEDIABLE


def cfavg_output(working_root: Path, mode: str = "xas") -> Path:
    """Path to the combined 6-col spectrum for ``mode`` at a working root."""
    return Path(working_root) / CFAVG_OUTPUT_TEMPLATE.format(mode=mode)


def mode_feff_dir(working_root: Path, mode: str = "xas") -> Path:
    """Resolve the CORVUS FEFF dir for a mode.

    CORVUS names it Corvus1Zn_<absorber-index>_FEFF (the index varies per
    structure, e.g. Corvus1Zn_0_FEFF here, Corvus1Zn_32_FEFF elsewhere), so we
    glob rather than hardcode. Returns the first match, or a deterministic
    non-existent path when none is present (so callers' is-it-there probes
    report False).
    """
    mode_root = Path(working_root) / f"Corvus3_cfavg_{mode}"
    matches = sorted(mode_root.glob("Corvus1Zn_*_FEFF"))
    return matches[0] if matches else mode_root / "Corvus1Zn_FEFF"


def component_xmu(feff_dir: Path, component: str) -> Path:
    """Path to a component's xmu.dat inside a combined-xas FEFF dir."""
    return Path(feff_dir) / component / "xmu.dat"


def diagnose_spectrum(cfavg_path: Path, feff_dir: Path) -> CorvusDiagnosis:
    """Classify a combined xas run from its deliverable + FEFF component dirs.

    Checked worst-first: the deliverable must exist and hold a non-empty
    6-column table; the exafs leg must have found scattering paths; the xanes
    leg's chi must not be identically zero.
    """
    cfavg_path = Path(cfavg_path)
    if not cfavg_path.is_file():
        return CorvusDiagnosis(
            ok=False,
            kind=CorvusFailureKind.MISSING_SPECTRUM,
            reason=f"missing {cfavg_path.name} (CORVUS produced no combined XAS spectrum)",
        )
    try:
        data = np.genfromtxt(cfavg_path, comments="#")
    except (OSError, ValueError) as exc:
        return CorvusDiagnosis(
            ok=False,
            kind=CorvusFailureKind.MALFORMED_SPECTRUM,
            reason=f"could not read {cfavg_path.name}: {exc}",
        )
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 6:
        return CorvusDiagnosis(
            ok=False,
            kind=CorvusFailureKind.MALFORMED_SPECTRUM,
            reason=f"{cfavg_path.name} is empty or malformed (expected a 6-column table)",
        )

    # EXAFS sanity check from the header. XANES legitimately reports 0/0 paths
    # (FMS, not a path expansion), so this is exafs-only.
    exafs_xmu = component_xmu(feff_dir, "exafs")
    if exafs_xmu.is_file() and _chem_feff.xmu_reports_zero_paths(exafs_xmu):
        return CorvusDiagnosis(
            ok=False,
            kind=CorvusFailureKind.NO_EXAFS_PATHS,
            reason="exafs xmu.dat reports 0/0 paths used (FEFF found no EXAFS scattering paths)",
        )

    # XANES fine structure, from the numbers rather than the header: the one
    # failure a header check cannot see (see this module's docstring).
    warnings: list[str] = []
    xanes_xmu = component_xmu(feff_dir, "xanes")
    if xanes_xmu.is_file():
        scan = _chem_feff.scan_chi_column(xanes_xmu)
        if scan.is_all_zero:
            return CorvusDiagnosis(
                ok=False,
                kind=CorvusFailureKind.XANES_ZERO_CHI,
                reason=f"xanes xmu.dat has no fine structure: {scan.detail}",
            )
        if not scan.is_clean:
            warnings.append(f"xanes xmu.dat chi is {scan.status}: {scan.detail}")

    return CorvusDiagnosis(ok=True, kind=CorvusFailureKind.OK, reason="ok", warnings=warnings)


def diagnose(working_root: Path, mode: str = "xas") -> CorvusDiagnosis:
    """Classify the ``mode`` run under one working root."""
    working_root = Path(working_root)
    return diagnose_spectrum(cfavg_output(working_root, mode), mode_feff_dir(working_root, mode))


def diagnose_run_dir(run_dir: Path, mode: str = "xas") -> CorvusDiagnosis:
    """Classify a run directory, flat or split.

    A post-processed run keeps its artifacts in ``working-<id>/``; one that has
    not been through process-feff yet is flat. Both are scanned (via
    :func:`xas_pipeline.layout.working_roots`) and the first root holding the
    deliverable wins; with no deliverable anywhere the verdict is the
    missing-spectrum one from the canonical (split) location.
    """
    run_dir = Path(run_dir)
    roots = layout.working_roots(run_dir)
    for root in roots:
        if cfavg_output(root, mode).is_file():
            return diagnose(root, mode)
    return diagnose(roots[-1] if roots else run_dir, mode)
