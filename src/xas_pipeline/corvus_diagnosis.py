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

Three kinds are auto-remediable, all for the same reason -- they are sporadic, so the same inputs recomputed come back
clean:

``XANES_ZERO_CHI``
    The xanes component's ``xmu.dat`` has ``chi`` identically 0 on every row --
    ``mu == mu0`` throughout, i.e. the XANES leg produced no fine structure at
    all and the spectrum carries no structural information. The FEFF header
    cannot show this: the "0/0 paths used" line is meaningless for XANES (an FMS
    calculation, not a path expansion). The failure is
    *sporadic* rather than structural -- the same geometry recomputes fine -- so
    the remedy is simply to archive the dead output and recompute, no input
    changes. That is what makes it safe to automate.

``FMS_GG_HEADER`` and ``FMS_GG_ZERO_SECTION``
    Two FEFF 10 bugs that show only in FMS's ``gg.bin`` (odd-XANES groups A and B; see
    :func:`xas_pipeline.chem.feff.scan_gg_bin` and audits/odd-xanes/feff-bugs-report.html). A carriage return or line
    feed in the ``#DF#`` header makes MKGTR misread or stop reading the file (group A, 1 run in 256); an all-zero
    section means FMS's rank 0 misfiled the workers' results (group B, seen at 16 MPI ranks). FEFF exits 0 and the
    spectrum looks like a spectrum, so only the ``gg.bin`` check catches them. A node-scratch run is checked on the node
    (``stages/corvus_pack.py`` writes ``<id>-gg-check.tsv``); an in-place run's loose ``gg.bin`` is scanned here.
    Runs from before the check have neither and are not checked.

Everything else (no spectrum written at all, a malformed table, an EXAFS leg with
no scattering paths) points at the inputs, the Hessian, or the run being killed,
and recomputing the same thing would just fail the same way -- so those escalate
to a human.

``BONDING_CHANGED`` is not a CORVUS failure at all: the spring-model stage's
bonding gate refused the geometry (its bonding network differs from its
whole-structure spring model's) and CORVUS was never started. It is recognised
by the gate's ``bonding-changed.txt`` and is never rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from xas_pipeline import layout
from xas_pipeline.chem import feff

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
    FMS_GG_HEADER = "fms_gg_header"
    FMS_GG_ZERO_SECTION = "fms_gg_zero_section"
    BONDING_CHANGED = "bonding_changed"


# Kinds the pipeline is allowed to recompute on its own. A dead XANES leg is
# sporadic and reproduces clean, so a plain recompute is a real remedy; every
# other kind means the inputs (or the job) were wrong, and rerunning them
# unchanged would only burn queue time.
AUTO_REMEDIABLE: frozenset[CorvusFailureKind] = frozenset({
    CorvusFailureKind.XANES_ZERO_CHI,
    CorvusFailureKind.FMS_GG_HEADER,
    CorvusFailureKind.FMS_GG_ZERO_SECTION,
})

# The node-scratch run's gg.bin verdict, written by stages/corvus_pack.py next to Corvus.cfavg_<mode>.out.
GG_CHECK_TEMPLATE = "{run_id}-gg-check.tsv"
GG_CHECK_GLOB = "*-gg-check.tsv"
GG_CHECK_COLUMNS = ("component", "status", "sections", "zero_sections", "header_byte", "detail")


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


def scan_gg_files(feff_dir: Path) -> dict[str, feff.GGScan]:
    """Scan each component's loose ``gg.bin`` under a FEFF dir ({} when there are none).

    A leg on which FMS does no work (``fms.inp``: mfms 0, or FMS radius rfms2 <= 0 -- always the combined-xas EXAFS
    leg, rfms2 = -1) is not scanned: its gg.bin is all zeros by design and no artifact in it can reach the spectrum. A
    leg without a readable ``fms.inp`` is scanned.
    """
    scans = {}
    for comp in XAS_COMPONENTS:
        path = Path(feff_dir) / comp / "gg.bin"
        if not path.is_file():
            continue
        if feff.fms_active(path.parent / "fms.inp") is False:
            scans[comp] = feff.GGScan(feff.GG_FMS_OFF, "fms.inp: FMS off on this leg (mfms 0 or rfms2 <= 0), gg.bin "
                                                       "all zero by design; not checked")
        else:
            scans[comp] = feff.scan_gg_bin(path)
    return scans


def write_gg_check(feff_dir: Path, dest: Path) -> dict[str, feff.GGScan]:
    """Scan the loose gg.bin files and write the verdict table *dest* (nothing written when there are none)."""
    scans = scan_gg_files(feff_dir)
    if scans:
        rows = ["\t".join(GG_CHECK_COLUMNS)]
        for comp, scan in scans.items():
            rows.append("\t".join([comp, scan.status, str(scan.n_sections),
                                   ",".join(map(str, scan.zero_sections)) or "-",
                                   f"0x{scan.header_byte:02X}" if scan.header_byte is not None else "-",
                                   scan.detail.replace("\t", " ")]))
        Path(dest).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return scans


def read_gg_check(path: Path) -> dict[str, tuple[str, str]]:
    """component -> (status, detail) from a gg-check table."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= len(GG_CHECK_COLUMNS):
            status = fields[1]
            # Tables from before the FMS-off rule (ab73a16) flagged the EXAFS leg's all-zero gg.bin: every section zero
            # means FMS computed nothing on that leg, never the misfiling bug.
            if status == feff.GG_ZERO_SECTION and fields[2].isdigit() and fields[3] != "-" \
                    and len(fields[3].split(",")) == int(fields[2]):
                status = feff.GG_FMS_OFF
            out[fields[0]] = (status, fields[-1])
    return out


def gg_verdict(root: Path, feff_dir: Path) -> CorvusDiagnosis | None:
    """The FMS_GG_* verdict when a gg.bin carries a known FEFF artifact, else None.

    Loose gg.bin files (an in-place run) are scanned; otherwise the node's ``*-gg-check.tsv`` at *root* is read. Neither
    present (a run from before the check) means no verdict.
    """
    scans = {comp: (scan.status, scan.detail) for comp, scan in scan_gg_files(feff_dir).items()}
    if not scans:
        tables = sorted(Path(root).glob(GG_CHECK_GLOB))
        if not tables:
            return None
        scans = read_gg_check(tables[0])
    for comp, (status, detail) in scans.items():
        if status in (feff.GG_HEADER_CR, feff.GG_HEADER_LF):
            return CorvusDiagnosis(ok=False, kind=CorvusFailureKind.FMS_GG_HEADER, reason=f"{comp} {detail}")
    for comp, (status, detail) in scans.items():
        if status == feff.GG_ZERO_SECTION:
            return CorvusDiagnosis(ok=False, kind=CorvusFailureKind.FMS_GG_ZERO_SECTION, reason=f"{comp} {detail}")
    return None


def diagnose_spectrum(cfavg_path: Path, feff_dir: Path) -> CorvusDiagnosis:
    """Classify a combined xas run from its deliverable + FEFF component dirs.

    Checked worst-first: no known FEFF artifact in gg.bin; the deliverable must exist and hold a non-empty
    6-column table; the exafs leg must have found scattering paths; the xanes
    leg's chi must not be identically zero.
    """
    cfavg_path = Path(cfavg_path)
    # First: a gg.bin artifact explains (and remedies) whatever else is wrong, including a spectrum MKGTR never
    # finished -- CORVUS does not check FEFF's exit codes.
    gg = gg_verdict(cfavg_path.parent, feff_dir)
    if gg is not None:
        return gg
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
    if exafs_xmu.is_file() and feff.xmu_reports_zero_paths(exafs_xmu):
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
        scan = feff.scan_chi_column(xanes_xmu)
        if scan.is_all_zero:
            return CorvusDiagnosis(
                ok=False,
                kind=CorvusFailureKind.XANES_ZERO_CHI,
                reason=f"xanes xmu.dat has no fine structure: {scan.detail}",
            )
        if not scan.is_clean:
            warnings.append(f"xanes xmu.dat chi is {scan.status}: {scan.detail}")

    return CorvusDiagnosis(ok=True, kind=CorvusFailureKind.OK, reason="ok", warnings=warnings)


def bonding_gate_verdict(root: Path) -> CorvusDiagnosis | None:
    """The BONDING_CHANGED verdict when the bonding gate refused the run at *root*, else None."""
    from xas_pipeline.stages.interp_hessian import BONDING_CHANGED_NAME

    marker = Path(root) / BONDING_CHANGED_NAME
    if not marker.is_file():
        return None
    bonds = [line.strip() for line in marker.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("    ")]
    return CorvusDiagnosis(
        ok=False,
        kind=CorvusFailureKind.BONDING_CHANGED,
        reason=("bonding network differs from the spring model, so no spectrum by design "
                f"(bonding gate; {BONDING_CHANGED_NAME}: {'; '.join(bonds) or 'see file'})"),
    )


def diagnose(working_root: Path, mode: str = "xas") -> CorvusDiagnosis:
    """Classify the ``mode`` run under one working root."""
    working_root = Path(working_root)
    gated = bonding_gate_verdict(working_root)
    if gated is not None:
        return gated
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
        gated = bonding_gate_verdict(root)
        if gated is not None:
            return gated
    for root in roots:
        if cfavg_output(root, mode).is_file():
            return diagnose(root, mode)
    return diagnose(roots[-1] if roots else run_dir, mode)
