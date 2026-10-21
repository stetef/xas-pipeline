"""Machine-readable diagnosis of a failed ORCA run.

``orca_check.classify_orca_run`` returns a human-readable ``(ok, reason)`` pair
for the batch convergence report. This module is its machine-readable sibling:
``diagnose(run_dir)`` returns a :class:`Diagnosis` carrying a :class:`FailureKind`
enum plus an :class:`Evidence` record, so a downstream policy
(:mod:`xas_pipeline.remedy`) can deterministically decide whether -- and how --
to auto-resubmit the run.

The failure signatures (regexes, marker texts) live in
:mod:`xas_pipeline.stages.orca_check`. Beyond the batch classifier, SCF
non-convergence is split three ways so the escalation names the likely cure
(none of them is applied automatically):

* ``SCF_NEAR_DEGENERACY`` -- ``Small HOMO/LUMO gap`` warnings: the SCF oscillates
  between near-degenerate frontier occupations. Cure: level shift + damping
  (SlowConv); not smearing, which ``! AnFreq`` cannot use.
* ``SCF_STALLED`` -- no small gap, energy flat over the last iterations. Cure:
  SlowConv + more iterations from the prior orbitals (MOREAD).
* ``SCF_DIVERGED`` -- energy still moving. Cure: SlowConv + level shift from a
  fresh guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from xas_pipeline import layout
from xas_pipeline.stages import orca_check as oc


class FailureKind(str, Enum):
    """What went wrong with an ORCA run (or OK)."""

    OK = "ok"
    NO_LOG = "no_log"
    CHARGE_MULT = "charge_mult"
    OOM = "oom"
    SCF_NEAR_DEGENERACY = "scf_near_degeneracy"
    SCF_STALLED = "scf_stalled"
    SCF_DIVERGED = "scf_diverged"
    OPT_NONCONVERGENCE = "opt_nonconvergence"
    POST_OPT = "post_opt"
    CRASH = "crash"
    UNKNOWN = "unknown"


# Kinds a script may auto-resubmit: only the two physics-neutral resource
# failures, an out-of-memory kill (more memory) and an optimization out of cycles
# (restart from the last geometry). Everything else escalates to a human:
# SCF non-convergence (curing it means changing SCF settings, which tetef01
# decided stays manual, 2026-09-21), charge/mult parity errors, post-optimization
# crashes, other crashes and missing logs.
AUTO_REMEDIABLE: frozenset[FailureKind] = frozenset(
    {
        FailureKind.OOM,
        FailureKind.OPT_NONCONVERGENCE,
    }
)

SMALL_GAP_TEXT = "Small HOMO/LUMO gap"
OPT_CYCLE_MARKER = "GEOMETRY OPTIMIZATION CYCLE"
# An SCF-iteration table row begins with the cycle index then the total energy,
# e.g. "  119      -4947.095053283695     2.195054e-05  (NR   MAcro)".
_SCF_ITER_RE = re.compile(r"^\s*\d+\s+(-\d+\.\d+)\s")
# How stable the last SCF energies must be (Eh range) to call it a "stall"
# rather than a divergence.
_ENERGY_STABLE_RANGE_EH = 1.0e-3


@dataclass
class Evidence:
    """Facts extracted from the log/run dir that drive remedy selection."""

    small_gap: bool = False
    gbw_present: bool = False
    n_opt_cycles: int = 0
    energy_stable: bool = False
    failed_module: str | None = None


@dataclass
class Diagnosis:
    ok: bool
    kind: FailureKind
    reason: str
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def auto_remediable(self) -> bool:
        return self.kind in AUTO_REMEDIABLE


def _scf_energy_stable(log_text: str) -> bool:
    """True if the last stretch of SCF-iteration energies is essentially flat.

    A stalled SCF (near-converged, just oscillating above threshold) holds its
    energy to ~1e-6 while the density flips; a diverging SCF keeps moving. We
    look at the tail of all iteration energies in the log.
    """
    energies: list[float] = []
    for line in log_text.splitlines():
        m = _SCF_ITER_RE.match(line)
        if m:
            try:
                energies.append(float(m.group(1)))
            except ValueError:
                pass
    if len(energies) < 5:
        return False
    tail = energies[-20:]
    return (max(tail) - min(tail)) < _ENERGY_STABLE_RANGE_EH


def _gbw_present(run_dir: Path, log_path: Path | None) -> bool:
    """True if a non-empty .gbw (usable as an SCF restart guess) exists.

    Searched in the log's dir and the run's own working roots only: a legacy
    run dir can host other modes' runs, whose orbitals are not this run's.
    """
    search_dirs = [log_path.parent] if log_path is not None else []
    search_dirs += layout.working_roots(run_dir)
    for directory in dict.fromkeys(search_dirs):
        for gbw in directory.glob("*.gbw"):
            try:
                if gbw.stat().st_size > 0:
                    return True
            except OSError:
                pass
    return False


def diagnose(run_dir: Path) -> Diagnosis:
    """Classify an ORCA run directory into a :class:`Diagnosis`."""
    log_path = oc.find_orca_log(run_dir)
    if log_path is None:
        return Diagnosis(
            ok=False,
            kind=FailureKind.NO_LOG,
            reason="ORCA log not found (job produced no log; likely never started)",
        )
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Diagnosis(
            ok=False,
            kind=FailureKind.NO_LOG,
            reason=f"failed to read {log_path.name} ({exc})",
        )

    ev = Evidence()
    ev.small_gap = SMALL_GAP_TEXT in log_text
    ev.n_opt_cycles = log_text.count(OPT_CYCLE_MARKER)
    ev.gbw_present = _gbw_present(run_dir, log_path)
    err = oc.ERROR_TERM_RE.search(log_text)
    ev.failed_module = err.group(1).upper() if err else None

    # Geometry optimization that ran out of cycles -- checked first (as in the
    # batch classifier): the electronic structure is fine, the geometry just did
    # not settle, so it is restartable from the last geometry.
    if oc.NON_CONVERGENCE_TEXT in log_text:
        return Diagnosis(
            ok=False,
            kind=FailureKind.OPT_NONCONVERGENCE,
            reason="optimization did not converge (reached the maximum number of cycles)",
            evidence=ev,
        )

    if oc.TERMINATION_MARKER in log_text:
        return Diagnosis(ok=True, kind=FailureKind.OK, reason="terminated normally", evidence=ev)

    # No normal termination -> name the specific cause.
    if oc.CHARGE_MULT_RE.search(log_text):
        return Diagnosis(
            ok=False,
            kind=FailureKind.CHARGE_MULT,
            reason=(
                "charge/multiplicity parity error -- the ORCA input never ran. "
                "Fix CHARGE/MULT (or the carved cluster); NOT auto-remediable"
            ),
            evidence=ev,
        )

    if oc.OOM_COSX_TEXT in log_text or oc.OOM_MAXCORE_TEXT in log_text:
        where = f" in the {ev.failed_module} step" if ev.failed_module else ""
        return Diagnosis(
            ok=False,
            kind=FailureKind.OOM,
            reason=f"out of memory{where}: RIJCOSX exchange build exhausted per-process memory",
            evidence=ev,
        )

    if oc.SCF_NONCONV_TEXT in log_text or oc.SCF_NONCONV_TEXT_ALT in log_text:
        ev.energy_stable = _scf_energy_stable(log_text)
        if ev.small_gap:
            kind = FailureKind.SCF_NEAR_DEGENERACY
            reason = "SCF did not converge; small/negative HOMO-LUMO gap -> occupation oscillation"
        elif ev.energy_stable:
            kind = FailureKind.SCF_STALLED
            reason = "SCF did not converge; energy stable (last-mile stall above threshold)"
        else:
            kind = FailureKind.SCF_DIVERGED
            reason = "SCF did not converge; energy still moving (divergence)"
        return Diagnosis(ok=False, kind=kind, reason=reason, evidence=ev)

    if ev.failed_module in oc.POST_OPT_MODULES:
        return Diagnosis(
            ok=False,
            kind=FailureKind.POST_OPT,
            reason=(
                f"post-optimization step failed (error terminated in {ev.failed_module}); "
                "optimization completed but no Hessian written; NOT auto-remediable"
            ),
            evidence=ev,
        )

    return Diagnosis(
        ok=False,
        kind=FailureKind.CRASH,
        reason="ORCA did not terminate normally (crashed or was killed mid-run); NOT auto-remediable",
        evidence=ev,
    )
