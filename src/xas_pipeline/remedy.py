#!/usr/bin/env python3
"""Deterministic remedy selection for auto-resubmitting failed ORCA runs.

:func:`select_remedy` is a pure function: given a :class:`~xas_pipeline.diagnosis.FailureKind`,
the :class:`~xas_pipeline.diagnosis.Evidence`, and the 1-based index of the
resubmission about to be launched, it returns the :class:`Remedy` to apply -- or
``None`` when the failure is not auto-remediable or the escalation ladder is
exhausted. Keeping it pure makes the whole policy unit-testable without a
scheduler.

The ladder is intentionally short (``MAX_ATTEMPTS`` reruns): SCF-convergence
tricks either work quickly or the run needs a human (a different spin state, a
re-carve, a functional change). Attempt 1 is the gentle/most-likely fix; attempt
2 escalates; beyond that we give up and flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xas_pipeline.diagnosis import Evidence, FailureKind

# Number of automatic reruns allowed per structure (the attempt counter is
# persisted per run dir; see xas_pipeline.rerun_state).
MAX_ATTEMPTS = 2

# Level-shift sub-block used to break an occupation limit cycle by opening the
# effective gap. Lives inside a %scf block.
_LEVEL_SHIFT = "Shift Shift 0.30 ErrOff 0.10 end"
# Finite electronic temperature (K) -> fractional occupations that bridge a
# near-zero gap. The follow-up tight run should read the resulting orbitals with
# smearing off; the stability analysis flag guards against a false minimum.
_SMEAR = "SmearTemp 5000"
_MAXITER = "MaxIter 300"


@dataclass
class Remedy:
    """A concrete set of edits to apply to the ORCA input before resubmitting."""

    label: str
    # Extra simple keywords added as ``! <kw>`` lines (e.g. "SlowConv").
    keywords: list[str] = field(default_factory=list)
    # Lines placed inside a ``%scf ... end`` block.
    scf_lines: list[str] = field(default_factory=list)
    # Read the previous run's orbitals as the SCF guess (``! MOREAD`` + %moinp).
    use_moread: bool = False
    # Restart the geometry optimization from the last completed geometry rather
    # than the original carved geometry.
    opt_restart: bool = False
    # Multiply %MaxCore (and the derived scheduler --mem) by this factor.
    maxcore_mult: float = 1.0
    # Append ``! SCFStabilityAnalysis`` to verify the forced-through wavefunction
    # is a genuine minimum (cheap insurance when the gap went near-zero).
    stability_analysis: bool = False


def select_remedy(kind: FailureKind, evidence: Evidence, attempt: int) -> Remedy | None:
    """Return the Remedy for the given failure and 1-based rerun attempt, or None.

    ``None`` means "do not auto-resubmit": either the failure kind is not
    remediable, or the ladder for this kind has no rung ``attempt``.
    """
    if attempt < 1 or attempt > MAX_ATTEMPTS:
        return None

    restart = evidence.n_opt_cycles >= 2

    if kind is FailureKind.OOM:
        # RIJCOSX/AnFreq ran out of per-process memory: bump %MaxCore (and the
        # derived --mem). No SCF tricks; reuse orbitals if we have them.
        mult = {1: 1.6, 2: 2.5}.get(attempt)
        if mult is None:
            return None
        return Remedy(
            label=f"oom-maxcore-x{mult:g}",
            maxcore_mult=mult,
            use_moread=evidence.gbw_present,
            opt_restart=restart,
        )

    if kind is FailureKind.SCF_NEAR_DEGENERACY:
        if attempt == 1:
            # Smearing is the most reliable cure for a near-zero/negative gap.
            return Remedy(
                label="scf-smear",
                scf_lines=[_SMEAR, _MAXITER],
                use_moread=evidence.gbw_present,
                opt_restart=restart,
                stability_analysis=True,
            )
        # attempt 2: level shift + damping from a fresh guess (a stale GBW may
        # re-seed the same oscillating solution).
        return Remedy(
            label="scf-shift-slowconv",
            keywords=["SlowConv"],
            scf_lines=[_LEVEL_SHIFT, _MAXITER],
            use_moread=False,
            opt_restart=restart,
            stability_analysis=True,
        )

    if kind is FailureKind.SCF_STALLED:
        if attempt == 1:
            # Near-converged, just oscillating above threshold: damp + more
            # iterations, reusing the (good) prior orbitals.
            return Remedy(
                label="scf-slowconv",
                keywords=["SlowConv"],
                scf_lines=[_MAXITER],
                use_moread=evidence.gbw_present,
                opt_restart=restart,
            )
        return Remedy(
            label="scf-smear",
            scf_lines=[_SMEAR, _MAXITER],
            use_moread=evidence.gbw_present,
            opt_restart=restart,
            stability_analysis=True,
        )

    if kind is FailureKind.SCF_DIVERGED:
        if attempt == 1:
            # Energy still moving: damp hard + level shift, from a FRESH guess.
            return Remedy(
                label="scf-slowconv-shift",
                keywords=["SlowConv"],
                scf_lines=[_LEVEL_SHIFT, _MAXITER],
                use_moread=False,
            )
        return Remedy(
            label="scf-smear",
            scf_lines=[_SMEAR, _MAXITER],
            use_moread=False,
            stability_analysis=True,
        )

    if kind is FailureKind.OPT_NONCONVERGENCE:
        if attempt == 1:
            # Geometry did not settle: restart the optimization from the last
            # geometry, reusing orbitals to save the first SCF.
            return Remedy(
                label="opt-restart",
                opt_restart=True,
                use_moread=evidence.gbw_present,
            )
        return None

    # CHARGE_MULT, POST_OPT, CRASH, UNKNOWN, NO_LOG, OK -> never auto-rerun.
    return None
