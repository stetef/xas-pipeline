#!/usr/bin/env python3
"""Deterministic remedy selection for auto-resubmitting failed ORCA runs.

:func:`select_remedy` is a pure function: given a :class:`~xas_pipeline.diagnosis.FailureKind`,
the :class:`~xas_pipeline.diagnosis.Evidence`, and the 1-based index of the
resubmission about to be launched, it returns the :class:`Remedy` to apply -- or
``None`` when the failure is not auto-remediable or the escalation ladder is
exhausted. Keeping it pure makes the whole policy unit-testable without a
scheduler.

The ladder is intentionally short (``MAX_ATTEMPTS`` reruns) and deliberately
narrow: it only applies *physics-neutral* remedies -- more memory (OOM) and a
geometry-optimization restart from the last geometry. It does NOT touch the SCF
solver settings.

Rationale (tetef01, 2026-09-21): auto-rerun should recover from resource and
infrastructure problems, not quietly re-tune the calculation. SCF-convergence
tricks (SlowConv damping, level shift, MaxIter) do not change the converged
answer, but they *do* edit the ORCA input, and the decision was to keep the
input the pipeline runs identical to the one a human wrote. So an SCF that will
not converge is now escalated to a human (see
:data:`xas_pipeline.diagnosis.AUTO_REMEDIABLE`) rather than retried with solver
aids. A silent MPI *hang* -- a job that stops making progress without ever
failing -- is a separate concern handled live by :mod:`xas_pipeline.cli.watchdog`,
not here (this module only runs after a job has already exited non-zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xas_pipeline.diagnosis import Evidence, FailureKind

# Number of automatic reruns allowed per structure (the attempt counter is
# persisted per run dir; see xas_pipeline.rerun_state).
MAX_ATTEMPTS = 2


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

    if kind is FailureKind.OPT_NONCONVERGENCE:
        if attempt == 1:
            # Geometry did not settle: restart the optimization from the last
            # geometry, reusing orbitals to save the first SCF. Physics-neutral --
            # the same input, continued from where it got to.
            return Remedy(
                label="opt-restart",
                opt_restart=True,
                use_moread=evidence.gbw_present,
            )
        return None

    # SCF non-convergence (near-degeneracy / stalled / diverged) is intentionally
    # NOT auto-remedied any more: curing it means editing the SCF solver settings
    # in the input, and the policy (see the module docstring) is to escalate to a
    # human instead. Together with CHARGE_MULT, POST_OPT, CRASH, UNKNOWN, NO_LOG,
    # OK -> never auto-rerun. These kinds are already excluded by
    # diagnosis.AUTO_REMEDIABLE, so rerun_orca escalates before ever calling this;
    # returning None keeps select_remedy correct on its own regardless.
    return None
