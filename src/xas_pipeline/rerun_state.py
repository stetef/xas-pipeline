#!/usr/bin/env python3
"""Per-run auto-rerun attempt state.

A small JSON file in the run directory records each automatic resubmission so the
ladder is bounded and idempotent: the end-of-job hook can fire ``xas-rerun-orca``
every time a run fails, and this state is what stops it from resubmitting forever.
One record per attempt.

Two independent ladders share this machinery, one file each so their attempt
counters never interfere (see :func:`state_path`):

* ``<id>-rerun-state.json`` -- ORCA (:mod:`xas_pipeline.cli.rerun_orca`);
* ``<id>-corvus-rerun-state.json`` -- CORVUS
  (:mod:`xas_pipeline.cli.auto_rerun_corvus`, recomputing a dead XANES leg).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Attempt:
    attempt: int
    kind: str
    remedy: str
    utc: str
    orca_job_id: str | None = None
    input_backup: str | None = None
    note: str | None = None
    # Set by the CORVUS ladder, which resubmits a corvus wrapper rather than an
    # ORCA job. Optional (and read back with a default) so a state file written
    # by either ladder, before or after this field existed, still loads.
    corvus_job_id: str | None = None


# Terminal resolutions recorded when the ladder stops (auto-rerun gives up). A
# non-None resolution makes re-triage idempotent: the hook can fire again without
# re-escalating.
RESOLUTION_NEEDS_HUMAN = "needs_human"


@dataclass
class RerunState:
    run_id: str
    attempts: list[Attempt] = field(default_factory=list)
    # None while auto-rerun is still in play; set to a RESOLUTION_* string once
    # the ladder terminates (not remediable / exhausted / no remedy).
    resolution: str | None = None

    @property
    def next_attempt(self) -> int:
        return len(self.attempts) + 1

    @property
    def is_terminal(self) -> bool:
        return self.resolution is not None


# Ladder name -> state-file infix. The ORCA ladder keeps the original, unprefixed
# filename so state files already on disk stay authoritative.
_STATE_INFIX = {"orca": "", "corvus": "-corvus"}


def state_path(run_dir: Path, run_id: str, kind: str = "orca") -> Path:
    """The state file for ``run_id``'s ``kind`` ladder ('orca' or 'corvus')."""
    try:
        infix = _STATE_INFIX[kind]
    except KeyError:
        raise ValueError(f"unknown rerun ladder {kind!r}; expected one of {sorted(_STATE_INFIX)}") from None
    return run_dir / f"{run_id}{infix}-rerun-state.json"


def load_state(path: Path, run_id: str) -> RerunState:
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            attempts = [Attempt(**a) for a in raw.get("attempts", [])]
            return RerunState(
                run_id=raw.get("run_id", run_id),
                attempts=attempts,
                resolution=raw.get("resolution"),
            )
        except (ValueError, TypeError):
            # Corrupt/incompatible state: start fresh rather than crash the hook.
            return RerunState(run_id=run_id)
    return RerunState(run_id=run_id)


def save_state(path: Path, state: RerunState) -> None:
    payload = {
        "run_id": state.run_id,
        "resolution": state.resolution,
        "attempts": [asdict(a) for a in state.attempts],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
