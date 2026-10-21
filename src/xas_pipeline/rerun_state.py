#!/usr/bin/env python3
"""Per-run, per-ladder record of automatic resubmissions.

Each ladder keeps its own JSON file in the run dir, so its attempt counter is
bounded and re-triage is idempotent: ``<id>-rerun-state.json`` (ORCA failures,
xas-rerun-orca), ``<id>-corvus-rerun-state.json`` (dead XANES legs),
``<id>-hang-rerun-state.json`` / ``<id>-corvus-hang-rerun-state.json`` (the
watchdog). A set ``resolution`` ends the ladder.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import batch_log


@dataclass
class Attempt:
    attempt: int
    kind: str
    remedy: str
    utc: str
    orca_job_id: str | None = None
    input_backup: str | None = None
    note: str | None = None
    corvus_job_id: str | None = None


RESOLUTION_NEEDS_HUMAN = "needs_human"


@dataclass
class RerunState:
    run_id: str
    attempts: list[Attempt] = field(default_factory=list)
    resolution: str | None = None

    @property
    def next_attempt(self) -> int:
        return len(self.attempts) + 1

    @property
    def is_terminal(self) -> bool:
        return self.resolution is not None


# Ladder -> state-file infix (kept separate so a hang and a real input failure
# never share one counter).
_STATE_INFIX = {"orca": "", "corvus": "-corvus", "hang": "-hang", "corvus-hang": "-corvus-hang"}


def state_path(run_dir: Path, run_id: str, kind: str = "orca") -> Path:
    """The state file for ``run_id``'s ``kind`` ladder ('orca', 'corvus', 'hang', 'corvus-hang')."""
    try:
        infix = _STATE_INFIX[kind]
    except KeyError:
        raise ValueError(f"unknown rerun ladder {kind!r}; expected one of {sorted(_STATE_INFIX)}") from None
    return run_dir / f"{run_id}{infix}-rerun-state.json"


def load_state(path: Path, run_id: str) -> RerunState:
    """The ladder state at *path* (fresh if absent).

    An unreadable file is moved aside to ``<name>.corrupt`` and the ladder is
    ended rather than restarted (a reset counter could resubmit forever): the
    needs_human state is written in its place, so later runs see it too, and a
    ``NEEDS_HUMAN`` line goes to the batch's batch-jobs.log.
    """
    if not path.is_file():
        return RerunState(run_id=run_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RerunState(
            run_id=raw.get("run_id", run_id),
            attempts=[Attempt(**a) for a in raw.get("attempts", [])],
            resolution=raw.get("resolution"),
        )
    except (ValueError, TypeError, AttributeError):
        corrupt = path.with_name(path.name + ".corrupt")
        path.replace(corrupt)
        state = RerunState(run_id=run_id, resolution=RESOLUTION_NEEDS_HUMAN)
        save_state(path, state)
        root = batch_log.find_batch_root(path.parent)
        if root is not None:
            batch_log.append_outcomes(root / batch_log.LOG_NAME, "Rerun state",
                                      [(run_id, "NEEDS_HUMAN", f"unreadable {path.name}; kept as {corrupt.name}")])
        return state


def save_state(path: Path, state: RerunState) -> None:
    """Write atomically, so a job killed mid-write cannot truncate the ladder."""
    payload = {
        "run_id": state.run_id,
        "resolution": state.resolution,
        "attempts": [asdict(a) for a in state.attempts],
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
