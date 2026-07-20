#!/usr/bin/env python3
"""Per-run auto-rerun attempt state.

A small JSON file (``<id>-rerun-state.json``) in the run directory records each
automatic resubmission so the ladder is bounded and idempotent: the end-of-job
hook can fire ``xas-rerun-orca`` every time a run fails, and this state is what
stops it from resubmitting forever. One record per attempt.
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


@dataclass
class RerunState:
    run_id: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def next_attempt(self) -> int:
        return len(self.attempts) + 1


def state_path(run_dir: Path, run_id: str) -> Path:
    return run_dir / f"{run_id}-rerun-state.json"


def load_state(path: Path, run_id: str) -> RerunState:
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            attempts = [Attempt(**a) for a in raw.get("attempts", [])]
            return RerunState(run_id=raw.get("run_id", run_id), attempts=attempts)
        except (ValueError, TypeError):
            # Corrupt/incompatible state: start fresh rather than crash the hook.
            return RerunState(run_id=run_id)
    return RerunState(run_id=run_id)


def save_state(path: Path, state: RerunState) -> None:
    payload = {"run_id": state.run_id, "attempts": [asdict(a) for a in state.attempts]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
