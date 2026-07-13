"""Recording authoritative per-run outcomes in ``batch-jobs.log``.

The orchestrator writes one line per job at *submission* time (status
``SUBMITTED`` = the scheduler accepted it). That is not the computational
outcome. After the batch finishes, the postprocess stages (ORCA convergence
check and FEFF output processing) call :func:`append_outcomes` here to append an
authoritative outcomes section so the log records *why* each run passed or
failed, not just that it was submitted.

Stdlib-only by design so the standalone postprocess entry points can use it
without pulling in the rest of the pipeline. (Formerly the top-level
``pipeline_batch_log.py``; moved into the package during the reorg.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def find_batch_log(parent_dir: Path) -> Path | None:
    """Return the batch-jobs.log for a batch output root, if it exists."""
    candidate = Path(parent_dir) / "batch-jobs.log"
    return candidate if candidate.is_file() else None


def _sanitize(text: str) -> str:
    """Flatten a reason to a single, tab-safe line for the tab-delimited log."""
    return " ".join(str(text).split())


def append_outcomes(
    batch_log: Path,
    section: str,
    outcomes: Iterable[tuple[str, str, str | None]],
) -> None:
    """Append an authoritative outcomes block to batch-jobs.log.

    ``outcomes`` is an iterable of ``(job_name, status, reason)`` tuples where
    ``status`` is e.g. ``OK`` / ``FAILED`` and ``reason`` is an optional
    human-readable explanation (kept for FAILED, omitted for OK). ``section`` is
    a short label (e.g. ``"ORCA outcomes"``) used in the block header comment.

    Best-effort: never raises if the log is missing/unwritable -- recording an
    outcome must not crash the postprocess job.
    """
    outcomes = list(outcomes)
    if not outcomes:
        return
    try:
        stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        lines = [f"# --- {section} {stamp} ---\n"]
        for job_name, status, reason in outcomes:
            if reason:
                lines.append(f"{job_name}\t{status}\treason=\"{_sanitize(reason)}\"\n")
            else:
                lines.append(f"{job_name}\t{status}\n")
        with Path(batch_log).open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
    except OSError:
        pass
