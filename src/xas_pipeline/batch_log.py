"""``batch-jobs.log``: a batch's record of every job submitted and every outcome.

One tab-separated line per job event -- ``<job name> <status> [job_id=<id>]``,
where status is the *submission* result (SUBMITTED / SUBMIT_FAILED / SKIPPED /
REPLACED) -- plus outcome blocks the postprocess stages append
(``# --- ORCA outcomes <utc> ---`` then ``<job name> OK|FAILED [reason="..."]``).

It is the batch's memory across invocations: adding a mode to an existing batch
reads it back to find CORVUS jobs still running and postprocess jobs to replace.
Readers never raise; a missing log means "nothing known".
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .scheduler import Scheduler

LOG_NAME = "batch-jobs.log"

# CORVUS work is logged under a different prefix by each entry point that can
# submit it; a gate that matches only one of them would see nothing outstanding.
CORVUS_JOB_PREFIXES = ("corvus-", "submit-corvus-", "rerun-corvus-")
POSTPROCESS_JOB_PREFIX = "postprocess-"

_STATUS_NOTE = (
    "#\n"
    "# The 'status' column is the SUBMISSION result (SUBMITTED = the scheduler\n"
    "# accepted the job), NOT the computational outcome. Authoritative per-run\n"
    "# outcomes are appended below by the postprocess stage under\n"
    "# '# --- ORCA outcomes ... ---' / '# --- CORVUS outcomes ... ---', and are\n"
    "# detailed in orca-convergence-report.log and corvus-failed-ids.txt.\n"
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def find_batch_log(batch_root: Path) -> Path | None:
    candidate = Path(batch_root) / LOG_NAME
    return candidate if candidate.is_file() else None


def find_batch_root(path: Path) -> Path | None:
    """The nearest directory at or above *path* holding a ``batch-jobs.log``."""
    path = Path(path).resolve()
    for directory in (path, *path.parents):
        if (directory / LOG_NAME).is_file():
            return directory
    return None


def init(log_path: Path, sched: Scheduler) -> None:
    """Create the log with its help header, or mark a new invocation in an existing one."""
    log_path = Path(log_path)
    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# --- invocation {utc_now()} ---\n")
        return
    dep_debug = sched.debug_command("<JOB_ID>")
    if sched.name == "pbs":
        header = (
            "# Helpful PBS debug commands (replace <JOB_ID>)\n"
            f"# dependency + exit-status check (dep_debug): {dep_debug}\n"
            "# full tracejob history:\n"
            "#   tracejob -n 200 <JOB_ID>\n"
            "# full scheduler history (if enabled):\n"
            "#   qstat -x -f <JOB_ID>\n"
        )
    else:
        header = (
            "# Helpful Slurm debug commands (replace <JOB_ID>)\n"
            f"# dependency + state check (dep_debug): {dep_debug}\n"
            "# queue status:\n"
            "#   squeue -j <JOB_ID>\n"
            "# memory high-water mark (to confirm an OOM kill):\n"
            "#   sacct -j <JOB_ID> --format=JobID,State,ExitCode,MaxRSS,ReqMem -n\n"
        )
    log_path.write_text(header + _STATUS_NOTE + "\njob_name\tstatus\tjob_id\n", encoding="utf-8")


def append(log_path: Path, job_name: str, status: str, job_id: str | None = None) -> None:
    """Record one submission event."""
    line = f"{job_name}\t{status}" + (f"\tjob_id={job_id}" if job_id is not None else "") + "\n"
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(line)


def submit_and_log(
    sched: Scheduler, log_path: Path, job_name: str, script: Path, cwd: Path, **submit_kwargs
) -> str:
    """Submit *script* and record SUBMITTED (or SUBMIT_FAILED, then re-raise)."""
    try:
        job_id = sched.submit(script, cwd, **submit_kwargs)
    except Exception:
        append(log_path, job_name, "SUBMIT_FAILED")
        raise
    append(log_path, job_name, "SUBMITTED", job_id=job_id)
    return job_id


def records(log_path: Path) -> list[tuple[str, str, str | None]]:
    """``(job name, status, job id or None)`` for every event line, in order."""
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for raw in text.splitlines():
        fields = raw.strip().split("\t")
        if len(fields) < 2 or fields[0].startswith("#") or fields[0] == "job_name":
            continue
        job_id = fields[2][len("job_id="):].strip() if len(fields) > 2 and fields[2].startswith("job_id=") else None
        out.append((fields[0], fields[1], job_id or None))
    return out


def submitted_job_ids(log_path: Path, name_prefix: str | tuple[str, ...]) -> list[str]:
    """Ids logged SUBMITTED for job names starting with *name_prefix*, first-seen order."""
    seen: list[str] = []
    for name, status, job_id in records(log_path):
        if status == "SUBMITTED" and job_id and name.startswith(name_prefix) and job_id not in seen:
            seen.append(job_id)
    return seen


def latest_job_id(log_path: Path, job_names: Iterable[str]) -> str | None:
    """The most recent SUBMITTED id for any of exactly these job names."""
    names = set(job_names)
    latest = None
    for name, status, job_id in records(log_path):
        if status == "SUBMITTED" and job_id and name in names:
            latest = job_id
    return latest


def outstanding_corvus_job_ids(log_path: Path, sched: Scheduler, exclude: Iterable[str] = ()) -> list[str]:
    """CORVUS jobs this batch submitted earlier that are still queued or running."""
    exclude = set(exclude)
    known = [j for j in submitted_job_ids(log_path, CORVUS_JOB_PREFIXES) if j not in exclude]
    return sched.active_job_ids(known) if known else []


def pending_postprocess_job_ids(log_path: Path, sched: Scheduler) -> list[str]:
    """Postprocess jobs for this batch still queued or running (to be replaced).

    Running ones count too: cancelling a hung CORVUS job satisfies afterany, so a
    postprocess can start before the watchdog re-gates it, and must then be
    replaced rather than left to judge a batch whose rerun is still going.
    """
    known = submitted_job_ids(log_path, POSTPROCESS_JOB_PREFIX)
    return sched.active_job_ids(known) if known else []


def append_outcomes(log_path: Path, section: str, outcomes: Iterable[tuple[str, str, str | None]]) -> None:
    """Append an outcomes block of ``(job name, OK|FAILED, reason)``. Never raises."""
    outcomes = list(outcomes)
    if not outcomes:
        return
    lines = [f"# --- {section} {utc_now()} ---\n"]
    for job_name, status, reason in outcomes:
        if reason:
            lines.append(f'{job_name}\t{status}\treason="{" ".join(str(reason).split())}"\n')
        else:
            lines.append(f"{job_name}\t{status}\n")
    try:
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
    except OSError:
        pass
