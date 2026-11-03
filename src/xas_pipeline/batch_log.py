"""``batch-jobs.log``: a batch's record of every job submitted and every outcome.

One tab-separated line per job event -- ``<job name> <status> [job_id=<id>]``,
where status is the *submission* result (SUBMITTED / SUBMIT_FAILED / SKIPPED /
REPLACED), or PREEMPTED once the watchdog has resubmitted a preempted job -- plus outcome blocks the postprocess stages append
(``# --- ORCA outcomes <utc> ---`` then ``<job name> OK|FAILED [reason="..."]``).

It is the batch's memory across invocations: adding a mode to an existing batch
reads it back to find CORVUS jobs still running and postprocess jobs to replace.
Readers never raise; a missing log means "nothing known".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .scheduler import Scheduler

LOG_NAME = "batch-jobs.log"

# CORVUS work is logged under a different prefix by each entry point that can
# submit it; a gate that matches only one of them would see nothing outstanding.
CORVUS_JOB_PREFIXES = ("corvus-", "submit-corvus-", "rerun-corvus-")
POSTPROCESS_JOB_PREFIX = "postprocess-"
# The jobs that do the computing: what a batch's compute QOS applies to. The
# postprocess and watchdog keep the account's default QOS, so preemption never
# takes out the job that would resubmit the preempted ones.
COMPUTE_JOB_PREFIXES = ("orca-", *CORVUS_JOB_PREFIXES)

# Per-batch choices every later submission must honour, whichever command makes it.
SETTINGS_NAME = "batch-settings.json"

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


def read_settings(batch_root: Path) -> dict:
    """The batch's ``batch-settings.json`` (empty when absent or unreadable)."""
    try:
        data = json.loads((Path(batch_root) / SETTINGS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(batch_root: Path, **changes) -> None:
    """Merge *changes* into ``batch-settings.json``; a value of None removes the key."""
    settings = read_settings(batch_root)
    for key, value in changes.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value
    (Path(batch_root) / SETTINGS_NAME).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def compute_qos(batch_root: Path) -> str | None:
    """The QOS the batch's ORCA and CORVUS jobs are submitted with (None: the account default)."""
    return read_settings(batch_root).get("compute_qos") or None


def submit_and_log(
    sched: Scheduler, log_path: Path, job_name: str, script: Path, cwd: Path, **submit_kwargs
) -> str:
    """Submit *script* and record SUBMITTED (or SUBMIT_FAILED, then re-raise).

    A compute job (ORCA or CORVUS) gets the batch's compute QOS, so first
    submissions, reruns and watchdog resubmissions all land on the same QOS.
    """
    qos = compute_qos(Path(log_path).parent) if job_name.startswith(COMPUTE_JOB_PREFIXES) else None
    if qos:
        submit_kwargs["extra"] = (*submit_kwargs.get("extra", ()), *sched.qos_flag(qos))
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
