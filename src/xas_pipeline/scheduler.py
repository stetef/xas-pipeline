"""SLURM and PBS: submission, cancellation, queries, and their syntax differences.

Every scheduler command goes through :func:`run_command`, the single seam tests
replace to run without a scheduler.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

# First integer on a line, optionally with a ``.server`` suffix: accepts both
# ``Submitted batch job 12345`` (sbatch) and ``12345.server`` (qsub).
JOB_ID_RE = re.compile(r"(?P<id>\d+)(?:\.[^\s]+)?")


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a scheduler command, capturing text output. Raises OSError if missing."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _query(cmd: list[str]) -> str | None:
    """stdout of an advisory query, or None on any failure (missing binary, non-zero exit)."""
    try:
        result = run_command(cmd)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def parse_elapsed_to_seconds(text: str) -> float | None:
    """``[DD-]HH:MM:SS[.ms]`` / ``MM:SS`` / ``SS`` -> seconds; None if unparseable."""
    text = (text or "").strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
    try:
        nums = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(nums) > 3:
        return None
    secs = 0.0
    for num in nums:
        secs = secs * 60 + num
    return days * 86400 + secs


def parse_job_id(stdout_text: str) -> str:
    """The job id in a submit command's output."""
    for line in stdout_text.splitlines():
        match = JOB_ID_RE.search(line.strip())
        if match:
            return match.group("id")
    raise ValueError(f"Unable to parse scheduler job id from output: {stdout_text!r}")


def check_executable(name: str) -> None:
    """Raise RuntimeError if *name* is not on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


class Scheduler(ABC):
    """One batch scheduler: commands, dependency syntax, queries."""

    name: str
    submit_command: str
    cancel_command: str
    template_subdir: str

    @abstractmethod
    def dependency_flag(self, kind: str, job_ids: list[str]) -> list[str]:
        """Submit arguments for an ``afterok`` / ``afterany`` dependency on *job_ids*."""

    @abstractmethod
    def debug_command(self, job_id: str) -> str:
        """A shell command for finding out why a job did not run or succeed."""

    def submit(
        self,
        script: Path,
        cwd: Path,
        *,
        afterok: Iterable[str] = (),
        afterany: Iterable[str] = (),
        extra: Iterable[str] = (),
    ) -> str:
        """Submit *script* from *cwd* (which becomes the job's submit dir); return its id.

        ``afterok``: run only if every parent succeeds. ``afterany``: run once
        every parent has finished. ``extra`` goes to the submit command verbatim.
        Raises RuntimeError when the submission is refused.
        """
        afterok, afterany = [str(j) for j in afterok], [str(j) for j in afterany]
        deps = (
            self.dependency_flag("afterok", afterok) if afterok
            else self.dependency_flag("afterany", afterany) if afterany
            else []
        )
        cmd = [self.submit_command, *deps, *map(str, extra), Path(script).name]
        result = run_command(cmd, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.submit_command} failed for {script} (cwd={cwd})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return parse_job_id(result.stdout)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job; True if the cancel command succeeded. Never raises."""
        try:
            return run_command([self.cancel_command, str(job_id)]).returncode == 0
        except OSError:
            return False

    def active_job_ids(self, job_ids: list[str]) -> list[str]:
        """The subset of *job_ids* still queued or running (empty on query failure).

        batch-jobs.log remembers every job ever submitted, including long-purged
        ones that would make a dependency unsatisfiable, so dependencies are built
        from this subset.
        """
        return []

    def job_states(self, job_ids: list[str]) -> dict[str, str]:
        """``{job id: RUNNING/PENDING/...}`` for jobs the scheduler still knows."""
        return {}

    def cpu_seconds(self, job_id: str) -> float | None:
        """CPU time a running job has used so far, or None if unknown."""
        return None


class SlurmScheduler(Scheduler):
    name = "slurm"
    submit_command = "sbatch"
    cancel_command = "scancel"
    template_subdir = "slurm-scripts"

    def dependency_flag(self, kind, job_ids):
        return [f"--dependency={kind}:{':'.join(job_ids)}"]

    def debug_command(self, job_id):
        return f"scontrol show job {job_id} && sacct -j {job_id} --format=JobID,State,ExitCode -n"

    def active_job_ids(self, job_ids):
        if not job_ids:
            return []
        # Unknown ids are simply absent (squeue may exit non-zero for them).
        out = _query(["squeue", "-h", "-o", "%i", "-j", ",".join(job_ids)])
        if out is None:
            return []
        listed = {line.strip().split(".")[0] for line in out.splitlines() if line.strip()}
        return [job_id for job_id in job_ids if job_id in listed]

    def job_states(self, job_ids):
        if not job_ids:
            return {}
        out = _query(["squeue", "-h", "-o", "%i %T", "-j", ",".join(job_ids)])
        states = {}
        for line in (out or "").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                states[fields[0].split(".")[0]] = fields[1].upper()
        return states

    def cpu_seconds(self, job_id):
        # sacct's TotalCPU stays 00:00:00 until a step ends, so a running job's
        # CPU comes from sstat: the cpu= total of every step (whichever step
        # mpirun's ranks land in), skipping .extern, whose figures are garbage.
        # sacct is the fallback for jobs sstat no longer reports.
        out = _query(["sstat", "-a", "-j", str(job_id), "-n", "-P", "-o", "JobID,TRESUsageInTot"])
        live = []
        for line in (out or "").splitlines():
            step, _, tres = line.partition("|")
            match = re.search(r"(?:^|,)cpu=([^,]+)", tres)
            if match and not step.strip().endswith(".extern"):
                live.append(parse_elapsed_to_seconds(match.group(1)))
        live = [s for s in live if s is not None]
        if live:
            return sum(live)
        out = _query(["sacct", "-j", str(job_id), "-o", "TotalCPU", "--noheader", "-P"])
        if out is None:
            return None
        seconds = [s for s in map(parse_elapsed_to_seconds, out.splitlines()) if s is not None]
        return max(seconds) if seconds else None


class PbsScheduler(Scheduler):
    name = "pbs"
    submit_command = "qsub"
    cancel_command = "qdel"
    template_subdir = "pbs-scripts"

    # PBS state letters -> the vocabulary the watchdog uses.
    _STATES: ClassVar[dict[str, str]] = {"R": "RUNNING", "Q": "PENDING", "H": "HELD", "E": "COMPLETING", "C": "COMPLETED"}

    def dependency_flag(self, kind, job_ids):
        return ["-W", f"depend={kind}:{':'.join(job_ids)}"]

    def debug_command(self, job_id):
        return (
            f"tracejob -n 100 {job_id} 2>&1 | "
            "grep -Ei 'deleted as result of dependency|Dependency on job|Exit_status|Obit'"
        )

    def _job_state(self, job_id) -> str | None:
        out = _query(["qstat", "-f", str(job_id)])
        match = re.search(r"job_state\s*=\s*(\S+)", out or "")
        return self._STATES.get(match.group(1), match.group(1).upper()) if match else None

    def active_job_ids(self, job_ids):
        # One query per id: qstat fails the whole call on one unknown id. Torque
        # keeps finished jobs listed as C for a while; those are not active.
        return [
            job_id for job_id in job_ids
            if self._job_state(job_id) not in (None, "COMPLETED")
        ]

    def job_states(self, job_ids):
        states = {}
        for job_id in job_ids:
            state = self._job_state(job_id)
            if state is not None:
                states[job_id.split(".")[0]] = state
        return states

    def cpu_seconds(self, job_id):
        out = _query(["qstat", "-f", str(job_id)])
        match = re.search(r"resources_used\.cput\s*=\s*(\S+)", out or "")
        return parse_elapsed_to_seconds(match.group(1)) if match else None


SCHEDULERS: dict[str, Scheduler] = {s.name: s for s in (SlurmScheduler(), PbsScheduler())}


def get_scheduler(name: str) -> Scheduler:
    try:
        return SCHEDULERS[name]
    except KeyError:
        raise ValueError(f"Unknown scheduler {name!r}; supported: {', '.join(SCHEDULERS)}") from None


def default_scheduler_name() -> str:
    """``$PIPELINE_SCHEDULER``, else ``pbs``."""
    name = os.environ.get("PIPELINE_SCHEDULER", "pbs").strip().lower()
    get_scheduler(name)
    return name


def add_scheduler_argument(parser) -> None:
    """``--scheduler``, defaulting to ``$PIPELINE_SCHEDULER``.

    The site ``.env`` is loaded first (without overriding the environment), so a
    ``PIPELINE_SCHEDULER`` set there applies on the login node too.
    """
    from . import config, resources

    config.load_env(resources.pipeline_env())
    parser.add_argument(
        "--scheduler", choices=sorted(SCHEDULERS), default=default_scheduler_name(),
        help="Batch scheduler (default: $PIPELINE_SCHEDULER, else pbs).",
    )
