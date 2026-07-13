"""Scheduler strategy: the single home for SLURM-vs-PBS differences.

Before the reorg this logic was duplicated across prepare-orca, prepare-corvus,
run-batch-pipeline (and diverging in submit-corvus-only): the submit command,
the template subdirectory, the dependency-flag syntax, job-id parsing, and the
debug command. It now lives here as a small strategy object so callers ask a
``Scheduler`` instead of branching on a name string.

Behavior is identical to the pre-reorg scripts (the characterization goldens
pin it): sbatch/qsub commands, ``--dependency=afterok:..`` vs ``-W depend=..``,
and the ``\\d+(\\.suffix)?`` job-id extraction that accepts both
``Submitted batch job 12345`` (Slurm) and ``12345.server`` (PBS).
"""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod

# First integer run on a line, optionally followed by a ``.server`` suffix.
JOB_ID_RE = re.compile(r"(?P<id>\d+)(?:\.[^\s]+)?")


def parse_job_id(stdout_text: str) -> str:
    """Extract the scheduler job id from a submit command's stdout."""
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = JOB_ID_RE.search(line)
        if match:
            return match.group("id")
    raise ValueError(f"Unable to parse scheduler job id from output: {stdout_text!r}")


def check_executable(name: str) -> None:
    """Raise if ``name`` is not resolvable on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


class Scheduler(ABC):
    """A batch scheduler backend (its command names + dependency/debug syntax)."""

    name: str
    submit_command: str
    template_subdir: str

    @abstractmethod
    def dependency_flag(self, kind: str, job_ids: list[str]) -> list[str]:
        """Submit-command args expressing ``kind`` (afterok/afterany) on job_ids."""

    @abstractmethod
    def debug_command(self, job_id: str) -> str:
        """A shell command to investigate why ``job_id`` did not run/succeed."""

    def parse_job_id(self, stdout_text: str) -> str:
        return parse_job_id(stdout_text)


class SlurmScheduler(Scheduler):
    name = "slurm"
    submit_command = "sbatch"
    template_subdir = "slurm-scripts"

    def dependency_flag(self, kind: str, job_ids: list[str]) -> list[str]:
        return [f"--dependency={kind}:{':'.join(job_ids)}"]

    def debug_command(self, job_id: str) -> str:
        return (
            f"scontrol show job {job_id} && "
            f"sacct -j {job_id} --format=JobID,State,ExitCode -n"
        )


class PbsScheduler(Scheduler):
    name = "pbs"
    submit_command = "qsub"
    template_subdir = "pbs-scripts"

    def dependency_flag(self, kind: str, job_ids: list[str]) -> list[str]:
        return ["-W", f"depend={kind}:{':'.join(job_ids)}"]

    def debug_command(self, job_id: str) -> str:
        return (
            f"tracejob -n 100 {job_id} 2>&1 | "
            "grep -Ei 'deleted as result of dependency|Dependency on job|Exit_status|Obit'"
        )


_REGISTRY: dict[str, Scheduler] = {s.name: s for s in (SlurmScheduler(), PbsScheduler())}

# Back-compat mappings used by the transitional top-level scripts.
NAMES = sorted(_REGISTRY)
SUBMIT_COMMAND = {name: sched.submit_command for name, sched in _REGISTRY.items()}
TEMPLATE_DIR = {name: sched.template_subdir for name, sched in _REGISTRY.items()}


def get_scheduler(name: str) -> Scheduler:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown scheduler {name!r}; supported: {', '.join(NAMES)}") from None


def default_scheduler_name() -> str:
    """Resolve the default scheduler from ``PIPELINE_SCHEDULER`` (else ``pbs``)."""
    import os

    scheduler = os.environ.get("PIPELINE_SCHEDULER", "pbs").strip().lower()
    if scheduler not in _REGISTRY:
        raise SystemExit(
            f"Invalid PIPELINE_SCHEDULER={scheduler!r}. Supported values: {', '.join(NAMES)}"
        )
    return scheduler
