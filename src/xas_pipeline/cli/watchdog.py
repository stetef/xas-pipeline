"""xas-watchdog: find silently hung ORCA/CORVUS jobs in a batch and resubmit them.

An MPI job can wedge -- no progress, no exit -- and hold its allocation until the
walltime, days later. xas-rerun-orca only sees jobs that exit, so this companion
job (submitted by xas-run-batch unless --no-watchdog, or run by hand) polls every
``--interval`` seconds:

1. the batch's RUNNING ORCA and CORVUS jobs (from batch-jobs.log + the scheduler);
2. per job, how long its progress log has been silent and whether its CPU time
   still advances;
3. a job is **hung** when the log is silent past ``--idle-threshold`` *and* the
   CPU time is flat -- two signals, so a quiet-but-busy step survives;
4. a hung job is cancelled and resubmitted clean, excluding its node: ORCA with
   its follower (CORVUS, or the next ``--pre`` stage), CORVUS with its prior
   output archived. The batch postprocess is re-gated if one was pending. A
   per-structure ladder bounds this; then ``NEEDS_HUMAN``.

It exits when nothing in the batch is outstanding, or at start if another live
watchdog already watches the batch.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, layout, orchestrate, recovery, rerun_state
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import Scheduler, add_scheduler_argument, get_scheduler
from xas_pipeline.stages import orca_check

log = logging.getLogger(__name__)

# CPU time must advance by more than this between polls to count as alive.
CPU_EPS_S = 1.0


@dataclass
class Sample:
    """One poll's reading of one RUNNING job."""

    job_id: str
    log_mtime: float | None  # last write to its progress log
    cpu_seconds: float | None
    in_flight: bool  # its .timing says started, not finished


@dataclass
class _Obs:
    mtime: float | None
    cpu: float | None


class HangMonitor:
    """Scheduler-free hang detection across polls.

    :meth:`update` returns the jobs hung *this* poll: log silent for at least
    ``idle_threshold`` and CPU not advanced since the previous poll. A job is
    never flagged on the first poll it is seen (no CPU baseline yet).
    """

    def __init__(self, idle_threshold: float, cpu_eps: float = CPU_EPS_S) -> None:
        self.idle_threshold = idle_threshold
        self.cpu_eps = cpu_eps
        self._prev: dict[str, _Obs] = {}

    def update(self, now: float, samples: list[Sample]) -> list[str]:
        hung: list[str] = []
        seen: set[str] = set()
        for s in samples:
            seen.add(s.job_id)
            if not s.in_flight or s.log_mtime is None:
                self._prev.pop(s.job_id, None)
                continue
            prev = self._prev.get(s.job_id)
            cpu_alive = (
                prev is not None and s.cpu_seconds is not None and prev.cpu is not None
                and s.cpu_seconds > prev.cpu + self.cpu_eps
            )
            if prev is not None and now - s.log_mtime >= self.idle_threshold and not cpu_alive:
                hung.append(s.job_id)
            self._prev[s.job_id] = _Obs(s.log_mtime, s.cpu_seconds)
        for job_id in [j for j in self._prev if j not in seen]:
            self._prev.pop(job_id, None)
        return hung


# ── Where each job's run lives, and how to read its liveness ───────────────────


def index_run_dirs(batch_root: Path) -> dict[str, Path]:
    """``{run id: dir holding its job scripts}``, quarantine dirs excluded.

    A post-processed run's scripts live in ``working-<run id>/``; it is keyed by
    the run id all the same, so its rerun jobs resolve.
    """
    index: dict[str, Path] = {}
    for pattern in ("generated-*-orca.script", "generated-*-corvus-*-wrapper.script"):
        for script in sorted(batch_root.rglob(pattern)):
            rel = script.relative_to(batch_root).parts
            if layout.SKIP_DIR_NAMES.intersection(rel):
                continue
            name = script.parent.name
            index.setdefault(name.removeprefix("working-"), script.parent)
    return index


def _hostname(timing: Path | None) -> str | None:
    try:
        for line in timing.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("hostname="):
                return line.removeprefix("hostname=").strip() or None
    except (OSError, AttributeError):
        pass
    return None


def find_corvus_log(run_dir: Path, run_id: str) -> Path | None:
    """The CORVUS driver log ``corvus-<run_id>-<mode>.out`` (its mtime is liveness)."""
    matches = [p for p in run_dir.glob(f"corvus-{run_id}-*.out") if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def find_corvus_timing(run_dir: Path, run_id: str) -> Path | None:
    matches = sorted(run_dir.glob(f"{run_id}-corvus-*.timing"))
    return matches[0] if matches else None


def corvus_in_flight(run_dir: Path, run_id: str) -> bool:
    """The wrapper wrote ``start_epoch`` but no ``exit_code`` yet."""
    timing = find_corvus_timing(run_dir, run_id)
    try:
        return timing is not None and "exit_code=" not in timing.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _mtime(path: Path | None) -> float | None:
    try:
        return path.stat().st_mtime if path is not None else None
    except OSError:
        return None


@dataclass
class Running:
    job_id: str
    run_id: str
    run_dir: Path
    stage: str  # "orca" or "corvus"


def running_jobs(sched: Scheduler, log_path: Path, run_dirs: dict[str, Path]) -> list[Running]:
    """The batch's RUNNING ORCA/CORVUS jobs, each resolved to its run dir."""
    names = {}
    for name, status, job_id in batch_log.records(log_path):
        if status == "SUBMITTED" and job_id and (name.startswith("orca-") or recovery.is_corvus_job(name)):
            names[job_id] = name
    if not names:
        return []
    out = []
    for job_id, state in sched.job_states(list(names)).items():
        run_id = recovery.run_id_from_job_name(names.get(job_id, ""))
        if state == "RUNNING" and run_id in run_dirs:
            stage = "orca" if names[job_id].startswith("orca-") else "corvus"
            out.append(Running(job_id, run_id, run_dirs[run_id], stage))
    return out


def sample(sched: Scheduler, job: Running) -> Sample:
    if job.stage == "orca":
        progress, in_flight = orca_check.find_orca_log(job.run_dir), orca_check.orca_run_is_in_flight(job.run_dir)
    else:
        progress, in_flight = find_corvus_log(job.run_dir, job.run_id), corvus_in_flight(job.run_dir, job.run_id)
    return Sample(job.job_id, _mtime(progress), sched.cpu_seconds(job.job_id), in_flight)


# ── Recovery ──────────────────────────────────────────────────────────────────


def recover(sched: Scheduler, batch_root: Path, job: Running, *, max_reruns: int, exclude_node: bool) -> str:
    """Cancel and resubmit one hung job; returns the outcome."""
    log_path = batch_root / batch_log.LOG_NAME
    ladder = "hang" if job.stage == "orca" else "corvus-hang"
    label = f"{job.stage}-{job.run_id}"
    state_file = rerun_state.state_path(job.run_dir, job.run_id, kind=ladder)
    state = rerun_state.load_state(state_file, job.run_id)
    if state.is_terminal:
        return "skipped-terminal"
    timing = (orca_check.find_orca_timing(job.run_dir) if job.stage == "orca"
              else find_corvus_timing(job.run_dir, job.run_id))
    node = _hostname(timing)
    log.warning("[watchdog] [%s] %s job %s hung on %s; cancelling", job.run_id, job.stage, job.job_id,
                node or "an unknown node")

    if not sched.cancel(job.job_id):
        # Resubmitting a job that may still be running would double it.
        recovery.escalate(state_file, state, log_path, label,
                          f"hung {job.stage} job {job.job_id} could not be cancelled", "watchdog escalation")
        return "escalated"
    if job.stage == "orca":
        recovery.cancel_stale_corvus(sched, log_path, job.run_id)

    attempt = state.next_attempt
    if attempt > max_reruns:
        recovery.escalate(state_file, state, log_path, label,
                          f"hang ladder exhausted after {len(state.attempts)} resubmit(s) "
                          f"(last hung on {node or 'an unknown node'})", "watchdog escalation")
        orchestrate.regate_postprocess(sched, batch_root)
        return "escalated"

    extra = (f"--exclude={node}",) if exclude_node and sched.name == "slurm" and node else ()
    record = rerun_state.Attempt(attempt=attempt, kind=ladder, remedy="clean-resubmit",
                                 utc=batch_log.utc_now(), note=f"hung on {node}" if node else None)
    try:
        if job.stage == "orca":
            record.orca_job_id, record.corvus_job_id = recovery.resubmit_orca(
                sched, job.run_dir, job.run_id, log_path, label=f"hang-rerun{attempt}", extra=extra
            )
        else:
            # A clean restart, not a resume into the half-built tree that hung.
            recovery.archive_corvus_outputs(job.run_dir, None, job.run_id, f"rerun-hang{attempt}")
            wrapper = jobscripts.corvus_wrapper(sched, job.run_dir, job.run_id)
            record.corvus_job_id = batch_log.submit_and_log(
                sched, log_path, f"corvus-xas-hang-rerun{attempt}-{job.run_id}", wrapper, job.run_dir, extra=extra
            )
    except (RuntimeError, ValueError, OSError) as exc:
        state.attempts.append(record)
        recovery.escalate(state_file, state, log_path, label,
                          f"resubmission after a hang failed: {exc}", "watchdog escalation")
        return "escalated"
    state.attempts.append(record)
    rerun_state.save_state(state_file, state)
    orchestrate.regate_postprocess(sched, batch_root)
    return "resubmitted"


# ── Loop ──────────────────────────────────────────────────────────────────────


def batch_has_active_jobs(sched: Scheduler, log_path: Path) -> bool:
    ids = batch_log.submitted_job_ids(
        log_path, ("orca-", *batch_log.CORVUS_JOB_PREFIXES, batch_log.POSTPROCESS_JOB_PREFIX)
    )
    return bool(sched.active_job_ids(ids)) if ids else False


def other_live_watchdog(sched: Scheduler, batch_root: Path) -> str | None:
    """A live watchdog job for this batch that this one should defer to, if any.

    Run by hand, any live watchdog job wins. As a job, only an earlier-submitted
    one does: two live at once must leave exactly one watching, not zero.
    """
    own = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID", "").split(".")[0]
    ids = batch_log.submitted_job_ids(batch_root / batch_log.LOG_NAME, f"watchdog-{batch_root.name}")
    others = [j for j in sched.active_job_ids(ids) if j != own]
    if own and own.isdigit():
        others = [j for j in others if j.isdigit() and int(j) < int(own)]
    return others[-1] if others else None


def sweep_once(sched: Scheduler, batch_root: Path, monitor: HangMonitor, run_dirs: dict[str, Path],
               handled: set[str], *, max_reruns: int, exclude_node: bool, now: float | None = None) -> list[str]:
    """One poll; returns the run ids acted on. *handled* remembers recovered job ids."""
    jobs = {job.job_id: job for job in running_jobs(sched, batch_root / batch_log.LOG_NAME, run_dirs)
            if job.job_id not in handled}
    hung = monitor.update(time.time() if now is None else now, [sample(sched, job) for job in jobs.values()])
    acted = []
    for job_id in hung:
        job = jobs[job_id]
        handled.add(job_id)
        outcome = recover(sched, batch_root, job, max_reruns=max_reruns, exclude_node=exclude_node)
        acted.append(f"{job.run_id} ({job.stage}: {outcome})")
    return acted


def watch(batch_root: Path, scheduler: str, *, interval: float, idle_threshold: float, max_reruns: int,
          exclude_node: bool = True, once: bool = False) -> int:
    sched = get_scheduler(scheduler)
    log_path = batch_root / batch_log.LOG_NAME
    other = other_live_watchdog(sched, batch_root)
    if other:
        log.info("[watchdog] job %s already watches %s; exiting", other, batch_root)
        return 0
    monitor, handled = HangMonitor(idle_threshold), set()
    log.info("[watchdog] watching %s (interval %gs, idle threshold %gs, max reruns %d)",
             batch_root, interval, idle_threshold, max_reruns)
    while True:
        if not batch_has_active_jobs(sched, log_path):
            log.info("[watchdog] nothing outstanding in the batch; exiting")
            return 0
        try:
            acted = sweep_once(sched, batch_root, monitor, index_run_dirs(batch_root), handled,
                               max_reruns=max_reruns, exclude_node=exclude_node)
        except Exception as exc:
            log.warning("[watchdog] sweep failed (%r); continuing", exc)
            acted = []
        if acted:
            log.info("[watchdog] acted on: %s", ", ".join(acted))
        if once:
            return 0
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a batch for hung ORCA/CORVUS jobs and resubmit them.")
    parser.add_argument("batch_root", type=Path, help="The batch root (holds batch-jobs.log)")
    add_scheduler_argument(parser)
    parser.add_argument("--interval", type=float, default=jobscripts.WATCHDOG_INTERVAL,
                        help="Seconds between polls (default: %(default)s)")
    parser.add_argument("--idle-threshold", type=float, default=jobscripts.WATCHDOG_IDLE_THRESHOLD,
                        help="Silent-log + flat-CPU seconds before a job counts as hung (default: %(default)s)")
    parser.add_argument("--max-reruns", type=int, default=jobscripts.WATCHDOG_MAX_RERUNS,
                        help="Hang resubmits per structure before escalating (default: %(default)s)")
    parser.add_argument("--no-exclude-node", dest="exclude_node", action="store_false",
                        help="Do not exclude the hung node when resubmitting")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    batch_root = args.batch_root.expanduser().resolve()
    if not (batch_root / batch_log.LOG_NAME).is_file():
        raise SystemExit(f"ERROR: no {batch_log.LOG_NAME} in {batch_root}")
    return watch(batch_root, args.scheduler, interval=args.interval, idle_threshold=args.idle_threshold,
                 max_reruns=args.max_reruns, exclude_node=args.exclude_node, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
