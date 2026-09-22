#!/usr/bin/env python3
"""Watch a running batch for silently-hung ORCA/CORVUS jobs and auto-resubmit them.

MPI jobs can wedge: a rank stalls on an intra-node collective or a shared scratch
I/O, the whole job stops making progress, but nothing exits -- the scheduler
keeps billing the allocation as RUNNING until it hits the wall (days later). This
bites both stages of the pipeline: ORCA 6 optimizations, and the CORVUS/FEFF
stage (which shells out to many ``mpirun -n N`` FEFF modules in turn, any one of
which can deadlock between steps). The per-structure auto-rerun
(:mod:`xas_pipeline.cli.rerun_orca`) cannot help: it only fires *after* a job
exits non-zero, and a hang that finally reaches walltime reads as a generic
crash, which is not auto-remediable.

This watchdog closes that gap. It runs as a lightweight companion job alongside
the batch (submitted automatically by ``xas-run-batch`` unless ``--no-watchdog``)
or standalone (``xas-watchdog <batch_root>``), and every ``--interval`` seconds:

  1. asks the scheduler which of the batch's ORCA *and* CORVUS jobs are RUNNING;
  2. for each, measures how long its progress log (ORCA ``.log`` /
     ``corvus-<id>-<mode>.out``) has been silent and whether its accumulated CPU
     time is still advancing;
  3. declares a job **hung** when the log has been silent longer than
     ``--idle-threshold`` *and* CPU time is not advancing (two signals, so a
     legitimately quiet-but-busy step -- a long AnFreq CPHF or a slow FEFF path
     expansion, say -- is not killed);
  4. on a hang, cancels the job, resubmits a clean restart excluding the node it
     hung on (an ORCA hang also cancels+resubmits its dependent CORVUS job; a
     CORVUS hang resubmits the wrapper alone -- its Hessian already exists), and
     re-gates the batch postprocess -- the same DAG-preserving recovery a human
     would do -- bounded by a persisted per-structure counter (a separate ladder
     per stage; escalating to ``NEEDS_HUMAN`` when exhausted).

It exits once the batch has no ORCA/CORVUS/postprocess jobs left outstanding.

Detection (:class:`HangMonitor`) is pure and unit-tested; the scheduler queries,
filesystem reads, and resubmission live in the thin functions around it. ORCA and
CORVUS jobs share one monitor (their job ids are disjoint) but recover through
stage-specific paths (:func:`recover_hung`, :func:`recover_hung_corvus`).
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

from xas_pipeline import batch_log as _batch_log
from xas_pipeline import orchestrate as bp
from xas_pipeline import rerun_state, scheduler as _sched
from xas_pipeline.batch_log import append_outcomes
from xas_pipeline.cli.rerun_orca import _find_stale_corvus_job_id
from xas_pipeline.cli.rerun_corvus import archive_mode_artifacts
from xas_pipeline.stages import orca_check as oc

# Defaults chosen for the caopt-anfreq workload (tetef01, 2026-09-21): a 60-min
# silent-log + flat-CPU threshold clears any legitimately quiet AnFreq/CPHF
# stretch, polled every 10 min; two hang-resubmits per structure before a human
# is asked.
DEFAULT_INTERVAL_S = 600
DEFAULT_IDLE_THRESHOLD_S = 3600
DEFAULT_MAX_RERUNS = 2
# CPU-time must advance by more than this (seconds) between polls to count as
# "still alive". Small, to tolerate sacct's coarse accounting granularity.
CPU_EPS_S = 1.0

_RERUN_DECORATION_RE = re.compile(r"^((hang|stage)-)?rerun\d+-")


# --------------------------------------------------------------------------- #
# Pure hang detection
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One poll's reading for a single RUNNING ORCA job."""

    job_id: str
    log_mtime: float | None  # epoch seconds of the ORCA log's last write, or None
    cpu_seconds: float | None  # accumulated CPU time, or None if unreadable
    in_flight: bool  # timing sidecar says started-but-not-finished


@dataclass
class _Obs:
    mtime: float | None
    cpu: float | None


class HangMonitor:
    """Stateful, scheduler-free hang detector.

    Fed a list of :class:`Sample` per poll via :meth:`update`, it returns the job
    ids that are hung *this* poll. A job is hung when its log has been silent for
    at least ``idle_threshold`` seconds and its CPU time has not advanced since
    the previous poll. A job is never flagged on the very first poll it is seen
    (no prior CPU baseline to rule out "quiet but busy"), so a stale mtime at
    startup costs at most one extra interval rather than a wrongful kill.
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
            # Not in flight (finished/never-started) or no log yet: forget it, so
            # a resubmitted job under the same id starts from a clean baseline.
            if not s.in_flight or s.log_mtime is None:
                self._prev.pop(s.job_id, None)
                continue
            prev = self._prev.get(s.job_id)
            age = now - s.log_mtime
            cpu_alive = (
                prev is not None
                and s.cpu_seconds is not None
                and prev.cpu is not None
                and s.cpu_seconds > prev.cpu + self.cpu_eps
            )
            is_hung = age >= self.idle_threshold and not cpu_alive and prev is not None
            self._prev[s.job_id] = _Obs(s.log_mtime, s.cpu_seconds)
            if is_hung:
                hung.append(s.job_id)
        for jid in [j for j in self._prev if j not in seen]:
            self._prev.pop(jid, None)
        return hung


# --------------------------------------------------------------------------- #
# Batch discovery
# --------------------------------------------------------------------------- #
def run_id_from_job_name(job_name: str) -> str:
    """``orca-[hang-|stage-]rerunN-<run_id>`` / ``orca-<run_id>`` -> ``<run_id>``."""
    name = job_name.strip()
    if name.startswith("orca-"):
        name = name[len("orca-"):]
    return _RERUN_DECORATION_RE.sub("", name)


def resolve_run_dir(job_name: str, run_dirs: dict[str, Path]) -> Path | None:
    """Find the run dir an ``orca-*`` job name belongs to.

    Tries the parsed run id first (the fast path for the pipeline's own naming),
    then falls back to the longest indexed run-dir name that appears anywhere in
    the job name. The fallback tolerates any rerun decoration -- including
    manually-submitted jobs whose name carries a non-standard suffix -- so the
    watchdog still guards a job however it was (re)submitted.
    """
    run_id = run_id_from_job_name(job_name)
    if run_id in run_dirs:
        return run_dirs[run_id]
    candidates = [name for name in run_dirs if name in job_name]
    if not candidates:
        return None
    return run_dirs[max(candidates, key=len)]


def index_run_dirs(batch_root: Path) -> dict[str, Path]:
    """Map each run-dir name to its path, found by its generated job scripts.

    Keyed by ``run_dir.name`` (== the run id), so a job name recovered from
    batch-jobs.log resolves to the directory holding its ``.in``, ``.log`` and
    generated scripts regardless of how deep the batch nests run dirs. Indexed by
    the ORCA job script *and* the CORVUS wrapper, so ORCA-free modes (the
    ``asis-spring`` interp-raw route, which has only a CORVUS wrapper) are found
    too.
    """
    index: dict[str, Path] = {}
    for pattern in ("generated-*-orca.script", "generated-*-corvus-*-wrapper.script"):
        for script in batch_root.rglob(pattern):
            index.setdefault(script.parent.name, script.parent)
    return index


def _submitted_job_names(batch_log: Path, prefixes: tuple[str, ...]) -> dict[str, str]:
    """``{job_id: job_name}`` for every SUBMITTED line whose name starts a prefix."""
    names: dict[str, str] = {}
    try:
        text = Path(batch_log).read_text(encoding="utf-8")
    except OSError:
        return names
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3 or fields[1] != "SUBMITTED" or not fields[0].startswith(prefixes):
            continue
        if fields[2].startswith("job_id="):
            names[fields[2][len("job_id="):].strip()] = fields[0]
    return names


def orca_job_names(batch_log: Path) -> dict[str, str]:
    """``{job_id: job_name}`` for every ``orca-*`` SUBMITTED line in the log."""
    return _submitted_job_names(batch_log, ("orca-",))


def corvus_job_names(batch_log: Path) -> dict[str, str]:
    """``{job_id: job_name}`` for every CORVUS SUBMITTED line in the log.

    CORVUS work is logged under several prefixes depending on the entry point
    that queued it (see :data:`orchestrate.CORVUS_JOB_PREFIX`); all are watched.
    """
    return _submitted_job_names(batch_log, bp.CORVUS_JOB_PREFIX)


def _hostname_from_timing_file(timing: Path | None) -> str | None:
    """The ``hostname=`` a job wrote into its ``.timing`` sidecar, or None."""
    if timing is None:
        return None
    try:
        for line in timing.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("hostname="):
                return line[len("hostname="):].strip() or None
    except OSError:
        return None
    return None


def _hostname_from_timing(run_dir: Path) -> str | None:
    """The compute node a run's ORCA job landed on (``hostname=`` in .timing)."""
    return _hostname_from_timing_file(oc.find_orca_timing(run_dir))


# --- CORVUS-side equivalents of orca_check's log/timing/in-flight helpers ---
def find_corvus_log(run_dir: Path) -> Path | None:
    """The CORVUS/FEFF progress log for this run (``corvus-<run_id>-<mode>.out``).

    This is the top-level driver log CORVUS appends to as it steps through the
    FEFF modules -- not the per-module FEFF ``.out`` files nested under
    ``Corvus*/``. Its mtime is the CORVUS stage's liveness signal.
    """
    matches = [p for p in run_dir.glob(f"corvus-{run_dir.name}-*.out") if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def find_corvus_timing(run_dir: Path) -> Path | None:
    """The CORVUS ``.timing`` sidecar (``<run_id>-corvus-<mode>.timing``)."""
    matches = sorted(run_dir.glob(f"{run_dir.name}-corvus-*.timing"))
    return matches[0] if matches else None


def corvus_run_is_in_flight(run_dir: Path) -> bool:
    """True when this run's CORVUS job started but has not finished.

    Mirrors :func:`orca_check.orca_run_is_in_flight`: the wrapper writes
    ``start_epoch`` up front and appends ``exit_code=`` only once CORVUS returns,
    so a timing file lacking an exit code means the stage is still going (or
    wedged). Absence of a timing file is not treated as in-flight.
    """
    timing = find_corvus_timing(run_dir)
    if timing is None:
        return False
    try:
        return "exit_code=" not in timing.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def sample_running_orca(
    sched: _sched.Scheduler, batch_log: Path, run_dirs: dict[str, Path]
) -> tuple[list[Sample], dict[str, Path]]:
    """Build a :class:`Sample` for every RUNNING ORCA job in the batch.

    Returns the samples plus a ``{job_id: run_dir}`` map for the recovery step.
    """
    names = orca_job_names(batch_log)
    if not names:
        return [], {}
    states = sched.job_states(list(names))
    samples: list[Sample] = []
    jid_dir: dict[str, Path] = {}
    for jid, state in states.items():
        if state != "RUNNING":
            continue
        run_dir = resolve_run_dir(names.get(jid, ""), run_dirs)
        if run_dir is None:
            continue
        jid_dir[jid] = run_dir
        log = oc.find_orca_log(run_dir)
        try:
            mtime = log.stat().st_mtime if log is not None else None
        except OSError:
            mtime = None
        samples.append(
            Sample(
                job_id=jid,
                log_mtime=mtime,
                cpu_seconds=sched.cpu_seconds(jid),
                in_flight=oc.orca_run_is_in_flight(run_dir),
            )
        )
    return samples, jid_dir


def sample_running_corvus(
    sched: _sched.Scheduler, batch_log: Path, run_dirs: dict[str, Path]
) -> tuple[list[Sample], dict[str, Path]]:
    """Build a :class:`Sample` for every RUNNING CORVUS job in the batch.

    The CORVUS analogue of :func:`sample_running_orca`: liveness comes from the
    ``corvus-<id>-<mode>.out`` driver log's mtime and the CORVUS ``.timing``
    in-flight flag. Returns the samples plus a ``{job_id: run_dir}`` map.
    """
    names = corvus_job_names(batch_log)
    if not names:
        return [], {}
    states = sched.job_states(list(names))
    samples: list[Sample] = []
    jid_dir: dict[str, Path] = {}
    for jid, state in states.items():
        if state != "RUNNING":
            continue
        run_dir = resolve_run_dir(names.get(jid, ""), run_dirs)
        if run_dir is None:
            continue
        jid_dir[jid] = run_dir
        log = find_corvus_log(run_dir)
        try:
            mtime = log.stat().st_mtime if log is not None else None
        except OSError:
            mtime = None
        samples.append(
            Sample(
                job_id=jid,
                log_mtime=mtime,
                cpu_seconds=sched.cpu_seconds(jid),
                in_flight=corvus_run_is_in_flight(run_dir),
            )
        )
    return samples, jid_dir


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def _find_script(run_dir: Path, run_id: str, exact: str, glob: str) -> Path | None:
    candidate = run_dir / exact
    if candidate.is_file():
        return candidate
    matches = sorted(run_dir.glob(glob))
    return matches[0] if matches else None


def _regate_postprocess(batch_root: Path, batch_log: Path, scheduler: str) -> None:
    """Replace any pending postprocess so it waits on the resubmitted CORVUS job.

    Same move the orchestrator makes when it adds work to a batch: cancel the
    now under-gated postprocess job(s) and submit the existing postprocess script
    afresh, ``afterany`` on the batch's currently-outstanding CORVUS jobs (which
    now include the resubmitted one and exclude the cancelled stale one).
    """
    pending = bp.pending_postprocess_job_ids(batch_log, scheduler)
    script = batch_root / f"generated-postprocess-{batch_root.name}.script"
    if not script.is_file():
        if pending:
            print(f"[watchdog] warning: postprocess script {script.name} missing; "
                  f"left {len(pending)} pending postprocess job(s) as-is")
        return
    for stale in pending:
        bp._cancel_job(stale, scheduler)
        _batch_log_append(batch_log, f"postprocess-{batch_root.name}", "REPLACED", job_id=stale)
    outstanding = bp.outstanding_corvus_job_ids(batch_log, scheduler)
    try:
        new_id = bp._submit_job(
            script, cwd=batch_root, scheduler=scheduler, depend_afterany=outstanding
        )
    except Exception as exc:  # noqa: BLE001 - best-effort re-gate; log and move on
        print(f"[watchdog] warning: could not resubmit postprocess: {exc}")
        return
    _batch_log_append(batch_log, f"postprocess-{batch_root.name}", "SUBMITTED", job_id=new_id)
    print(f"[watchdog] re-gated postprocess -> {new_id} (afterany on "
          f"{len(outstanding)} CORVUS job(s))")


def _batch_log_append(batch_log: Path, name: str, status: str, job_id: str | None = None) -> None:
    if batch_log.is_file() or job_id is not None:
        bp._append_batch_job_log(batch_log, name, status, job_id=job_id)


def _escalate_hang(
    run_dir: Path, run_id: str, state: rerun_state.RerunState, state_file: Path,
    batch_log: Path, reason: str, *, job_label: str | None = None,
) -> None:
    state.resolution = rerun_state.RESOLUTION_NEEDS_HUMAN
    rerun_state.save_state(state_file, state)
    if batch_log.is_file():
        append_outcomes(batch_log, "watchdog escalation",
                        [(job_label or f"orca-{run_id}", "NEEDS_HUMAN", reason)])
    print(f"[watchdog] [{run_id}] NEEDS_HUMAN: {reason}")


def recover_hung(
    batch_root: Path,
    run_dir: Path,
    hung_job_id: str,
    scheduler: str,
    *,
    max_reruns: int,
    exclude_hung_node: bool,
    corvus_mode: str = "xas",
) -> str:
    """Cancel and clean-resubmit one hung ORCA run, keeping the batch DAG intact.

    Returns a short outcome string ('resubmitted' / 'escalated' / 'skipped-...').
    """
    run_id = run_dir.name
    batch_log = batch_root / "batch-jobs.log"
    state_file = rerun_state.state_path(run_dir, run_id, kind="hang")
    state = rerun_state.load_state(state_file, run_id)

    if state.is_terminal:
        return "skipped-terminal"

    attempt = state.next_attempt
    node = _hostname_from_timing(run_dir)

    # The dependent CORVUS job was queued afterok on this ORCA; it can never run
    # now. Cancel it whether we resubmit or escalate, so it does not linger as a
    # DependencyNeverSatisfied zombie blocking the postprocess.
    stale_corvus = None
    if batch_log.is_file():
        stale_corvus = _find_stale_corvus_job_id(
            batch_log.read_text(encoding="utf-8", errors="replace"), run_id
        )

    # Always stop the wedged job itself: it is making no progress and only holds
    # the allocation. (scancel triggers the wrapper's TERM trap, which exits
    # before the per-structure rerun hook, so the two paths never race.)
    print(f"[watchdog] [{run_id}] hung on {node or 'unknown node'}; cancelling ORCA {hung_job_id}"
          + (f" + CORVUS {stale_corvus}" if stale_corvus else ""))
    bp._cancel_job(hung_job_id, scheduler)
    if stale_corvus:
        bp._cancel_job(stale_corvus, scheduler)
        _batch_log_append(batch_log, f"corvus-{run_id}", "CANCELLED")

    if attempt > max_reruns:
        _escalate_hang(run_dir, run_id, state, state_file, batch_log,
                       f"hang-resubmit ladder exhausted after {len(state.attempts)} attempt(s) "
                       f"(last hung on {node or 'unknown node'})")
        _regate_postprocess(batch_root, batch_log, scheduler)
        return "escalated"

    orca_script = _find_script(run_dir, run_id, f"generated-{run_id}-orca.script",
                               "generated-*-orca.script")
    if orca_script is None:
        _escalate_hang(run_dir, run_id, state, state_file, batch_log,
                       "no generated ORCA job script found to resubmit")
        return "skipped-no-script"

    extra: list[str] = []
    if exclude_hung_node and scheduler == "slurm" and node:
        extra = [f"--exclude={node}"]

    new_orca = bp._submit_job(orca_script, cwd=run_dir, scheduler=scheduler,
                              extra_submit_args=extra)
    _batch_log_append(batch_log, f"orca-hang-rerun{attempt}-{run_id}", "SUBMITTED", job_id=new_orca)
    print(f"[watchdog] [{run_id}] resubmitted ORCA -> {new_orca}"
          + (f" ({' '.join(extra)})" if extra else ""))

    new_corvus = None
    corvus_script = _find_script(
        run_dir, run_id, f"generated-{run_id}-corvus-{corvus_mode}-wrapper.script",
        "generated-*-corvus-*-wrapper.script",
    )
    if corvus_script is not None:
        new_corvus = bp._submit_job(corvus_script, cwd=run_dir, scheduler=scheduler,
                                    depend_afterok=[new_orca])
        _batch_log_append(batch_log, f"corvus-{corvus_mode}-hang-rerun{attempt}-{run_id}",
                          "SUBMITTED", job_id=new_corvus)
        print(f"[watchdog] [{run_id}] resubmitted CORVUS -> {new_corvus} (afterok:{new_orca})")
    else:
        print(f"[watchdog] [{run_id}] warning: no CORVUS wrapper found; ORCA resubmitted alone")

    _regate_postprocess(batch_root, batch_log, scheduler)

    state.attempts.append(
        rerun_state.Attempt(
            attempt=attempt, kind="hang", remedy="clean-resubmit",
            utc=bp._utc_now_iso(), orca_job_id=new_orca, corvus_job_id=new_corvus,
            note=f"hung on {node}" if node else None,
        )
    )
    rerun_state.save_state(state_file, state)
    return "resubmitted"


def recover_hung_corvus(
    batch_root: Path,
    run_dir: Path,
    hung_job_id: str,
    scheduler: str,
    *,
    max_reruns: int,
    exclude_hung_node: bool,
    corvus_mode: str = "xas",
) -> str:
    """Cancel and clean-resubmit one hung CORVUS run, keeping the batch DAG intact.

    Simpler than the ORCA path: the CORVUS wrapper reruns prepare-corvus + CORVUS
    inline and needs no upstream job (its Hessian already exists on disk, from
    ORCA or the spring interpolation), so there is no dependent job to cancel and
    the resubmission carries no dependency. Bounded by the run's ``corvus-hang``
    ladder; re-gates the postprocess so it waits on the fresh CORVUS job.

    Returns 'resubmitted' / 'escalated' / 'skipped-...'.
    """
    run_id = run_dir.name
    batch_log = batch_root / "batch-jobs.log"
    state_file = rerun_state.state_path(run_dir, run_id, kind="corvus-hang")
    state = rerun_state.load_state(state_file, run_id)

    if state.is_terminal:
        return "skipped-terminal"

    attempt = state.next_attempt
    node = _hostname_from_timing_file(find_corvus_timing(run_dir))

    print(f"[watchdog] [{run_id}] CORVUS hung on {node or 'unknown node'}; "
          f"cancelling {hung_job_id}")
    bp._cancel_job(hung_job_id, scheduler)
    _batch_log_append(batch_log, f"corvus-{corvus_mode}-{run_id}", "CANCELLED")

    if attempt > max_reruns:
        _escalate_hang(run_dir, run_id, state, state_file, batch_log,
                       f"CORVUS hang-resubmit ladder exhausted after {len(state.attempts)} "
                       f"attempt(s) (last hung on {node or 'unknown node'})",
                       job_label=f"corvus-{run_id}")
        _regate_postprocess(batch_root, batch_log, scheduler)
        return "escalated"

    corvus_script = _find_script(
        run_dir, run_id, f"generated-{run_id}-corvus-{corvus_mode}-wrapper.script",
        "generated-*-corvus-*-wrapper.script",
    )
    if corvus_script is None:
        _escalate_hang(run_dir, run_id, state, state_file, batch_log,
                       "no generated CORVUS wrapper script found to resubmit",
                       job_label=f"corvus-{run_id}")
        _regate_postprocess(batch_root, batch_log, scheduler)
        return "skipped-no-script"

    # Archive the wedged run's partial CORVUS output aside so the resubmission is
    # a genuine clean restart, not a resume into the same half-built (and possibly
    # re-wedging) Corvus3_cfavg_<mode>/ tree. Same move xas-rerun-corvus makes.
    try:
        archive_mode_artifacts(run_dir, None, run_id, corvus_mode,
                               f"hang-rerun{attempt}", dry_run=False)
    except OSError as exc:
        print(f"[watchdog] [{run_id}] warning: could not archive prior CORVUS output: {exc}")

    extra: list[str] = []
    if exclude_hung_node and scheduler == "slurm" and node:
        extra = [f"--exclude={node}"]

    new_corvus = bp._submit_job(corvus_script, cwd=run_dir, scheduler=scheduler,
                                extra_submit_args=extra)
    _batch_log_append(batch_log, f"corvus-{corvus_mode}-hang-rerun{attempt}-{run_id}",
                      "SUBMITTED", job_id=new_corvus)
    print(f"[watchdog] [{run_id}] resubmitted CORVUS -> {new_corvus}"
          + (f" ({' '.join(extra)})" if extra else ""))

    _regate_postprocess(batch_root, batch_log, scheduler)

    state.attempts.append(
        rerun_state.Attempt(
            attempt=attempt, kind="corvus-hang", remedy="clean-resubmit",
            utc=bp._utc_now_iso(), corvus_job_id=new_corvus,
            note=f"hung on {node}" if node else None,
        )
    )
    rerun_state.save_state(state_file, state)
    return "resubmitted"


# --------------------------------------------------------------------------- #
# Watch loop
# --------------------------------------------------------------------------- #
def _batch_has_active_jobs(batch_log: Path, scheduler: str) -> bool:
    sched = _sched.get_scheduler(scheduler)
    orca = _batch_log.submitted_job_ids(batch_log, ("orca-",))
    corvus = _batch_log.submitted_job_ids(batch_log, bp.CORVUS_JOB_PREFIX)
    post = _batch_log.submitted_job_ids(batch_log, (bp.POSTPROCESS_JOB_PREFIX,))
    return bool(sched.active_job_ids(orca + corvus + post))


def sweep_once(
    batch_root: Path, scheduler: str, monitor: HangMonitor, run_dirs: dict[str, Path],
    *, max_reruns: int, exclude_hung_node: bool, now: float | None = None,
) -> list[str]:
    """One poll: sample RUNNING ORCA jobs, recover any the monitor calls hung.

    Returns the run ids acted on this sweep (empty when all quiet).
    """
    sched = _sched.get_scheduler(scheduler)
    batch_log = batch_root / "batch-jobs.log"
    orca_samples, orca_dir = sample_running_orca(sched, batch_log, run_dirs)
    corvus_samples, corvus_dir = sample_running_corvus(sched, batch_log, run_dirs)
    # One monitor tracks both stages: ORCA and CORVUS job ids are disjoint, so a
    # single per-id CPU/mtime history is unambiguous.
    hung_ids = monitor.update(time.time() if now is None else now,
                              orca_samples + corvus_samples)
    acted: list[str] = []
    for jid in hung_ids:
        if jid in orca_dir:
            recover_hung(batch_root, orca_dir[jid], jid, scheduler,
                         max_reruns=max_reruns, exclude_hung_node=exclude_hung_node)
            acted.append(orca_dir[jid].name)
        elif jid in corvus_dir:
            recover_hung_corvus(batch_root, corvus_dir[jid], jid, scheduler,
                                max_reruns=max_reruns, exclude_hung_node=exclude_hung_node)
            acted.append(f"{corvus_dir[jid].name} (corvus)")
    return acted


def watch(
    batch_root: Path, scheduler: str, *, interval: float, idle_threshold: float,
    max_reruns: int, exclude_hung_node: bool, once: bool = False,
) -> int:
    batch_log = batch_root / "batch-jobs.log"
    monitor = HangMonitor(idle_threshold)
    run_dirs = index_run_dirs(batch_root)
    print(f"[watchdog] watching {batch_root} (scheduler={scheduler}, "
          f"interval={interval:g}s, idle-threshold={idle_threshold:g}s, "
          f"max-reruns={max_reruns}); {len(run_dirs)} run dir(s) indexed")
    while True:
        if not _batch_has_active_jobs(batch_log, scheduler):
            print("[watchdog] no active ORCA/CORVUS/postprocess jobs left; exiting")
            return 0
        # Re-index in case a mode was added to the batch since we started.
        run_dirs = index_run_dirs(batch_root) or run_dirs
        try:
            acted = sweep_once(batch_root, scheduler, monitor, run_dirs,
                               max_reruns=max_reruns, exclude_hung_node=exclude_hung_node)
        except Exception as exc:  # noqa: BLE001 - a bad poll must not kill the watchdog
            print(f"[watchdog] warning: sweep failed ({exc!r}); continuing")
            acted = []
        if acted:
            print(f"[watchdog] acted on: {', '.join(acted)}")
        if once:
            return 0
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path, help="Batch output root (holds batch-jobs.log)")
    parser.add_argument("--scheduler", choices=sorted(bp.SCHEDULER_SUBMIT_COMMAND),
                        default=bp._default_scheduler(),
                        help="Scheduler backend (default: $PIPELINE_SCHEDULER or pbs).")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"Seconds between polls (default: {DEFAULT_INTERVAL_S}).")
    parser.add_argument("--idle-threshold", type=float, default=DEFAULT_IDLE_THRESHOLD_S,
                        help="Silent-log + flat-CPU seconds before a job is called hung "
                             f"(default: {DEFAULT_IDLE_THRESHOLD_S}).")
    parser.add_argument("--max-reruns", type=int, default=DEFAULT_MAX_RERUNS,
                        help=f"Hang-resubmits per structure before escalating "
                             f"(default: {DEFAULT_MAX_RERUNS}).")
    parser.add_argument("--no-exclude-node", dest="exclude_hung_node", action="store_false",
                        help="Do not add --exclude=<node> when resubmitting a hung job.")
    parser.add_argument("--once", action="store_true",
                        help="Run a single poll and exit (for cron/testing).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"ERROR: batch_root is not a directory: {batch_root}")
    if not (batch_root / "batch-jobs.log").is_file():
        raise SystemExit(f"ERROR: no batch-jobs.log in {batch_root}")
    return watch(
        batch_root, args.scheduler, interval=args.interval,
        idle_threshold=args.idle_threshold, max_reruns=args.max_reruns,
        exclude_hung_node=args.exclude_hung_node, once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
