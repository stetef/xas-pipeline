"""Tests for the hang watchdog (:mod:`xas_pipeline.cli.watchdog`).

Detection (:class:`HangMonitor`) is pure and driven directly with synthetic
samples; recovery runs against a fake scheduler installed at the single seam,
:func:`xas_pipeline.scheduler.run_command`, so no scheduler is needed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from xas_pipeline import recovery, rerun_state
from xas_pipeline import scheduler as _sched
from xas_pipeline.cli import watchdog as wd


# --------------------------------------------------------------------------- #
# fake scheduler (the run_command seam)
# --------------------------------------------------------------------------- #
class FakeScheduler:
    """Answers sbatch/scancel/squeue/sstat/sacct (and qsub/qdel/qstat) in memory.

    ``states`` maps job id -> RUNNING/PENDING; a submitted job starts PENDING and
    a cancelled one disappears. ``submits`` records each accepted submission.
    """

    def __init__(self, states=None):
        self.states: dict[str, str] = dict(states or {})
        self.calls: list[list[str]] = []
        self.submits: list[dict] = []
        self.cancelled: list[str] = []
        self.cpu: dict[str, str] = {}          # sstat AveCPU per job id
        self.fail_cancel: set[str] = set()
        self.fail_submit: set[str] = set()     # script names refused
        self.keep_cancelled_listed = False     # emulate scancel -> squeue lag
        self._next = 1000

    @staticmethod
    def _done(stdout="", rc=0):
        return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="" if rc == 0 else "refused")

    def __call__(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        prog = cmd[0]
        if prog in ("sbatch", "qsub"):
            script = cmd[-1]
            if script in self.fail_submit:
                return self._done(rc=1)
            self._next += 1
            job_id = str(self._next)
            deps, extra = {}, []
            args = cmd[1:-1]
            i = 0
            while i < len(args):
                arg = args[i]
                if arg.startswith("--dependency="):
                    kind, _, ids = arg.removeprefix("--dependency=").partition(":")
                    deps[kind] = ids.split(":")
                elif arg == "-W":
                    kind, _, ids = args[i + 1].removeprefix("depend=").partition(":")
                    deps[kind] = ids.split(":")
                    i += 1
                else:
                    extra.append(arg)
                i += 1
            self.submits.append({"script": script, "cwd": Path(cwd) if cwd else None, "id": job_id,
                                 "afterok": deps.get("afterok", []), "afterany": deps.get("afterany", []),
                                 "extra": extra})
            self.states[job_id] = "PENDING"
            return self._done(f"Submitted batch job {job_id}\n" if prog == "sbatch" else f"{job_id}.pbs\n")
        if prog in ("scancel", "qdel"):
            job_id = cmd[1]
            if job_id in self.fail_cancel:
                return self._done(rc=1)
            self.cancelled.append(job_id)
            if not self.keep_cancelled_listed:
                self.states.pop(job_id, None)
            return self._done()
        if prog == "squeue":
            ids = cmd[cmd.index("-j") + 1].split(",")
            with_state = "%i %T" in cmd
            lines = [f"{j} {self.states[j]}" if with_state else j for j in ids if j in self.states]
            return self._done("\n".join(lines) + ("\n" if lines else ""))
        if prog == "sstat":
            job_id = cmd[cmd.index("-j") + 1].split(".")[0]
            return self._done(self.cpu.get(job_id, "") + "\n")
        if prog == "sacct":
            return self._done("00:00:00\n")
        raise AssertionError(f"unexpected scheduler command {cmd}")

    def by_script(self, suffix: str) -> list[dict]:
        return [s for s in self.submits if s["script"].endswith(suffix)]


@pytest.fixture
def fake(monkeypatch):
    sched = FakeScheduler()
    monkeypatch.setattr(_sched, "run_command", sched)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    return sched


SLURM = _sched.get_scheduler("slurm")


# --------------------------------------------------------------------------- #
# duration parsing (scheduler helper the CPU-veto relies on)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("00:00:00", 0.0),
    ("01:02:03", 3723.0),
    ("2-03:00:00", 183600.0),
    ("05:30.5", 330.5),
    ("123.4", 123.4),
    ("", None),
    ("INVALID", None),
])
def test_parse_elapsed_to_seconds(text, expected):
    assert _sched.parse_elapsed_to_seconds(text) == expected


# --------------------------------------------------------------------------- #
# job-name -> run-id
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,run_id", [
    ("orca-8zg9_cluster8_Zn-caopt-anfreq", "8zg9_cluster8_Zn-caopt-anfreq"),
    ("orca-rerun1-8zg9_cluster8_Zn-caopt-anfreq", "8zg9_cluster8_Zn-caopt-anfreq"),
    ("orca-hang-rerun2-8zg9_cluster8_Zn-caopt-anfreq", "8zg9_cluster8_Zn-caopt-anfreq"),
    ("orca-stage-rerun1-id-caopt-anfreq", "id-caopt-anfreq"),
    ("orca-stage-hang-rerun1-id-caopt-anfreq", "id-caopt-anfreq"),
    ("corvus-xas-A_zn1-hopt-spring", "A_zn1-hopt-spring"),
    ("corvus-xas-hang-rerun1-A_zn1-hopt-spring", "A_zn1-hopt-spring"),
    ("rerun-corvus-xas-B_zn1-asis-spring", "B_zn1-asis-spring"),
    ("submit-corvus-xas-B_zn1-asis-spring", "B_zn1-asis-spring"),
    ("postprocess-batch", None),
    ("watchdog-batch", None),
])
def test_run_id_from_job_name(name, run_id):
    assert recovery.run_id_from_job_name(name) == run_id


# --------------------------------------------------------------------------- #
# HangMonitor
# --------------------------------------------------------------------------- #
def _sample(jid="J1", *, log_mtime, cpu=None, in_flight=True):
    return wd.Sample(job_id=jid, log_mtime=log_mtime, cpu_seconds=cpu, in_flight=in_flight)


def test_monitor_never_flags_on_first_sighting():
    m = wd.HangMonitor(idle_threshold=3600)
    # Already silent for 2h, but first time we see it -> establish a baseline.
    assert m.update(now=10_000, samples=[_sample(log_mtime=10_000 - 7200, cpu=None)]) == []


def test_monitor_flags_persistent_silence_with_flat_cpu():
    m = wd.HangMonitor(idle_threshold=3600)
    frozen = 100.0
    m.update(now=5000, samples=[_sample(log_mtime=frozen, cpu=42.0)])           # baseline
    hung = m.update(now=5600, samples=[_sample(log_mtime=frozen, cpu=42.0)])    # still silent, CPU flat
    assert hung == ["J1"]


def test_monitor_cpu_advance_vetoes_a_quiet_log():
    m = wd.HangMonitor(idle_threshold=3600)
    frozen = 100.0
    m.update(now=5000, samples=[_sample(log_mtime=frozen, cpu=42.0)])
    # Log still hasn't been written (quiet), but CPU climbed -> busy, not hung.
    hung = m.update(now=5600, samples=[_sample(log_mtime=frozen, cpu=99.0)])
    assert hung == []


def test_monitor_below_threshold_not_flagged():
    m = wd.HangMonitor(idle_threshold=3600)
    m.update(now=5000, samples=[_sample(log_mtime=4900, cpu=1.0)])
    hung = m.update(now=5001, samples=[_sample(log_mtime=4900, cpu=1.0)])  # age ~101s
    assert hung == []


def test_monitor_forgets_finished_job():
    m = wd.HangMonitor(idle_threshold=3600)
    m.update(now=5000, samples=[_sample(log_mtime=100.0, cpu=1.0)])
    # No longer in flight -> baseline cleared, and not flagged.
    assert m.update(now=5600, samples=[_sample(log_mtime=100.0, cpu=1.0, in_flight=False)]) == []
    # Even if it reappears silent, it must re-baseline (one clean poll) first.
    assert m.update(now=9200, samples=[_sample(log_mtime=100.0, cpu=1.0)]) == []


def test_monitor_flags_when_cpu_unreadable_but_persistently_silent():
    m = wd.HangMonitor(idle_threshold=3600)
    frozen = 0.0
    m.update(now=5000, samples=[_sample(log_mtime=frozen, cpu=None)])          # baseline
    hung = m.update(now=5600, samples=[_sample(log_mtime=frozen, cpu=None)])   # can't veto -> mtime rules
    assert hung == ["J1"]


# --------------------------------------------------------------------------- #
# batch discovery
# --------------------------------------------------------------------------- #
def test_running_jobs_reads_only_submitted_running_pipeline_jobs(tmp_path, fake):
    """Was orca_job_names/corvus_job_names: only SUBMITTED ORCA/CORVUS lines,
    every CORVUS entry point included, and only jobs the scheduler says RUN."""
    log = tmp_path / "batch-jobs.log"
    log.write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-A_zn1\tSUBMITTED\tjob_id=100\n"
        "corvus-xas-A_zn1\tSUBMITTED\tjob_id=101\n"
        "orca-B_zn1\tSUBMIT_FAILED\n"                              # not submitted
        "orca-hang-rerun1-C_zn1\tSUBMITTED\tjob_id=200\n"
        "rerun-corvus-xas-D_zn1\tSUBMITTED\tjob_id=303\n"          # alt entry point
        "postprocess-batch\tSUBMITTED\tjob_id=900\n",               # not a run job
        encoding="utf-8",
    )
    dirs = {rid: tmp_path / rid for rid in ("A_zn1", "B_zn1", "C_zn1", "D_zn1")}
    fake.states.update({"100": "PENDING", "101": "RUNNING", "200": "RUNNING", "303": "RUNNING", "900": "RUNNING"})
    jobs = {(j.job_id, j.run_id, j.stage) for j in wd.running_jobs(SLURM, log, dirs)}
    assert jobs == {("101", "A_zn1", "corvus"), ("200", "C_zn1", "orca"), ("303", "D_zn1", "corvus")}


def test_index_run_dirs_keys_by_run_id(tmp_path):
    rd = tmp_path / "struct" / "A_zn1-caopt-anfreq"
    rd.mkdir(parents=True)
    (rd / "generated-A_zn1-caopt-anfreq-orca.script").write_text("#!/bin/bash\n")
    index = wd.index_run_dirs(tmp_path)
    assert index == {"A_zn1-caopt-anfreq": rd}


def test_index_run_dirs_split_layout_and_quarantine(tmp_path):
    """A post-processed run's scripts live in working-<id>/; quarantined runs are not indexed."""
    working = tmp_path / "A_zn1" / "working-A_zn1"
    working.mkdir(parents=True)
    (working / "generated-A_zn1-orca.script").write_text("#!/bin/bash\n")
    bad = tmp_path / "failed-orca" / "B_zn1"
    bad.mkdir(parents=True)
    (bad / "generated-B_zn1-orca.script").write_text("#!/bin/bash\n")
    assert wd.index_run_dirs(tmp_path) == {"A_zn1": working}


def test_job_names_resolve_by_exact_run_id(tmp_path, fake):
    """Behaviour change (WIP 77e7365): job names are parsed exactly; the old
    substring fallback (``orca-<id>-rerun-hang2``) and suffix matching are gone,
    so one run id that prefixes another can no longer capture its jobs."""
    log = tmp_path / "batch-jobs.log"
    log.write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-A_zn1-caopt-anfreq\tSUBMITTED\tjob_id=1\n"
        "orca-hang-rerun2-A_zn1-caopt-anfreq\tSUBMITTED\tjob_id=2\n"
        "orca-A_zn1-caopt-anfreq-rerun-hang2\tSUBMITTED\tjob_id=3\n"   # non-standard: not ours
        "orca-something-else\tSUBMITTED\tjob_id=4\n",
        encoding="utf-8",
    )
    fake.states.update({j: "RUNNING" for j in "1234"})
    dirs = {"A_zn1": tmp_path / "A_zn1", "A_zn1-caopt-anfreq": tmp_path / "A_zn1-caopt-anfreq"}
    resolved = {j.job_id: j.run_id for j in wd.running_jobs(SLURM, log, dirs)}
    assert resolved == {"1": "A_zn1-caopt-anfreq", "2": "A_zn1-caopt-anfreq"}


# --------------------------------------------------------------------------- #
# recovery wiring
# --------------------------------------------------------------------------- #
def _make_batch(tmp_path: Path, run_id: str = "A_zn1-caopt-anfreq", *, postprocess_logged: bool = True):
    batch = tmp_path / "batch"
    run_dir = batch / "A_zn1" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / f"generated-{run_id}-orca.script").write_text("#!/bin/bash\n")
    (run_dir / f"generated-{run_id}-corvus-xas-wrapper.script").write_text("#!/bin/bash\n")
    # timing sidecar: started, no exit_code -> in flight; hostname for --exclude
    (run_dir / f"{run_id}-orca.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\n", encoding="utf-8"
    )
    (run_dir / f"{run_id}-orca.log").write_text("frozen\n", encoding="utf-8")
    (batch / f"generated-postprocess-{batch.name}.script").write_text("#!/bin/bash\n")
    (batch / "batch-jobs.log").write_text(
        "job_name\tstatus\tjob_id\n"
        f"orca-{run_id}\tSUBMITTED\tjob_id=500\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n"
        + (f"postprocess-{batch.name}\tSUBMITTED\tjob_id=900\n" if postprocess_logged else ""),
        encoding="utf-8",
    )
    return batch, run_dir


def _orca_job(run_dir: Path, job_id: str = "500") -> wd.Running:
    return wd.Running(job_id=job_id, run_id=run_dir.name, run_dir=run_dir, stage="orca")


def test_recover_hung_resubmits_and_regates(tmp_path, fake):
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING", "900": "PENDING"})

    outcome = wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True)
    assert outcome == "resubmitted"

    # Hung ORCA (500) + its stale CORVUS (501) + the under-gated postproc (900) all cancelled.
    assert set(fake.cancelled) == {"500", "501", "900"}

    [orca] = fake.by_script("-orca.script")
    [corvus] = fake.by_script("-corvus-xas-wrapper.script")
    [postproc] = fake.by_script(f"generated-postprocess-{batch.name}.script")

    # ORCA resubmitted excluding the node it hung on.
    assert orca["extra"] == ["--exclude=sdfmilanTEST"]
    # CORVUS chained afterok on the fresh ORCA id.
    assert corvus["afterok"] == [orca["id"]]
    # Postproc re-gated afterany on the outstanding CORVUS set (the new wrapper).
    assert postproc["afterany"] == [corvus["id"]]
    # The replacement postprocess is submitted BEFORE the stale one is cancelled.
    calls = [c[0] + ":" + c[-1] for c in fake.calls if c[0] in ("sbatch", "scancel")]
    assert calls.index(f"sbatch:generated-postprocess-{batch.name}.script") < calls.index("scancel:900")

    # Bounded-ladder state recorded under the separate 'hang' file.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = json.loads(state_file.read_text())
    assert len(state["attempts"]) == 1
    assert state["attempts"][0]["kind"] == "hang"
    assert state["attempts"][0]["orca_job_id"] == orca["id"]

    # And the log records the hang-rerun submissions.
    log_text = (batch / "batch-jobs.log").read_text()
    assert f"orca-hang-rerun1-{run_dir.name}\tSUBMITTED\tjob_id={orca['id']}" in log_text
    assert f"corvus-xas-hang-rerun1-{run_dir.name}\tSUBMITTED" in log_text


def test_recover_hung_escalates_when_ladder_exhausted(tmp_path, fake):
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING"})

    # Pre-seed two prior hang attempts so the next one exceeds max_reruns=2.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    for i in (1, 2):
        state.attempts.append(rerun_state.Attempt(
            attempt=i, kind="hang", remedy="clean-resubmit", utc="x"))
    rerun_state.save_state(state_file, state)

    outcome = wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True)
    assert outcome == "escalated"

    # Still cancels the wedged job + its corvus, but resubmits NO new ORCA.
    assert "500" in fake.cancelled and "501" in fake.cancelled
    assert not fake.by_script("-orca.script")

    reloaded = json.loads(state_file.read_text())
    assert reloaded["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert "NEEDS_HUMAN" in (batch / "batch-jobs.log").read_text()


def test_recover_hung_is_idempotent_once_terminal(tmp_path, fake):
    batch, run_dir = _make_batch(tmp_path)
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    state.resolution = rerun_state.RESOLUTION_NEEDS_HUMAN
    rerun_state.save_state(state_file, state)

    assert wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True) == "skipped-terminal"
    assert fake.cancelled == [] and fake.submits == []


# --------------------------------------------------------------------------- #
# CORVUS-stage hang coverage
# --------------------------------------------------------------------------- #
def test_index_run_dirs_includes_corvus_only_dirs(tmp_path):
    # asis-spring (interp-raw) runs have NO ORCA script, only a CORVUS wrapper.
    rd = tmp_path / "struct" / "A_zn1-asis-spring"
    rd.mkdir(parents=True)
    (rd / "generated-A_zn1-asis-spring-corvus-xas-wrapper.script").write_text("#!/bin/bash\n")
    index = wd.index_run_dirs(tmp_path)
    assert index == {"A_zn1-asis-spring": rd}


def test_find_corvus_log_and_in_flight(tmp_path):
    rd = tmp_path / "A_zn1-hopt-spring"
    rd.mkdir()
    run_id = rd.name
    (rd / "corvus-A_zn1-hopt-spring-xas.out").write_text("running\n", encoding="utf-8")
    # in flight: start but no exit_code
    (rd / "A_zn1-hopt-spring-corvus-xas.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\n", encoding="utf-8")
    assert wd.find_corvus_log(rd, run_id) == rd / "corvus-A_zn1-hopt-spring-xas.out"
    assert wd.corvus_in_flight(rd, run_id) is True
    # finished: exit_code present -> not in flight
    (rd / "A_zn1-hopt-spring-corvus-xas.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\nexit_code=0\n", encoding="utf-8")
    assert wd.corvus_in_flight(rd, run_id) is False


def _make_corvus_batch(tmp_path: Path, run_id: str = "A_zn1-asis-spring"):
    """A CORVUS-only (interp-raw) run mid-CORVUS: wrapper + in-flight timing + log."""
    batch = tmp_path / "batch"
    run_dir = batch / "A_zn1" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / f"generated-{run_id}-corvus-xas-wrapper.script").write_text("#!/bin/bash\n")
    (run_dir / f"{run_id}-corvus-xas.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\n", encoding="utf-8")
    (run_dir / f"corvus-{run_id}-xas.out").write_text("frozen\n", encoding="utf-8")
    # partial CORVUS working tree the wedged run left behind
    (run_dir / "Corvus3_cfavg_xas").mkdir()
    (run_dir / "Corvus3_cfavg_xas" / "partial.dat").write_text("x\n", encoding="utf-8")
    (batch / f"generated-postprocess-{batch.name}.script").write_text("#!/bin/bash\n")
    (batch / "batch-jobs.log").write_text(
        "job_name\tstatus\tjob_id\n"
        f"orca-{run_id}\tSKIPPED\n"                       # interp-raw: no ORCA stage
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n"
        f"postprocess-{batch.name}\tSUBMITTED\tjob_id=900\n",
        encoding="utf-8",
    )
    return batch, run_dir


def _corvus_job(run_dir: Path) -> wd.Running:
    return wd.Running(job_id="501", run_id=run_dir.name, run_dir=run_dir, stage="corvus")


def test_recover_hung_corvus_resubmits_and_regates(tmp_path, fake):
    batch, run_dir = _make_corvus_batch(tmp_path)
    fake.states.update({"501": "RUNNING", "900": "PENDING"})

    outcome = wd.recover(SLURM, batch, _corvus_job(run_dir), max_reruns=2, exclude_node=True)
    assert outcome == "resubmitted"

    # Hung CORVUS (501) + under-gated postproc (900) cancelled; no ORCA involved.
    assert set(fake.cancelled) == {"501", "900"}

    [corvus] = fake.by_script("-corvus-xas-wrapper.script")
    [postproc] = fake.by_script(f"generated-postprocess-{batch.name}.script")
    # CORVUS resubmitted standalone (no dependency) excluding the hung node.
    assert corvus["extra"] == ["--exclude=sdfmilanTEST"]
    assert corvus["afterok"] == [] and corvus["afterany"] == []
    # No ORCA resubmit for a CORVUS-only run.
    assert not fake.by_script("-orca.script")
    # Postproc re-gated afterany on the outstanding CORVUS set.
    assert postproc["afterany"] == [corvus["id"]]

    # The partial CORVUS working tree was archived aside for a clean restart.
    # (Tag is now rerun-hang<N>: the rerun- prefix lets the postprocess cleanup
    # delete the snapshot; it was hang-rerun<N> before 77e7365.)
    assert not (run_dir / "Corvus3_cfavg_xas").exists()
    assert (run_dir / "Corvus3_cfavg_xas.rerun-hang1").is_dir()

    # Bounded ladder recorded under the dedicated 'corvus-hang' file.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="corvus-hang")
    state = json.loads(state_file.read_text())
    assert len(state["attempts"]) == 1 and state["attempts"][0]["kind"] == "corvus-hang"
    assert state["attempts"][0]["corvus_job_id"] == corvus["id"]
    assert f"corvus-xas-hang-rerun1-{run_dir.name}" in (batch / "batch-jobs.log").read_text()


def test_recover_hung_corvus_escalates_when_ladder_exhausted(tmp_path, fake):
    batch, run_dir = _make_corvus_batch(tmp_path)
    fake.states.update({"501": "RUNNING"})
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="corvus-hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    for i in (1, 2):
        state.attempts.append(rerun_state.Attempt(
            attempt=i, kind="corvus-hang", remedy="clean-resubmit", utc="x"))
    rerun_state.save_state(state_file, state)

    outcome = wd.recover(SLURM, batch, _corvus_job(run_dir), max_reruns=2, exclude_node=True)
    assert outcome == "escalated"
    assert "501" in fake.cancelled
    assert not fake.by_script("wrapper.script")  # no new CORVUS wrapper resubmitted
    reloaded = json.loads(state_file.read_text())
    assert reloaded["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert "NEEDS_HUMAN" in (batch / "batch-jobs.log").read_text()


# --------------------------------------------------------------------------- #
# regressions for the 77e7365 recovery fixes
# --------------------------------------------------------------------------- #
def test_cancel_failure_escalates_without_resubmitting(tmp_path, fake):
    """Cancel result checked: resubmitting a job that may still run would double it."""
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING", "900": "PENDING"})
    fake.fail_cancel.add("500")

    assert wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True) == "escalated"
    assert fake.submits == []
    assert "501" not in fake.cancelled  # nothing else touched either
    state = json.loads(rerun_state.state_path(run_dir, run_dir.name, kind="hang").read_text())
    assert state["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN and state["attempts"] == []
    assert "could not be cancelled" in (batch / "batch-jobs.log").read_text()


def test_submit_failure_after_cancel_keeps_the_run_recorded(tmp_path, fake):
    """The hung job is gone; a refused resubmit must leave a trace, not lose the run."""
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING"})
    fake.fail_submit.add(f"generated-{run_dir.name}-orca.script")

    assert wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True) == "escalated"
    assert "500" in fake.cancelled
    state = json.loads(rerun_state.state_path(run_dir, run_dir.name, kind="hang").read_text())
    assert state["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert len(state["attempts"]) == 1 and state["attempts"][0]["orca_job_id"] is None
    log_text = (batch / "batch-jobs.log").read_text()
    assert f"orca-hang-rerun1-{run_dir.name}\tSUBMIT_FAILED" in log_text
    assert "NEEDS_HUMAN" in log_text and "resubmission after a hang failed" in log_text


def test_declined_postprocess_is_not_submitted(tmp_path, fake):
    """An operator who ran --no-postprocess keeps that choice: regate only if pending."""
    batch, run_dir = _make_batch(tmp_path, postprocess_logged=False)
    fake.states.update({"500": "RUNNING", "501": "PENDING"})

    assert wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True) == "resubmitted"
    assert not fake.by_script(f"generated-postprocess-{batch.name}.script")


def test_finished_postprocess_is_not_resubmitted(tmp_path, fake):
    """A postprocess that already ran (no longer queued) is not 'pending' either."""
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING"})  # 900 is gone

    assert wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True) == "resubmitted"
    assert not fake.by_script(f"generated-postprocess-{batch.name}.script")


def test_stale_corvus_of_a_prefix_sibling_is_not_cancelled(tmp_path, fake):
    """Exact run-id matching: A_zn1's hang must not cancel A_zn1-caopt-anfreq's CORVUS."""
    batch, run_dir = _make_batch(tmp_path, run_id="A_zn1")
    with (batch / "batch-jobs.log").open("a") as fh:
        fh.write("corvus-xas-A_zn1-caopt-anfreq\tSUBMITTED\tjob_id=777\n")
    fake.states.update({"500": "RUNNING", "501": "PENDING", "777": "PENDING"})

    wd.recover(SLURM, batch, _orca_job(run_dir), max_reruns=2, exclude_node=True)
    assert "501" in fake.cancelled and "777" not in fake.cancelled


def _hung_sweep_setup(tmp_path, fake):
    batch, run_dir = _make_batch(tmp_path)
    fake.states.update({"500": "RUNNING", "501": "PENDING", "900": "PENDING"})
    return batch, run_dir, wd.HangMonitor(idle_threshold=60), set()


def test_sweep_acts_once_per_hung_job(tmp_path, fake):
    """A cancelled job squeue still lists (scancel lag) is not recovered twice."""
    batch, run_dir, monitor, handled = _hung_sweep_setup(tmp_path, fake)
    fake.keep_cancelled_listed = True
    fake.cpu["500"] = "00:05:00"   # flat CPU
    index = wd.index_run_dirs(batch)
    kw = dict(max_reruns=2, exclude_node=True)

    assert wd.sweep_once(SLURM, batch, monitor, index, handled, now=1e10, **kw) == []   # baseline
    acted = wd.sweep_once(SLURM, batch, monitor, index, handled, now=1e10 + 600, **kw)
    assert acted == [f"{run_dir.name} (orca: resubmitted)"]
    n_orca = len(fake.by_script("-orca.script"))

    fake.states["500"] = "RUNNING"  # still listed on the next polls
    for t in (1200, 1800):
        assert wd.sweep_once(SLURM, batch, monitor, index, handled, now=1e10 + t, **kw) == []
    assert len(fake.by_script("-orca.script")) == n_orca == 1
    assert len(rerun_state.load_state(
        rerun_state.state_path(run_dir, run_dir.name, kind="hang"), run_dir.name).attempts) == 1


def test_sstat_cpu_vetoes_a_quiet_running_job(tmp_path, fake):
    """CPU veto on Slurm comes from sstat (sacct TotalCPU is 0 while a step runs)."""
    batch, _run_dir, monitor, handled = _hung_sweep_setup(tmp_path, fake)
    index = wd.index_run_dirs(batch)
    kw = dict(max_reruns=2, exclude_node=True)
    fake.cpu["500"] = "01:00:00"
    wd.sweep_once(SLURM, batch, monitor, index, handled, now=1e10, **kw)
    fake.cpu["500"] = "01:10:00"   # busy, log silent
    assert wd.sweep_once(SLURM, batch, monitor, index, handled, now=1e10 + 600, **kw) == []
    assert fake.cancelled == [] and fake.submits == []
    assert any(c[0] == "sstat" and "500.batch" in c for c in fake.calls)


# --------------------------------------------------------------------------- #
# one watchdog per batch
# --------------------------------------------------------------------------- #
def _log_watchdogs(batch: Path, *ids: str) -> None:
    with (batch / "batch-jobs.log").open("a") as fh:
        for job_id in ids:
            fh.write(f"watchdog-{batch.name}\tSUBMITTED\tjob_id={job_id}\n")


def test_hand_run_watchdog_defers_to_the_live_job(tmp_path, fake):
    batch, _ = _make_batch(tmp_path)
    _log_watchdogs(batch, "800")
    fake.states.update({"500": "RUNNING", "800": "RUNNING"})
    assert wd.other_live_watchdog(SLURM, batch) == "800"
    assert wd.watch(batch, "slurm", interval=0, idle_threshold=60, max_reruns=2, once=True) == 0
    # It exited at start: never even looked at the batch's jobs.
    assert not any(c[0] == "squeue" and "500" in c[-1] for c in fake.calls)


def test_watchdog_does_not_count_itself(tmp_path, fake, monkeypatch):
    batch, _ = _make_batch(tmp_path)
    _log_watchdogs(batch, "800")
    fake.states.update({"800": "RUNNING"})
    monkeypatch.setenv("SLURM_JOB_ID", "800")
    assert wd.other_live_watchdog(SLURM, batch) is None


def test_two_live_watchdogs_leave_exactly_one(tmp_path, fake, monkeypatch):
    """Two watchdog jobs live at once must not both exit (the batch would go
    unwatched) nor both stay: the earlier-submitted one keeps watching."""
    batch, _ = _make_batch(tmp_path)
    _log_watchdogs(batch, "800", "801")
    fake.states.update({"800": "RUNNING", "801": "RUNNING"})
    monkeypatch.setenv("SLURM_JOB_ID", "800")
    assert wd.other_live_watchdog(SLURM, batch) is None
    monkeypatch.setenv("SLURM_JOB_ID", "801")
    assert wd.other_live_watchdog(SLURM, batch) == "800"
