"""Tests for the hang watchdog (:mod:`xas_pipeline.cli.watchdog`).

Detection (:class:`HangMonitor`) is pure and driven directly with synthetic
samples; recovery is driven with the scheduler-interaction helpers on
``orchestrate`` monkeypatched, so no scheduler is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xas_pipeline import orchestrate as bp
from xas_pipeline import rerun_state, scheduler as _sched
from xas_pipeline.cli import watchdog as wd


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
])
def test_run_id_from_job_name(name, run_id):
    assert wd.run_id_from_job_name(name) == run_id


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
def test_orca_job_names_reads_only_submitted_orca_lines(tmp_path):
    log = tmp_path / "batch-jobs.log"
    log.write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-A_zn1\tSUBMITTED\tjob_id=100\n"
        "corvus-xas-A_zn1\tSUBMITTED\tjob_id=101\n"       # not orca
        "orca-B_zn1\tSUBMIT_FAILED\n"                       # not submitted
        "orca-hang-rerun1-A_zn1\tSUBMITTED\tjob_id=200\n",
        encoding="utf-8",
    )
    assert wd.orca_job_names(log) == {"100": "orca-A_zn1", "200": "orca-hang-rerun1-A_zn1"}


def test_index_run_dirs_keys_by_run_id(tmp_path):
    rd = tmp_path / "struct" / "A_zn1-caopt-anfreq"
    rd.mkdir(parents=True)
    (rd / "generated-A_zn1-caopt-anfreq-orca.script").write_text("#!/bin/bash\n")
    index = wd.index_run_dirs(tmp_path)
    assert index == {"A_zn1-caopt-anfreq": rd}


def test_resolve_run_dir_handles_standard_and_nonstandard_names(tmp_path):
    rd = tmp_path / "A_zn1-caopt-anfreq"
    index = {"A_zn1-caopt-anfreq": rd}
    # Clean pipeline naming (prefix decoration) via the parsed run id.
    assert wd.resolve_run_dir("orca-A_zn1-caopt-anfreq", index) is rd
    assert wd.resolve_run_dir("orca-hang-rerun2-A_zn1-caopt-anfreq", index) is rd
    # Non-standard manual naming (suffix decoration) via the substring fallback.
    assert wd.resolve_run_dir("orca-A_zn1-caopt-anfreq-rerun-hang2", index) is rd
    # No match -> None.
    assert wd.resolve_run_dir("orca-something-else", index) is None


# --------------------------------------------------------------------------- #
# recovery wiring
# --------------------------------------------------------------------------- #
def _make_batch(tmp_path: Path, run_id: str = "A_zn1-caopt-anfreq"):
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
        f"postprocess-{batch.name}\tSUBMITTED\tjob_id=900\n",
        encoding="utf-8",
    )
    return batch, run_dir


@pytest.fixture
def patched_scheduler(monkeypatch):
    """Record cancels; hand out fresh ids for submits; fake postproc discovery."""
    cancelled: list[str] = []
    submitted: list[dict] = []
    counter = {"n": 0}

    def fake_cancel(jid, sched):
        cancelled.append(str(jid))
        return True

    def fake_submit(script, cwd, scheduler, depend_afterok=None, depend_afterany=None,
                    extra_submit_args=None):
        counter["n"] += 1
        new_id = f"NEW{counter['n']}"
        submitted.append({
            "script": Path(script).name,
            "afterok": list(depend_afterok or []),
            "afterany": list(depend_afterany or []),
            "extra": list(extra_submit_args or []),
            "id": new_id,
        })
        return new_id

    monkeypatch.setattr(bp, "_cancel_job", fake_cancel)
    monkeypatch.setattr(bp, "_submit_job", fake_submit)
    monkeypatch.setattr(bp, "pending_postprocess_job_ids", lambda log, sched: ["900"])
    monkeypatch.setattr(bp, "outstanding_corvus_job_ids", lambda log, sched, **k: ["NEW2"])
    return cancelled, submitted


def test_recover_hung_resubmits_and_regates(tmp_path, patched_scheduler):
    cancelled, submitted = patched_scheduler
    batch, run_dir = _make_batch(tmp_path)

    outcome = wd.recover_hung(
        batch, run_dir, "500", "slurm", max_reruns=2, exclude_hung_node=True
    )
    assert outcome == "resubmitted"

    # Hung ORCA (500) + its stale CORVUS (501) + the under-gated postproc (900) all cancelled.
    assert set(cancelled) == {"500", "501", "900"}

    by_script = {s["script"].split("-")[1]: s for s in submitted}  # orca / corvus / postprocess
    orca = next(s for s in submitted if s["script"].endswith("-orca.script"))
    corvus = next(s for s in submitted if "corvus" in s["script"])
    postproc = next(s for s in submitted if "postprocess" in s["script"])

    # ORCA resubmitted excluding the node it hung on.
    assert orca["extra"] == ["--exclude=sdfmilanTEST"]
    # CORVUS chained afterok on the fresh ORCA id.
    assert corvus["afterok"] == [orca["id"]]
    # Postproc re-gated afterany on the outstanding CORVUS set.
    assert postproc["afterany"] == ["NEW2"]

    # Bounded-ladder state recorded under the separate 'hang' file.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = json.loads(state_file.read_text())
    assert len(state["attempts"]) == 1
    assert state["attempts"][0]["kind"] == "hang"

    # And the log records the hang-rerun submissions.
    log_text = (batch / "batch-jobs.log").read_text()
    assert f"orca-hang-rerun1-{run_dir.name}" in log_text


def test_recover_hung_escalates_when_ladder_exhausted(tmp_path, patched_scheduler):
    cancelled, submitted = patched_scheduler
    batch, run_dir = _make_batch(tmp_path)

    # Pre-seed two prior hang attempts so the next one exceeds max_reruns=2.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    for i in (1, 2):
        state.attempts.append(rerun_state.Attempt(
            attempt=i, kind="hang", remedy="clean-resubmit", utc="x"))
    rerun_state.save_state(state_file, state)

    outcome = wd.recover_hung(
        batch, run_dir, "500", "slurm", max_reruns=2, exclude_hung_node=True
    )
    assert outcome == "escalated"

    # Still cancels the wedged job + its corvus, but resubmits NO new ORCA.
    assert "500" in cancelled and "501" in cancelled
    assert not any(s["script"].endswith("-orca.script") for s in submitted)

    reloaded = json.loads(state_file.read_text())
    assert reloaded["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert "NEEDS_HUMAN" in (batch / "batch-jobs.log").read_text()


def test_recover_hung_is_idempotent_once_terminal(tmp_path, patched_scheduler):
    cancelled, submitted = patched_scheduler
    batch, run_dir = _make_batch(tmp_path)
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    state.resolution = rerun_state.RESOLUTION_NEEDS_HUMAN
    rerun_state.save_state(state_file, state)

    assert wd.recover_hung(batch, run_dir, "500", "slurm",
                           max_reruns=2, exclude_hung_node=True) == "skipped-terminal"
    assert cancelled == [] and submitted == []


# --------------------------------------------------------------------------- #
# CORVUS-stage hang coverage
# --------------------------------------------------------------------------- #
def test_corvus_job_names_reads_only_corvus_submitted_lines(tmp_path):
    log = tmp_path / "batch-jobs.log"
    log.write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-A_zn1-hopt-spring\tSUBMITTED\tjob_id=100\n"                 # not corvus
        "corvus-xas-A_zn1-hopt-spring\tSUBMITTED\tjob_id=101\n"
        "corvus-xas-A_zn1-asis-spring\tSUBMIT_FAILED\n"                    # not submitted
        "corvus-xas-hang-rerun1-A_zn1-hopt-spring\tSUBMITTED\tjob_id=202\n"
        "rerun-corvus-B_zn1-asis-spring\tSUBMITTED\tjob_id=303\n",         # alt entry point
        encoding="utf-8",
    )
    assert wd.corvus_job_names(log) == {
        "101": "corvus-xas-A_zn1-hopt-spring",
        "202": "corvus-xas-hang-rerun1-A_zn1-hopt-spring",
        "303": "rerun-corvus-B_zn1-asis-spring",
    }


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
    (rd / "corvus-A_zn1-hopt-spring-xas.out").write_text("running\n", encoding="utf-8")
    # in flight: start but no exit_code
    (rd / "A_zn1-hopt-spring-corvus-xas.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\n", encoding="utf-8")
    assert wd.find_corvus_log(rd) == rd / "corvus-A_zn1-hopt-spring-xas.out"
    assert wd.corvus_run_is_in_flight(rd) is True
    # finished: exit_code present -> not in flight
    (rd / "A_zn1-hopt-spring-corvus-xas.timing").write_text(
        "start_epoch=1\nhostname=sdfmilanTEST\nexit_code=0\n", encoding="utf-8")
    assert wd.corvus_run_is_in_flight(rd) is False


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


def test_recover_hung_corvus_resubmits_and_regates(tmp_path, patched_scheduler):
    cancelled, submitted = patched_scheduler
    batch, run_dir = _make_corvus_batch(tmp_path)

    outcome = wd.recover_hung_corvus(
        batch, run_dir, "501", "slurm", max_reruns=2, exclude_hung_node=True
    )
    assert outcome == "resubmitted"

    # Hung CORVUS (501) + under-gated postproc (900) cancelled; no ORCA involved.
    assert set(cancelled) == {"501", "900"}

    corvus = next(s for s in submitted if "corvus" in s["script"])
    postproc = next(s for s in submitted if "postprocess" in s["script"])
    # CORVUS resubmitted standalone (no dependency) excluding the hung node.
    assert corvus["extra"] == ["--exclude=sdfmilanTEST"]
    assert corvus["afterok"] == [] and corvus["afterany"] == []
    # No ORCA resubmit for a CORVUS-only run.
    assert not any(s["script"].endswith("-orca.script") for s in submitted)
    # Postproc re-gated afterany on the outstanding CORVUS set.
    assert postproc["afterany"] == ["NEW2"]

    # The partial CORVUS working tree was archived aside for a clean restart.
    assert not (run_dir / "Corvus3_cfavg_xas").exists()
    assert (run_dir / "Corvus3_cfavg_xas.hang-rerun1").is_dir()

    # Bounded ladder recorded under the dedicated 'corvus-hang' file.
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="corvus-hang")
    state = json.loads(state_file.read_text())
    assert len(state["attempts"]) == 1 and state["attempts"][0]["kind"] == "corvus-hang"
    assert f"corvus-xas-hang-rerun1-{run_dir.name}" in (batch / "batch-jobs.log").read_text()


def test_recover_hung_corvus_escalates_when_ladder_exhausted(tmp_path, patched_scheduler):
    cancelled, submitted = patched_scheduler
    batch, run_dir = _make_corvus_batch(tmp_path)
    state_file = rerun_state.state_path(run_dir, run_dir.name, kind="corvus-hang")
    state = rerun_state.load_state(state_file, run_dir.name)
    for i in (1, 2):
        state.attempts.append(rerun_state.Attempt(
            attempt=i, kind="corvus-hang", remedy="clean-resubmit", utc="x"))
    rerun_state.save_state(state_file, state)

    outcome = wd.recover_hung_corvus(
        batch, run_dir, "501", "slurm", max_reruns=2, exclude_hung_node=True
    )
    assert outcome == "escalated"
    assert "501" in cancelled
    assert not any("corvus" in s["script"] and s["script"].endswith("wrapper.script")
                   for s in submitted)  # no new CORVUS wrapper resubmitted
    reloaded = json.loads(state_file.read_text())
    assert reloaded["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert "NEEDS_HUMAN" in (batch / "batch-jobs.log").read_text()
