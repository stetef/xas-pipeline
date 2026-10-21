"""Regression tests for the Phase-1 orchestration fixes, run offline.

Every scheduler command goes through ``xas_pipeline.scheduler.run_command``;
:class:`FakeScheduler` replaces it, hands out job ids, answers ``squeue`` from a
set of "live" ids, and records every call in order.

Covered fixes:
* the batch postprocess is submitted *before* the stale one it replaces is
  cancelled (a refused submission used to leave the batch with none);
* a submission failure mid-loop no longer aborts the batch: later structures
  are still submitted and the postprocess still waits on every CORVUS job;
* ``xas-postprocess --submit`` replaces a pending, under-gated postprocess;
* ``submit_batch`` does not start a second watchdog while one is live.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xas_pipeline import batch_log, orchestrate, scheduler
from xas_pipeline.cli import postprocess as postprocess_cli

FIXTURE_XYZ = Path(__file__).resolve().parents[1] / "fixtures" / "xyz_his" / "4j1v_cluster2_Zn.xyz"
MODE = "quick"


class FakeScheduler:
    """Scripted stand-in for sbatch/squeue/scancel behind ``run_command``."""

    def __init__(self, *, live=(), refuse=()):
        self.live = set(live)
        #: Script names whose submission is refused (sbatch exits 1).
        self.refuse = set(refuse)
        self.calls: list[list[str]] = []
        self.next_id = 1000

    def __call__(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        prog = cmd[0]
        if prog == "sbatch":
            if cmd[-1] in self.refuse:
                return subprocess.CompletedProcess(cmd, 1, "", "sbatch: error: refused\n")
            self.next_id += 1
            self.live.add(str(self.next_id))
            return subprocess.CompletedProcess(cmd, 0, f"Submitted batch job {self.next_id}\n", "")
        if prog == "squeue":
            asked = cmd[cmd.index("-j") + 1].split(",")
            return subprocess.CompletedProcess(cmd, 0, "".join(f"{j}\n" for j in asked if j in self.live), "")
        if prog == "scancel":
            self.live.discard(cmd[1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected scheduler command: {cmd}")

    def submissions(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "sbatch"]

    def submitted(self, script_name: str) -> list[str]:
        """The sbatch argv for *script_name* (exactly one expected)."""
        hits = [c for c in self.submissions() if c[-1] == script_name]
        assert len(hits) == 1, f"{script_name}: {len(hits)} submissions"
        return hits[0]

    def index(self, prog: str, last_arg: str) -> int:
        return next(i for i, c in enumerate(self.calls) if c[0] == prog and c[-1] == last_arg)


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        fake = FakeScheduler(**kwargs)
        monkeypatch.setattr(scheduler, "run_command", fake)
        return fake

    return install


SLURM = scheduler.get_scheduler("slurm")


def _seed_log(batch_root: Path, lines: list[str]) -> Path:
    batch_root.mkdir(parents=True, exist_ok=True)
    log = batch_root / batch_log.LOG_NAME
    batch_log.init(log, SLURM)
    with log.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
    return log


def _records(log: Path, status: str) -> list[tuple[str, str | None]]:
    return [(name, job_id) for name, st, job_id in batch_log.records(log) if st == status]


# ── postprocess replacement: submit first, cancel after ──────────────────────


def test_postprocess_is_submitted_before_the_stale_one_is_cancelled(tmp_path, fake):
    batch = tmp_path / "batch"
    log = _seed_log(batch, [
        "corvus-xas-OLD-quick\tSUBMITTED\tjob_id=101\n",   # earlier mode, still running
        "postprocess-batch\tSUBMITTED\tjob_id=301\n",      # its postprocess, still pending
    ])
    script = batch / "generated-postprocess-batch.script"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    sched = fake(live={"101", "301"})

    job_id = orchestrate.submit_postprocess(SLURM, batch, script, ["555"])

    argv = sched.submitted(script.name)
    assert "--dependency=afterany:555:101" in argv, "gated on this invocation's and earlier CORVUS jobs"
    assert sched.index("sbatch", script.name) < sched.index("scancel", "301"), \
        "the stale postprocess was cancelled before its replacement was accepted"
    assert _records(log, "SUBMITTED")[-1] == ("postprocess-batch", job_id)
    assert ("postprocess-batch", "301") in _records(log, "REPLACED")


def test_refused_replacement_leaves_the_stale_postprocess_in_place(tmp_path, fake):
    batch = tmp_path / "batch"
    log = _seed_log(batch, ["postprocess-batch\tSUBMITTED\tjob_id=301\n"])
    script = batch / "generated-postprocess-batch.script"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    sched = fake(live={"301"}, refuse={script.name})

    with pytest.raises(RuntimeError, match="sbatch failed"):
        orchestrate.submit_postprocess(SLURM, batch, script, ["555"])

    assert not [c for c in sched.calls if c[0] == "scancel"], "batch left with no postprocess at all"
    assert ("postprocess-batch", None) in _records(log, "SUBMIT_FAILED")
    assert not _records(log, "REPLACED")


# ── submit_batch: failures mid-loop ──────────────────────────────────────────


def _three_structures(tmp_path: Path) -> list[Path]:
    xyz_dir = tmp_path / "xyz"
    xyz_dir.mkdir()
    files = []
    for stem in ("A", "B", "C"):
        dest = xyz_dir / f"{stem}.xyz"
        shutil.copy2(FIXTURE_XYZ, dest)
        files.append(dest)
    return files


def test_mid_loop_submit_failure_continues_and_orphans_nothing(tmp_path, fake):
    """A's ORCA and B's CORVUS are refused; the batch carries on.

    Before the fix the first refusal raised out of the loop: C was never
    submitted, no postprocess or watchdog was queued for the jobs already
    accepted, and no state file recorded them.
    """
    xyz_files = _three_structures(tmp_path)
    batch = tmp_path / "batch"
    sched = fake(refuse={f"generated-A-{MODE}-orca.script", f"generated-B-{MODE}-corvus-xas-wrapper.script"})

    result = orchestrate.submit_batch(xyz_files, batch, MODE, SLURM)

    assert set(result.errors) == {f"orca-A-{MODE}", f"corvus-xas-B-{MODE}"}
    runs = {r.run_id: r for r in result.runs}
    assert f"A-{MODE}" not in runs, "no CORVUS job may be queued behind a refused ORCA job"
    assert runs[f"B-{MODE}"].orca_job_id.isdigit() and runs[f"B-{MODE}"].corvus_job_id is None
    c_run = runs[f"C-{MODE}"]
    assert c_run.orca_job_id.isdigit() and c_run.corvus_job_id.isdigit(), "C was submitted after the failures"
    assert f"--dependency=afterok:{c_run.orca_job_id}" in sched.submitted(f"generated-C-{MODE}-corvus-xas-wrapper.script")

    pp_argv = sched.submitted(f"generated-postprocess-{batch.name}.script")
    assert f"--dependency=afterany:{c_run.corvus_job_id}" in pp_argv
    assert result.postprocess_job_id and result.watchdog_job_id

    log = batch / batch_log.LOG_NAME
    assert {name for name, _ in _records(log, "SUBMIT_FAILED")} == set(result.errors)
    state = orchestrate.render_state(result, sched=SLURM, input_path=xyz_files[0].parent,
                                     download_destination=batch / "downloading-station")
    for name in result.errors:
        assert f"SUBMISSION FAILED: {name}" in state
    assert f"orca_job_id:    {runs[f'B-{MODE}'].orca_job_id}" in state, "B's accepted ORCA job is on record"


def test_refused_postprocess_is_reported_not_raised(tmp_path, fake):
    xyz_files = _three_structures(tmp_path)[:1]
    batch = tmp_path / "batch"
    fake(refuse={"generated-postprocess-batch.script"})

    result = orchestrate.submit_batch(xyz_files, batch, MODE, SLURM)

    assert set(result.errors) == {"postprocess-batch"}
    assert result.postprocess_job_id is None
    assert result.watchdog_job_id, "the watchdog is still started for the jobs that were queued"


# ── one watchdog per batch ───────────────────────────────────────────────────


def test_submit_batch_does_not_start_a_second_watchdog(tmp_path, fake):
    xyz_files = _three_structures(tmp_path)[:1]
    batch = tmp_path / "batch"
    _seed_log(batch, ["watchdog-batch\tSUBMITTED\tjob_id=500\n"])
    sched = fake(live={"500"})

    result = orchestrate.submit_batch(xyz_files, batch, MODE, SLURM)

    assert result.watchdog_job_id == "500"
    assert not [c for c in sched.submissions() if c[-1] == "generated-watchdog-batch.script"]
    assert [j for n, j in _records(batch / batch_log.LOG_NAME, "SUBMITTED") if n.startswith("watchdog-")] == ["500"]


def test_a_finished_watchdog_is_replaced(tmp_path, fake):
    xyz_files = _three_structures(tmp_path)[:1]
    batch = tmp_path / "batch"
    _seed_log(batch, ["watchdog-batch\tSUBMITTED\tjob_id=500\n"])
    sched = fake(live=set())  # 500 has exited

    result = orchestrate.submit_batch(xyz_files, batch, MODE, SLURM)

    sched.submitted("generated-watchdog-batch.script")
    assert result.watchdog_job_id not in (None, "500")


# ── xas-postprocess --submit ─────────────────────────────────────────────────


def test_postprocess_cli_submit_replaces_a_stale_under_gated_job(tmp_path, fake, monkeypatch, capsys):
    batch = tmp_path / "batch"
    log = _seed_log(batch, [
        "corvus-xas-CL1-quick\tSUBMITTED\tjob_id=101\n",
        "rerun-corvus-xas-CL2-quick\tSUBMITTED\tjob_id=102\n",
        "postprocess-batch\tSUBMITTED\tjob_id=301\n",   # gated on 101 only
    ])
    sched = fake(live={"101", "102", "301"})
    monkeypatch.setattr(sys, "argv", ["xas-postprocess", str(batch), "--submit", "--scheduler", "slurm"])

    assert postprocess_cli.main() == 0

    script = "generated-postprocess-batch.script"
    assert "--dependency=afterany:101:102" in sched.submitted(script)
    assert sched.index("sbatch", script) < sched.index("scancel", "301")
    assert ("postprocess-batch", "301") in _records(log, "REPLACED")
    new_id = _records(log, "SUBMITTED")[-1][1]
    assert f"Submitted postprocess job {new_id}" in capsys.readouterr().out
