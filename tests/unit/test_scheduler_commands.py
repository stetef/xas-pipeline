"""Scheduler.submit / cancel / cpu_seconds through the run_command seam."""

from __future__ import annotations

import subprocess

import pytest

from xas_pipeline import scheduler


class Recorder:
    def __init__(self, responses):
        #: program -> (returncode, stdout) or an exception to raise
        self.responses = responses
        self.calls = []

    def __call__(self, cmd, cwd=None):
        self.calls.append(([str(c) for c in cmd], cwd))
        response = self.responses[cmd[0]]
        if isinstance(response, Exception):
            raise response
        code, out = response
        return subprocess.CompletedProcess(cmd, code, out, "boom" if code else "")


@pytest.fixture
def seam(monkeypatch):
    def install(**responses):
        rec = Recorder(responses)
        monkeypatch.setattr(scheduler, "run_command", rec)
        return rec

    return install


def test_slurm_submit_argv_and_cwd(tmp_path, seam):
    rec = seam(sbatch=(0, "Submitted batch job 77\n"))
    job = scheduler.get_scheduler("slurm").submit(
        tmp_path / "job.script", tmp_path, afterok=["1", "2"], extra=["--exclude=n1"]
    )
    assert job == "77"
    assert rec.calls == [(["sbatch", "--dependency=afterok:1:2", "--exclude=n1", "job.script"], tmp_path)]


def test_pbs_submit_afterany(tmp_path, seam):
    rec = seam(qsub=(0, "88.server\n"))
    assert scheduler.get_scheduler("pbs").submit(tmp_path / "j.script", tmp_path, afterany=[5]) == "88"
    assert rec.calls[0][0] == ["qsub", "-W", "depend=afterany:5", "j.script"]


def test_afterok_takes_precedence_and_no_deps_means_no_flag(tmp_path, seam):
    rec = seam(sbatch=(0, "Submitted batch job 1\n"))
    slurm = scheduler.get_scheduler("slurm")
    slurm.submit(tmp_path / "a", tmp_path, afterok=["9"], afterany=["8"])
    slurm.submit(tmp_path / "b", tmp_path)
    assert rec.calls[0][0] == ["sbatch", "--dependency=afterok:9", "a"]
    assert rec.calls[1][0] == ["sbatch", "b"]


def test_refused_submission_raises(tmp_path, seam):
    seam(sbatch=(1, ""))
    with pytest.raises(RuntimeError, match="sbatch failed"):
        scheduler.get_scheduler("slurm").submit(tmp_path / "a", tmp_path)


def test_cancel_reports_the_result_and_never_raises(seam):
    slurm = scheduler.get_scheduler("slurm")
    seam(scancel=(0, ""))
    assert slurm.cancel("5") is True
    seam(scancel=(1, ""))
    assert slurm.cancel("5") is False
    seam(scancel=OSError("no scancel"))
    assert slurm.cancel("5") is False


def test_running_slurm_job_cpu_comes_from_sstat(seam):
    """sacct TotalCPU reads 0 until a step ends; sstat has the live figure."""
    rec = seam(sstat=(0, "01:00:05\n"), sacct=(0, "00:00:00\n"))
    assert scheduler.get_scheduler("slurm").cpu_seconds("42") == 3605
    assert [c[0][0] for c in rec.calls] == ["sstat"]


def test_sacct_is_the_fallback_when_sstat_is_silent(seam):
    seam(sstat=(0, ""), sacct=(0, "1-00:00:00\n"))
    assert scheduler.get_scheduler("slurm").cpu_seconds("42") == 86400
