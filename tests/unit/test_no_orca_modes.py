"""Unit tests for modes that run no ORCA stage (``asis-spring``, ``caopt-spring``).

Two invariants keep such a run from being mistaken for a broken ORCA run:

1. The ORCA convergence scan must not see it at all. It has no log and no
   generated ORCA script, and ``classify_orca_run`` treats a missing log as a
   failure -- so if the scan picked the dir up it would quarantine a perfectly
   good run into ``failed-orca/`` on the first postprocess pass.
2. The orchestrator must submit its CORVUS job with no ORCA dependency, since
   there is no ORCA job id to wait on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xas_pipeline import layout, modes, scheduler
from xas_pipeline.stages import orca_check


def _no_orca_run_dir(batch_root: Path, id_name: str, mode: str) -> Path:
    """A run dir as prepare-orca leaves it for a mode with no ORCA stage."""
    run_id = layout.run_id_for(id_name, mode)
    run_dir = layout.run_dir_for(batch_root, id_name, mode)
    run_dir.mkdir(parents=True)
    # What the scaffolding actually writes: the input copy, the cleaned geometry,
    # the comments sidecar, and the geometry of record. No .in, no job script.
    (run_dir / f"{id_name}.xyz").write_text("1\ncomment\nZn 0.0 0.0 0.0\n", encoding="utf-8")
    (run_dir / f"{run_id}_clean.xyz").write_text("1\ncomment\nZn 0.0 0.0 0.0\n", encoding="utf-8")
    (run_dir / f"{run_id}.xyz").write_text("1\ncomment\nZn 0.0 0.0 0.0\n", encoding="utf-8")
    (run_dir / f"{run_id}_comments.txt").write_text("Atom 0: # RES=ZN\n", encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("mode", sorted(modes.NO_ORCA_MODES))
def test_orca_scan_ignores_a_run_with_no_orca_stage(tmp_path, mode):
    """It has no ORCA log to judge, so it must not be judged.

    ``classify_orca_run`` returns "ORCA log not found ... likely never started"
    for a dir with no log, which would move the run into failed-orca/. The gate
    that prevents it is ``looks_like_run_dir``.
    """
    batch_root = tmp_path / "batch-out"
    run_dir = _no_orca_run_dir(batch_root, "2j6a_ZN_cluster1", mode)

    assert orca_check.find_orca_log(run_dir) is None
    assert not orca_check.looks_like_run_dir(run_dir)
    assert run_dir not in list(orca_check.find_run_dirs(batch_root))

    # And to be explicit about what the gate is protecting against:
    ok, reason = orca_check.classify_orca_run(run_dir)
    assert not ok and "log not found" in reason


def test_orca_scan_still_sees_a_mode_that_does_run_orca(tmp_path):
    """The gate must not be so broad that it hides real ORCA runs (hopt-spring runs ORCA)."""
    batch_root = tmp_path / "batch-out"
    run_dir = layout.run_dir_for(batch_root, "2j6a_ZN_cluster1", "hopt-spring")
    run_dir.mkdir(parents=True)
    run_id = layout.run_id_for("2j6a_ZN_cluster1", "hopt-spring")
    (run_dir / f"generated-{run_id}-orca.script").write_text("#!/bin/bash\n", encoding="utf-8")

    assert orca_check.looks_like_run_dir(run_dir)
    assert run_dir in list(orca_check.find_run_dirs(batch_root))


FIXTURE_XYZ = (
    Path(__file__).resolve().parent.parent / "fixtures" / "xyz_files"
    / "2j6a_ZN_homo_d2.60_cluster1.xyz"
)


class _Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def _submit_calls_for(tmp_path, monkeypatch, flag):
    """Run the orchestrator for one mode, capturing every job it would submit.

    Submission is the one thing --no-submit cannot exercise, and it is where the
    ORCA dependency is either attached or not -- so replace the scheduler seam
    (``scheduler.run_command``) rather than skipping submission.
    """
    import shutil

    from xas_pipeline import orchestrate

    in_root = tmp_path / "xyz_files"
    in_root.mkdir()
    shutil.copy2(FIXTURE_XYZ, in_root / FIXTURE_XYZ.name)
    out_root = tmp_path / "batch-out"

    calls = []

    def fake_run_command(cmd, cwd=None):
        if cmd[0] != "sbatch":
            return _Done("")  # queries: nothing is queued
        deps = [a for a in cmd if a.startswith("--dependency=")]
        calls.append({"script": cmd[-1], "deps": deps})
        return _Done(f"Submitted batch job {100 + len(calls)}\n")

    monkeypatch.setattr(scheduler, "run_command", fake_run_command)
    monkeypatch.setattr(orchestrate, "check_executable", lambda *a, **k: None)

    argv = [
        "xas-run-batch",
        str(in_root),
        "--out-dir",
        str(out_root),
        "--scheduler",
        "slurm",
        "--no-postprocess",
        # This helper counts only the ORCA/CORVUS submissions; the hang-watchdog
        # companion job is a separate concern (covered in tests/cli/test_watchdog.py).
        "--no-watchdog",
    ]
    if flag:
        argv.append(flag)
    monkeypatch.setattr("sys.argv", argv)

    assert orchestrate.main() == 0
    return calls


@pytest.mark.parametrize("flag", ["--interp-raw", "--caopt-spring"])
def test_no_orca_mode_submits_corvus_with_no_dependency(tmp_path, monkeypatch, flag):
    """There is no ORCA job id to wait on, so no --dependency may be attached.

    Passing one would make the CORVUS job wait on a job that was never submitted;
    depending on the scheduler that is either an immediate rejection or a job that
    sits pending forever.
    """
    calls = _submit_calls_for(tmp_path, monkeypatch, flag)

    assert len(calls) == 1, calls
    assert "corvus" in calls[0]["script"]
    assert calls[0]["deps"] == []


def test_orca_running_mode_still_chains_corvus_after_orca(tmp_path, monkeypatch):
    """The counterpart: hopt-spring does run ORCA, so the chain must stay."""
    calls = _submit_calls_for(tmp_path, monkeypatch, "--interp")

    assert len(calls) == 2, calls
    orca, corvus = calls
    assert "orca" in orca["script"]
    assert "corvus" in corvus["script"]
    assert corvus["deps"] == ["--dependency=afterok:101"]


@pytest.mark.parametrize("sched_name", ["slurm", "pbs"])
@pytest.mark.parametrize("mode", sorted(modes.NO_ORCA_MODES))
def test_corvus_wrapper_interpolates_the_hessian_for_no_orca_modes(tmp_path, mode, sched_name):
    """No ORCA means no ORCA Hessian, so the wrapper has to build one."""
    from xas_pipeline import jobscripts

    run_id = layout.run_id_for("2j6a_ZN_cluster1", mode)
    script = jobscripts.corvus_wrapper(
        scheduler.get_scheduler(sched_name), tmp_path, run_id, mode=mode,
        dest=tmp_path / "wrapper.script",
    )

    executable = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "echo "))
    ]
    assert f"OPTIMIZATION_MODE={mode}" in executable
    assert any("stages.interp_hessian" in line for line in executable)


# ── The ORCA scan: group dirs and queued runs (Phase 1 regressions) ──────────


def _orca_run(run_dir: Path, *, log: str | None = None) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / f"generated-{run_dir.name}-orca.script").write_text("#!/bin/bash\n", encoding="utf-8")
    if log is not None:
        (run_dir / f"{run_dir.name}-orca.log").write_text(log, encoding="utf-8")
    return run_dir


def test_a_group_dir_is_not_an_orca_run_even_with_a_stray_file(tmp_path):
    """Regression: the scan used ``rglob``, so a group dir saw its children's logs.

    A group dir holding anything besides its ``<id>-<mode>`` runs (here a copy of
    the input XYZ) counts as a possible run dir in :func:`layout.iter_id_dirs`;
    with rglob it then "had" an ORCA log and script -- its child's -- and a
    failure judged on that log would quarantine the whole group.
    """
    group = tmp_path / "cluster1"
    child = _orca_run(group / "cluster1-caopt-anfreq", log="...\n")
    (group / "cluster1.xyz").write_text("1\n\nZn 0 0 0\n", encoding="utf-8")
    (child / "working-cluster1-caopt-anfreq").mkdir()

    assert orca_check.find_orca_log(group) is None
    assert not orca_check.looks_like_run_dir(group)
    assert orca_check.find_run_dirs(tmp_path) == [child]


def test_a_run_dir_finds_its_own_log_flat_or_split_but_never_a_nested_runs(tmp_path):
    legacy = tmp_path / "cluster1"
    _orca_run(legacy / "cluster1-hopt-spring", log="nested\n")
    (legacy / "working-cluster1").mkdir()
    (legacy / "working-cluster1" / "cluster1-orca.log").write_text("own\n", encoding="utf-8")

    assert orca_check.find_orca_log(legacy) == legacy / "working-cluster1" / "cluster1-orca.log"
    (legacy / "working-cluster1" / "cluster1-orca.log").unlink()
    assert orca_check.find_orca_log(legacy) is None


def _batch_with_one_unstarted_run(tmp_path: Path) -> tuple[Path, Path]:
    batch = tmp_path / "batch"
    run_dir = _orca_run(batch / "cluster1" / "cluster1-caopt-anfreq")
    (batch / "batch-jobs.log").write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-cluster1-caopt-anfreq\tSUBMITTED\tjob_id=4242\n",
        encoding="utf-8",
    )
    return batch, run_dir


def _run_orca_check(monkeypatch, batch: Path, *extra: str) -> int:
    monkeypatch.setattr(
        "sys.argv", ["xas-check-orca", str(batch), "--output-dir", str(batch), "--no-batch-log", *extra]
    )
    return orca_check.main()


def test_a_queued_run_is_skipped_not_quarantined_as_never_started(tmp_path, monkeypatch):
    """Regression: no log yet looked like "never started" and went to failed-orca/."""
    batch, run_dir = _batch_with_one_unstarted_run(tmp_path)
    queries = []

    def fake_run_command(cmd, cwd=None):
        queries.append(cmd)
        return _Done("4242\n" if cmd[0] == "squeue" else "")

    monkeypatch.setattr(scheduler, "run_command", fake_run_command)
    assert _run_orca_check(monkeypatch, batch, "--scheduler", "slurm") == 0

    assert run_dir.is_dir()
    assert not (batch / "failed-orca").exists()
    assert "cluster1-caopt-anfreq: SKIPPED (ORCA job still queued)" in (
        batch / "orca-convergence-report.log"
    ).read_text()
    assert queries and queries[0][0] == "squeue" and "4242" in queries[0][-1]


def test_a_queued_run_is_skipped_under_pbs_too(tmp_path, monkeypatch):
    batch, run_dir = _batch_with_one_unstarted_run(tmp_path)
    monkeypatch.setattr(
        scheduler, "run_command",
        lambda cmd, cwd=None: _Done("Job Id: 4242\n    job_state = Q\n" if cmd[0] == "qstat" else ""),
    )
    monkeypatch.setenv("PIPELINE_SCHEDULER", "pbs")  # the default when --scheduler is omitted
    assert _run_orca_check(monkeypatch, batch) == 0
    assert run_dir.is_dir()


def test_a_run_the_scheduler_no_longer_lists_is_still_judged(tmp_path, monkeypatch):
    """The counterpart: a job that left the queue without a log never started."""
    batch, run_dir = _batch_with_one_unstarted_run(tmp_path)
    monkeypatch.setattr(scheduler, "run_command", lambda cmd, cwd=None: _Done(""))
    assert _run_orca_check(monkeypatch, batch, "--scheduler", "slurm") == 0

    assert not run_dir.exists()
    assert (batch / "failed-orca" / "cluster1-caopt-anfreq").is_dir()


def test_without_a_scheduler_nothing_is_queried(tmp_path, monkeypatch):
    """Off the cluster (no --scheduler, no $PIPELINE_SCHEDULER) the check cannot ask,
    so it judges as before rather than failing on a missing squeue/qstat."""
    batch, _run_dir = _batch_with_one_unstarted_run(tmp_path)
    monkeypatch.delenv("PIPELINE_SCHEDULER", raising=False)

    def boom(cmd, cwd=None):
        raise AssertionError(f"unexpected scheduler call {cmd}")

    monkeypatch.setattr(scheduler, "run_command", boom)
    assert _run_orca_check(monkeypatch, batch) == 0
    assert (batch / "failed-orca" / "cluster1-caopt-anfreq").is_dir()


def test_a_queued_rerun_is_not_judged_on_the_previous_attempts_log(tmp_path, monkeypatch):
    """xas-rerun-orca resubmits in place, leaving the failed attempt's log and
    timing (with its exit_code) beside the queued rerun. Judging that stale log
    would move the run dir out from under the queued job."""
    batch, run_dir = _batch_with_one_unstarted_run(tmp_path)
    (run_dir / f"{run_dir.name}-orca.log").write_text("SCF NOT CONVERGED\n", encoding="utf-8")
    (run_dir / f"{run_dir.name}-orca.timing").write_text("start_epoch=1\nexit_code=1\n", encoding="utf-8")
    with (batch / "batch-jobs.log").open("a", encoding="utf-8") as handle:
        handle.write("orca-rerun1-cluster1-caopt-anfreq\tSUBMITTED\tjob_id=4343\n")
    monkeypatch.setattr(
        scheduler, "run_command", lambda cmd, cwd=None: _Done("4343\n" if cmd[0] == "squeue" else "")
    )
    assert _run_orca_check(monkeypatch, batch, "--scheduler", "slurm") == 0

    assert run_dir.is_dir()
    assert not (batch / "failed-orca").exists()
