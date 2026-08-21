"""Unit tests for modes that run no ORCA stage (``interp-raw``, ``opt-interp``).

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

from xas_pipeline import layout
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


@pytest.mark.parametrize("mode", sorted(layout.NO_ORCA_MODES))
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
    """The gate must not be so broad that it hides real ORCA runs."""
    batch_root = tmp_path / "batch-out"
    run_dir = layout.run_dir_for(batch_root, "2j6a_ZN_cluster1", "interp-hopt")
    run_dir.mkdir(parents=True)
    run_id = layout.run_id_for("2j6a_ZN_cluster1", "interp-hopt")
    (run_dir / f"generated-{run_id}-orca.script").write_text("#!/bin/bash\n", encoding="utf-8")

    assert orca_check.looks_like_run_dir(run_dir)
    assert run_dir in list(orca_check.find_run_dirs(batch_root))


FIXTURE_XYZ = (
    Path(__file__).resolve().parent.parent / "fixtures" / "xyz_files"
    / "2j6a_ZN_homo_d2.60_cluster1.xyz"
)


def _submit_calls_for(tmp_path, monkeypatch, flag):
    """Run the orchestrator for one mode, capturing every job it would submit.

    Submission is the one thing --no-submit cannot exercise, and it is where the
    ORCA dependency is either attached or not -- so patch the submit helper rather
    than skipping it.
    """
    import os
    import shutil
    import sys

    from xas_pipeline import orchestrate

    # The orchestrator shells out to prepare-orca as a bare "python", so that name
    # has to resolve. Under pytest the venv is usually not activated; the running
    # interpreter's own directory is the venv bin, so putting it on PATH gives the
    # subprocess the same interpreter as the test.
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")

    in_root = tmp_path / "xyz_files"
    in_root.mkdir()
    shutil.copy2(FIXTURE_XYZ, in_root / FIXTURE_XYZ.name)
    out_root = tmp_path / "batch-out"

    calls = []

    def fake_submit(script_path, cwd, scheduler, depend_afterok=None, depend_afterany=None):
        calls.append(
            {
                "script": Path(script_path).name,
                "afterok": list(depend_afterok) if depend_afterok else None,
                "afterany": list(depend_afterany) if depend_afterany else None,
            }
        )
        return f"job{len(calls)}"

    monkeypatch.setattr(orchestrate, "_submit_job", fake_submit)
    monkeypatch.setattr(orchestrate, "_check_executable", lambda *a, **k: None)

    argv = [
        "xas-run-batch",
        str(in_root),
        "--out-dir",
        str(out_root),
        "--scheduler",
        "slurm",
        "--no-postprocess",
    ]
    if flag:
        argv.append(flag)
    monkeypatch.setattr("sys.argv", argv)

    assert orchestrate.main() == 0
    return calls


def test_no_orca_mode_submits_corvus_with_no_dependency(tmp_path, monkeypatch):
    """There is no ORCA job id to wait on, so no --dependency may be attached.

    Passing one would make the CORVUS job wait on a job that was never submitted;
    depending on the scheduler that is either an immediate rejection or a job that
    sits pending forever.
    """
    calls = _submit_calls_for(tmp_path, monkeypatch, "--interp-raw")

    assert len(calls) == 1, calls
    assert "corvus" in calls[0]["script"]
    assert calls[0]["afterok"] is None
    assert calls[0]["afterany"] is None


def test_orca_running_mode_still_chains_corvus_after_orca(tmp_path, monkeypatch):
    """The counterpart: interp-hopt does run ORCA, so the chain must stay."""
    calls = _submit_calls_for(tmp_path, monkeypatch, "--interp")

    assert len(calls) == 2, calls
    orca, corvus = calls
    assert "orca" in orca["script"]
    assert "corvus" in corvus["script"]
    assert corvus["afterok"] == ["job1"]


@pytest.mark.parametrize("mode", sorted(layout.NO_ORCA_MODES))
def test_corvus_wrapper_interpolates_the_hessian_for_no_orca_modes(tmp_path, mode):
    """No ORCA means no ORCA Hessian, so the wrapper has to build one."""
    from xas_pipeline import orchestrate

    run_id = layout.run_id_for("2j6a_ZN_cluster1", mode)
    script = tmp_path / "wrapper.script"
    orchestrate._write_corvus_wrapper_script(
        script,
        tmp_path,
        run_id,
        "slurm",
        corvus_mode="xas",
        optimization_mode=mode,
    )

    executable = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "echo "))
    ]
    assert f"OPTIMIZATION_MODE={mode}" in executable
    assert any("stages.interp_hessian" in line for line in executable)
