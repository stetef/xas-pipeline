"""Regression tests for ``layout.quarantine_move`` (Phase 1: "quarantine moves nested good runs").

A pre-grouping batch has bare ``<id>/`` run dirs, and a mode added later nests
inside it as ``<id>/<id>-<mode>/`` (see :func:`layout.iter_id_dirs`). When the old
run failed, quarantine used to ``shutil.move`` the whole ``<id>/`` into
``failed-orca/`` -- dragging the healthy nested runs (and their CORVUS output)
with it. Now only the old run's own contents move.
"""

from __future__ import annotations

import sys

from xas_pipeline import layout
from xas_pipeline.stages import orca_check

GOOD_LOG = "...\n****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: 0 days 0 hours 1 minutes 2 seconds 0 msec\n"
CRASHED_LOG = "...\nORCA finished by error termination in SCF\n"


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _orca_run(run_dir, log_text):
    name = run_dir.name
    _write(run_dir / f"generated-{name}-orca.script", "#!/bin/bash\n")
    _write(run_dir / f"{name}.in", "! x\n")
    _write(run_dir / f"{name}-orca.log", log_text)
    return run_dir


def test_a_plain_run_dir_moves_whole(tmp_path):
    run_dir = _orca_run(tmp_path / "cluster1" / "cluster1-quick", CRASHED_LOG)
    _write(run_dir / "working-cluster1-quick" / "deep.txt", "x")

    dest = layout.quarantine_move(run_dir, tmp_path / "failed-orca")

    assert dest == tmp_path / "failed-orca" / "cluster1-quick"
    assert not run_dir.exists()
    assert (dest / "cluster1-quick-orca.log").is_file()
    assert (dest / "working-cluster1-quick" / "deep.txt").is_file()


def test_a_legacy_run_dir_keeps_its_nested_mode_runs(tmp_path):
    legacy = _orca_run(tmp_path / "cluster1", CRASHED_LOG)
    _write(legacy / "working-cluster1" / "scratch.txt", "x")
    nested = _orca_run(legacy / "cluster1-hopt-spring", GOOD_LOG)
    _write(nested / "Corvus3_cfavg_xas" / "spectrum.dat", "1 2\n")

    dest = layout.quarantine_move(legacy, tmp_path / "failed-orca")

    # The failed run's own files moved ...
    assert (dest / "cluster1-orca.log").is_file()
    assert (dest / "working-cluster1" / "scratch.txt").is_file()
    assert not (legacy / "cluster1-orca.log").exists()
    # ... the healthy nested run did not, and lost nothing.
    assert not (dest / "cluster1-hopt-spring").exists()
    assert (nested / "cluster1-hopt-spring-orca.log").read_text() == GOOD_LOG
    assert (nested / "Corvus3_cfavg_xas" / "spectrum.dat").is_file()
    # The scan still finds it where it was.
    assert nested in list(layout.iter_id_dirs(tmp_path))


def test_an_unrelated_subdir_is_not_mistaken_for_a_nested_run(tmp_path):
    """Only ``<name>-<known mode>`` children are spared; anything else is run content."""
    legacy = _orca_run(tmp_path / "cluster1", CRASHED_LOG)
    _write(legacy / "cluster1-archive" / "old.txt", "x")

    dest = layout.quarantine_move(legacy, tmp_path / "failed-orca")

    assert not legacy.exists()
    assert (dest / "cluster1-archive" / "old.txt").is_file()


def test_a_stale_quarantined_copy_is_replaced(tmp_path):
    _write(tmp_path / "failed-orca" / "cluster1-quick" / "stale.txt", "old")
    run_dir = _orca_run(tmp_path / "cluster1" / "cluster1-quick", CRASHED_LOG)

    dest = layout.quarantine_move(run_dir, tmp_path / "failed-orca")

    assert not (dest / "stale.txt").exists()
    assert (dest / "cluster1-quick-orca.log").is_file()


def test_orca_check_quarantines_a_failed_legacy_run_but_not_its_nested_good_run(tmp_path, monkeypatch, capsys):
    """End to end through the postprocess's ORCA check."""
    batch = tmp_path / "batch"
    legacy = _orca_run(batch / "cluster1", CRASHED_LOG)
    nested = _orca_run(legacy / "cluster1-hopt-spring", GOOD_LOG)

    monkeypatch.delenv("PIPELINE_SCHEDULER", raising=False)
    monkeypatch.setattr(sys, "argv", ["xas-check-orca", str(batch), "--output-dir", str(tmp_path / "report")])
    assert orca_check.main() == 0

    failed = batch / "failed-orca" / "cluster1"
    assert (failed / "cluster1-orca.log").is_file()
    assert not (failed / "cluster1-hopt-spring").exists()
    assert (nested / "cluster1-hopt-spring-orca.log").is_file()
    csv_text = (tmp_path / "report" / "batch-orca-compute-times.csv").read_text()
    assert "cluster1-hopt-spring," in csv_text
