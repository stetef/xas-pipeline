"""Tests for postprocess gating across several submissions into one batch root.

Two workflows have to hold:

*concurrent* -- submit mode A, then mode B a minute later, both still running.
The postprocess must wait for both, and an early one must not damage a live run.

*serial* -- add a mode to a batch that finished months ago. Its old job ids are
long purged from the scheduler, so they must not end up in a dependency, and its
completed runs must still be seen alongside the new one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xas_pipeline import batch_log as bl
from xas_pipeline import layout, orchestrate
from xas_pipeline.stages import orca_check


# ── batch-jobs.log as the batch's memory ─────────────────────────────────────


def _write_log(path, lines):
    path.write_text(
        "# header comment\n\njob_name\tstatus\tjob_id\n" + "".join(lines), encoding="utf-8"
    )
    return path


def test_reads_back_submitted_job_ids(tmp_path):
    log = _write_log(
        tmp_path / "batch-jobs.log",
        [
            "prepare-orca\tSUCCEEDED\n",
            "orca-CL1-ca-fixed\tSUBMITTED\tjob_id=100\n",
            "corvus-xas-CL1-ca-fixed\tSUBMITTED\tjob_id=101\n",
            "corvus-xas-CL2-ca-fixed\tSUBMITTED\tjob_id=102\n",
            "postprocess-batch\tSUBMITTED\tjob_id=103\n",
        ],
    )
    assert bl.submitted_job_ids(log, "corvus-") == ["101", "102"]
    assert bl.submitted_job_ids(log, "postprocess-") == ["103"]
    assert bl.submitted_job_ids(log, "orca-") == ["100"]


def test_ignores_outcome_lines_and_dedupes(tmp_path):
    log = _write_log(
        tmp_path / "batch-jobs.log",
        [
            "corvus-xas-CL1\tSUBMITTED\tjob_id=101\n",
            "corvus-xas-CL1\tSUBMITTED\tjob_id=101\n",  # re-submitted, same id
            "# --- CORVUS outcomes ---\n",
            "corvus-xas-CL1\tOK\n",
            "corvus-xas-CL2\tSUBMIT_FAILED\n",
        ],
    )
    assert bl.submitted_job_ids(log, "corvus-") == ["101"]


def test_missing_log_is_not_an_error(tmp_path):
    assert bl.submitted_job_ids(tmp_path / "nope.log", "corvus-") == []


# ── dependency discovery ─────────────────────────────────────────────────────


class _FakeScheduler:
    """Stands in for squeue/qstat: only these ids are still known."""

    def __init__(self, active):
        self.active = list(active)
        self.queried = None

    def active_job_ids(self, job_ids):
        self.queried = list(job_ids)
        return [j for j in job_ids if j in self.active]


@pytest.fixture
def fake_scheduler(monkeypatch):
    def install(active):
        sched = _FakeScheduler(active)
        monkeypatch.setattr(orchestrate._sched, "get_scheduler", lambda name: sched)
        return sched

    return install


def test_outstanding_corvus_excludes_finished_and_just_submitted(tmp_path, fake_scheduler):
    log = _write_log(
        tmp_path / "batch-jobs.log",
        [
            "corvus-xas-CL1-ca-fixed\tSUBMITTED\tjob_id=101\n",  # still running
            "corvus-xas-CL2-ca-fixed\tSUBMITTED\tjob_id=102\n",  # already finished
            "corvus-xas-CL1-interp\tSUBMITTED\tjob_id=201\n",    # submitted just now
        ],
    )
    sched = fake_scheduler(active=["101", "201"])

    outstanding = orchestrate.outstanding_corvus_job_ids(log, "slurm", exclude=["201"])

    assert outstanding == ["101"]
    assert "201" not in sched.queried, "the current invocation's own jobs are not re-queried"


def test_serial_case_drops_purged_job_ids(tmp_path, fake_scheduler):
    """A months-old batch: its ids are gone from the scheduler, so no dependency.

    Depending on a purged id makes the submission fail outright, which is how
    adding a mode to an old batch would break.
    """
    log = _write_log(
        tmp_path / "batch-jobs.log",
        ["corvus-xas-CL1\tSUBMITTED\tjob_id=9\n", "corvus-xas-CL2\tSUBMITTED\tjob_id=10\n"],
    )
    fake_scheduler(active=[])
    assert orchestrate.outstanding_corvus_job_ids(log, "slurm") == []


def test_pending_postprocess_found_for_replacement(tmp_path, fake_scheduler):
    log = _write_log(
        tmp_path / "batch-jobs.log",
        [
            "postprocess-batch\tSUBMITTED\tjob_id=300\n",  # already ran
            "postprocess-batch\tSUBMITTED\tjob_id=301\n",  # still pending
        ],
    )
    fake_scheduler(active=["301"])
    assert orchestrate.pending_postprocess_job_ids(log, "slurm") == ["301"]


# ── the in-flight guard ──────────────────────────────────────────────────────


def _run_dir_with_timing(tmp_path, name, timing_body, *, log_text="partial output\n"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / f"{name}-orca.log").write_text(log_text, encoding="utf-8")
    if timing_body is not None:
        (run_dir / f"{name}-orca.timing").write_text(timing_body, encoding="utf-8")
    return run_dir


def test_running_job_is_in_flight(tmp_path):
    """start_epoch written, no exit_code yet -> ORCA is still going."""
    run_dir = _run_dir_with_timing(tmp_path, "CL1-ca-fixed", "start_epoch=1\nhostname=n1\n")
    assert orca_check.orca_run_is_in_flight(run_dir)


def test_finished_job_is_not_in_flight(tmp_path):
    run_dir = _run_dir_with_timing(
        tmp_path, "CL1-ca-fixed", "start_epoch=1\nexit_code=0\nterminated_normally=true\n"
    )
    assert not orca_check.orca_run_is_in_flight(run_dir)


def test_crashed_job_that_wrote_an_exit_code_is_classified(tmp_path):
    """exit_code present but non-zero: finished, and should be judged normally."""
    run_dir = _run_dir_with_timing(
        tmp_path, "CL1-ca-fixed", "start_epoch=1\nexit_code=1\nterminated_normally=false\n"
    )
    assert not orca_check.orca_run_is_in_flight(run_dir)
    ok, _reason = orca_check.classify_orca_run(run_dir)
    assert not ok


def test_batch_without_timing_files_is_still_classified(tmp_path):
    """Batches predating the .timing sidecar must not all look in-flight."""
    run_dir = _run_dir_with_timing(tmp_path, "CL1", None)
    assert not orca_check.orca_run_is_in_flight(run_dir)


def test_in_flight_run_is_not_quarantined(tmp_path, capsys):
    """The whole point: an early postprocess must leave a live run alone."""
    batch = tmp_path / "batch"
    batch.mkdir()
    live = _run_dir_with_timing(batch, "CL1-ca-fixed", "start_epoch=1\n")
    done = _run_dir_with_timing(
        batch,
        "CL2-interp",
        "start_epoch=1\nexit_code=0\n",
        log_text="****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: 0 days 0 hours\n",
    )

    scanned = list(orca_check.find_run_dirs(batch))
    assert set(scanned) == {live, done}
    assert orca_check.orca_run_is_in_flight(live)
    assert not orca_check.orca_run_is_in_flight(done)


# ── serial layout: an existing run dir that gains a mode run ─────────────────


def test_legacy_run_dir_hosting_a_mode_run_yields_both(tmp_path):
    """`--interp` pointed at a finished batch: original + new run must both scan."""
    batch = tmp_path / "first-set"
    legacy = batch / "CL1"
    (legacy / "working-CL1").mkdir(parents=True)
    (legacy / "output-CL1").mkdir(parents=True)
    added = legacy / "CL1-interp"
    added.mkdir()
    (added / "CL1-interp.xyz").write_text("", encoding="utf-8")

    assert layout.is_own_run_dir(legacy), "the original run is still a run"
    assert layout.nested_mode_run_dirs(legacy) == [added]
    assert [p.name for p in layout.iter_id_dirs(batch)] == ["CL1", "CL1-interp"]


def test_pure_group_dir_does_not_yield_itself(tmp_path):
    batch = tmp_path / "batch"
    group = batch / "CL1"
    for mode in ("ca-fixed", "interp"):
        run = group / f"CL1-{mode}"
        run.mkdir(parents=True)
        (run / f"CL1-{mode}.in").write_text("", encoding="utf-8")

    assert not layout.is_own_run_dir(group)
    assert [p.name for p in layout.iter_id_dirs(batch)] == ["CL1-ca-fixed", "CL1-interp"]


def test_run_dir_holding_only_subdirs_is_still_a_run(tmp_path):
    """A CORVUS run dir can legitimately have no top-level files yet."""
    batch = tmp_path / "batch"
    run = batch / "CL1"
    (run / "Corvus3_cfavg_xas").mkdir(parents=True)

    assert layout.is_own_run_dir(run)
    assert [p.name for p in layout.iter_id_dirs(batch)] == ["CL1"]


# ── deliverable geometry for non-optimizing modes ────────────────────────────


def test_geometry_is_shipped_for_a_mode_that_does_not_optimize(tmp_path):
    """--interp runs a single point, so ORCA writes no <run_id>.xyz.

    The only geometry is the input copy, named for the *structure* rather than
    the run. Without a fallback the spectrum ships with no structure at all,
    which is exactly the provenance needed to compare it against another mode.
    """
    from xas_pipeline.stages import feff_process

    run_id = "CL1-interp"
    system_dir = tmp_path / run_id
    working = system_dir / f"working-{run_id}"
    output = system_dir / f"output-{run_id}"
    working.mkdir(parents=True)
    output.mkdir(parents=True)

    # What an interp run actually leaves behind: no "<run_id>.xyz".
    (working / "CL1.xyz").write_text("1\ncomment\nZn 0.0 0.0 0.0\n", encoding="utf-8")
    (working / f"{run_id}_clean.xyz").write_text("1\nc\nZn 0 0 0\n", encoding="utf-8")
    (working / f"corvus-begin-{run_id}.xyz").write_text("1\nc\nZn 0 0 0\n", encoding="utf-8")

    feff_process._copy_xyz(system_dir, working, output, run_id)

    shipped = output / f"{run_id}.xyz"
    assert shipped.is_file(), "no geometry shipped with the interp spectrum"
    # The input copy, not the cleaned or CORVUS-standardized rewrites.
    assert shipped.read_text(encoding="utf-8") == (working / "CL1.xyz").read_text(encoding="utf-8")


def test_optimized_geometry_still_preferred(tmp_path):
    from xas_pipeline.stages import feff_process

    run_id = "CL1-ca-fixed"
    system_dir = tmp_path / run_id
    working = system_dir / f"working-{run_id}"
    output = system_dir / f"output-{run_id}"
    working.mkdir(parents=True)
    output.mkdir(parents=True)
    (working / f"{run_id}.xyz").write_text("1\noptimized\nZn 1 1 1\n", encoding="utf-8")
    (working / "CL1.xyz").write_text("1\ninput\nZn 0 0 0\n", encoding="utf-8")

    feff_process._copy_xyz(system_dir, working, output, run_id)
    assert "optimized" in (output / f"{run_id}.xyz").read_text(encoding="utf-8")


# ── dym2feffinp filename-length guard ────────────────────────────────────────


def test_dym2feffinp_runs_on_short_scratch_names(tmp_path, monkeypatch):
    """Long run ids must not reach dym2feffinp.

    It holds filenames in a fixed-length buffer and silently writes nothing past
    ~50 characters while still exiting 0, so the failure surfaced minutes later
    as a missing centered DYM with no indication of the cause.
    """
    from xas_pipeline.stages import corvus_prep

    run_dir = tmp_path
    long_id = "3qwp_ZN_homo_d2.60_cluster3_altlocLIGA-interp"
    dym = run_dir / f"{long_id}.dym"
    dym.write_text("dym", encoding="utf-8")
    feff_dym = run_dir / f"corvus-{long_id}.dym"
    assert len(feff_dym.name) > corvus_prep.DYM2FEFFINP_MAX_FILENAME

    seen = {}

    def fake_run(cmd, check, cwd):
        seen["out"] = cmd[cmd.index("--d") + 1]
        seen["in"] = cmd[-1]
        (Path(cwd) / seen["out"]).write_text("centered", encoding="utf-8")
        return None

    monkeypatch.setattr(corvus_prep.subprocess, "run", fake_run)
    corvus_prep._run_dym2feffinp("dym2feffinp", run_dir, dym, feff_dym, 1)

    assert len(seen["out"]) <= corvus_prep.DYM2FEFFINP_MAX_FILENAME
    assert len(seen["in"]) <= corvus_prep.DYM2FEFFINP_MAX_FILENAME
    assert feff_dym.read_text(encoding="utf-8") == "centered"
    assert not (run_dir / corvus_prep._DYM_SCRATCH_IN).exists()


def test_silent_dym2feffinp_failure_raises(tmp_path, monkeypatch):
    """Exit 0 with no output file is the actual failure mode; it must not pass."""
    from xas_pipeline.stages import corvus_prep

    dym = tmp_path / "x.dym"
    dym.write_text("dym", encoding="utf-8")
    monkeypatch.setattr(corvus_prep.subprocess, "run", lambda cmd, check, cwd: None)

    with pytest.raises(RuntimeError, match="produced no centered DYM"):
        corvus_prep._run_dym2feffinp("dym2feffinp", tmp_path, dym, tmp_path / "out.dym", 1)
