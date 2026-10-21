"""Regression tests for the stage-level fixes of the 2026-10 clean-up."""

from __future__ import annotations

from pathlib import Path

from xas_pipeline import diagnosis, layout
from xas_pipeline.stages import cleanup, count_imag_freq, download, feff_process


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_keep_reruns_spares_scratch_inside_a_nested_rerun_snapshot(tmp_path):
    feff = tmp_path / "R1" / "working-R1" / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    live_bin = _write(feff / "xanes" / "fort.bin")
    snapshot_bin = _write(feff / "xanes.rerun-1" / "fort.bin")
    _write(tmp_path / "R1" / "output-R1" / "xas-R1.dat")

    cleaner = cleanup.Cleaner(execute=True, force=False, do_orca=True, do_feff=True, do_reruns=False)
    cleaner.clean_cluster(tmp_path / "R1")

    assert not live_bin.exists()
    assert snapshot_bin.exists()


def test_download_dry_run_changes_nothing(tmp_path):
    batch = tmp_path / "batch"
    _write(batch / "R-ok" / "output-R-ok" / "xas-R-ok.dat")
    _write(batch / "R-bad" / "working-R-bad" / "note.txt")
    _write(batch / layout.FAILED_CORVUS_MANIFEST, "R-bad\n")
    dest = tmp_path / "dest"

    summary = download.prepare_downloads(batch, dest, dry_run=True)

    assert (summary.copied, summary.quarantined) == (1, 1)
    assert not dest.exists()
    assert (batch / "R-bad").is_dir() and not (batch / "failed-corvus").exists()


def test_failed_corvus_manifest_round_trip(tmp_path):
    assert layout.read_failed_corvus_ids(tmp_path) == []
    _write(tmp_path / layout.FAILED_CORVUS_MANIFEST, "b\n\n a \n")
    assert layout.read_failed_corvus_ids(tmp_path) == ["b", "a"]


def test_imag_freq_count_reads_the_first_out_file_by_name(tmp_path):
    working = tmp_path / "R1" / "working-R1"
    _write(working / "corvus-b.out", "")
    _write(working / "corvus-a.out", f"{count_imag_freq.MARKER}\n\0binary\n{count_imag_freq.MARKER}\n")
    assert count_imag_freq.count_imaginary_frequencies(tmp_path) == {"R1": 2}


def test_gbw_of_a_nested_mode_run_is_not_this_runs_restart_guess(tmp_path):
    run = tmp_path / "CL1"
    _write(run / "CL1-orca.log", "GEOMETRY OPTIMIZATION CYCLE\n")
    _write(run / "CL1-hopt-spring" / "CL1-hopt-spring.gbw", "orbitals")
    assert diagnosis.diagnose(run).evidence.gbw_present is False

    _write(run / "working-CL1" / "CL1.gbw", "orbitals")
    assert diagnosis.diagnose(run).evidence.gbw_present is True


def test_process_batch_writes_an_empty_manifest_when_nothing_is_processable(tmp_path):
    (tmp_path / "not-a-run").mkdir()
    assert feff_process.process_batch(tmp_path, record_in_batch_log=False) == []
    assert (tmp_path / layout.FAILED_CORVUS_MANIFEST).read_text(encoding="utf-8") == ""


def test_an_empty_id_selection_selects_no_runs(tmp_path):
    (tmp_path / "A").mkdir()
    _write(tmp_path / "A" / "A-orca.log")
    assert list(layout.iter_id_dirs(tmp_path, only_ids=set())) == []
    assert [p.name for p in layout.iter_id_dirs(tmp_path, only_ids=None)] == ["A"]


def test_stage_times_finds_grouped_and_split_runs(tmp_path):
    """The old heredoc read only first-level dirs and a `-corvus.timing` name nobody writes."""
    from xas_pipeline.stages import stage_times

    flat = tmp_path / "A" / "A-hopt-spring"
    _write(flat / "A-hopt-spring-orca.timing", "start_epoch=1\nelapsed_seconds=40\nexit_code=0\n")
    _write(flat / "A-hopt-spring-corvus-xas.timing", "elapsed_seconds=7\nexit_code=0\n")
    split = tmp_path / "B" / "working-B"
    _write(split / "B-orca.timing", "elapsed_seconds=99\nexit_code=1\n")
    _write(tmp_path / "B" / "output-B" / "xas-B.dat")

    rows = (stage_times.write_stage_times(tmp_path)).read_text(encoding="utf-8").splitlines()
    assert rows == [
        ",".join(stage_times.COLUMNS),
        "A-hopt-spring,40,0,7,0",
        "B,99,1,,",
    ]
