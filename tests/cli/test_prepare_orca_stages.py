"""CLI tests for the staged ORCA run (``xas-prepare-orca --pre``).

What has to hold for a chain to actually work on a cluster, and what these cover:

* each stage gets its own ``<id>-<mode>`` run dir and its own job script;
* stage N+1's ``*xyzfile`` points at stage N's *optimized* output
  (``<stage_N_run_id>.xyz``), by absolute path -- the job script runs ORCA from a
  scratch directory, so a relative path would not resolve;
* stage N+1 is submitted ``afterok`` on stage N, because that output does not
  exist at prepare time;
* the sidecar that lets the auto-rerun re-chain a dropped stage is written.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from xas_pipeline import layout

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"
ID_NAME = XYZ.stem


@pytest.fixture
def staged_dry_run(tmp_path, run_cli):
    """`--pre quick-ca-fixed` into the default caopt-anfreq mode, without AnFreq."""
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py",
        str(XYZ),
        "--out-dir", str(out),
        "--scheduler", "slurm",
        "--pre", "quick-ca-fixed",
        "--no-anfreq",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out, result


def _run_dir(out: Path, mode: str) -> Path:
    return layout.run_dir_for(out, ID_NAME, mode)


def _xyzfile_line(orca_input: Path) -> str:
    lines = [ln for ln in orca_input.read_text().splitlines() if ln.startswith("*xyzfile")]
    assert len(lines) == 1, lines
    return lines[0]


def test_each_stage_gets_its_own_run_dir_and_job_script(staged_dry_run):
    out, _ = staged_dry_run
    for mode in ("quick-ca-fixed", "caopt-anfreq"):
        run_dir = _run_dir(out, mode)
        run_id = layout.run_id_for(ID_NAME, mode)
        assert run_dir.is_dir(), f"{mode}: no run dir"
        assert (run_dir / f"{run_id}.in").is_file(), f"{mode}: no ORCA input"
        assert (run_dir / f"generated-{run_id}-orca.script").is_file(), f"{mode}: no job script"


def test_first_stage_starts_from_its_own_cleaned_input_geometry(staged_dry_run):
    out, _ = staged_dry_run
    run_id = layout.run_id_for(ID_NAME, "quick-ca-fixed")
    run_dir = _run_dir(out, "quick-ca-fixed")
    geometry = _xyzfile_line(run_dir / f"{run_id}.in").split()[-1]
    assert geometry == str(run_dir / f"{run_id}_clean.xyz")
    assert Path(geometry).is_file()


def test_second_stage_starts_from_the_first_stage_optimized_output(staged_dry_run):
    """Not `_clean.xyz` (that is the un-optimized input) and not its own dir."""
    out, _ = staged_dry_run
    pre_id = layout.run_id_for(ID_NAME, "quick-ca-fixed")
    pre_dir = _run_dir(out, "quick-ca-fixed")
    final_id = layout.run_id_for(ID_NAME, "caopt-anfreq")

    geometry = _xyzfile_line(_run_dir(out, "caopt-anfreq") / f"{final_id}.in").split()[-1]
    assert geometry == str(pre_dir / f"{pre_id}.xyz")
    assert Path(geometry).is_absolute()
    # It cannot exist yet -- ORCA writes it. The afterok dependency is the guarantee.
    assert not Path(geometry).exists()


def test_no_anfreq_applies_to_the_final_stage_and_pre_stages_never_get_it(staged_dry_run):
    out, _ = staged_dry_run
    for mode in ("quick-ca-fixed", "caopt-anfreq"):
        run_id = layout.run_id_for(ID_NAME, mode)
        text = (_run_dir(out, mode) / f"{run_id}.in").read_text()
        assert "! AnFreq" not in text, f"{mode}: AnFreq present despite --no-anfreq"


def test_default_final_stage_still_runs_anfreq(tmp_path, run_cli):
    """The counterpart: without --no-anfreq the caopt stage keeps its Hessian."""
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out),
        "--scheduler", "slurm", "--pre", "quick-ca-fixed", "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    final_id = layout.run_id_for(ID_NAME, "caopt-anfreq")
    text = (_run_dir(out, "caopt-anfreq") / f"{final_id}.in").read_text()
    assert "! AnFreq" in text

    pre_id = layout.run_id_for(ID_NAME, "quick-ca-fixed")
    assert "! AnFreq" not in (_run_dir(out, "quick-ca-fixed") / f"{pre_id}.in").read_text()


def test_dry_run_reports_the_dependency_it_would_attach(staged_dry_run):
    _, result = staged_dry_run
    # Wording changed with the move to logging (WIP 77e7365); the content is the same.
    assert "Stages: quick-ca-fixed(no AnFreq) -> caopt-anfreq(no AnFreq)" in result.stdout
    # Nothing to depend on in a dry run (no job ids exist), but the chain is stated.
    assert "(previous stage)" in result.stdout
    assert "starting geometry:" in result.stdout


def test_next_stage_sidecar_lets_the_auto_rerun_rechain(staged_dry_run):
    out, _ = staged_dry_run
    pre_dir = _run_dir(out, "quick-ca-fixed")
    pre_id = layout.run_id_for(ID_NAME, "quick-ca-fixed")
    final_id = layout.run_id_for(ID_NAME, "caopt-anfreq")

    sidecar = pre_dir / f"{pre_id}-next-stage.json"
    assert sidecar.is_file(), "pre-stage must record what was queued after it"
    record = json.loads(sidecar.read_text())
    assert record["next_run_id"] == final_id
    assert Path(record["next_run_dir"]) == _run_dir(out, "caopt-anfreq")
    assert record["next_job_script"] == f"generated-{final_id}-orca.script"

    # The final stage has nothing after it.
    assert not (_run_dir(out, "caopt-anfreq") / f"{final_id}-next-stage.json").exists()


def test_single_stage_writes_no_sidecar(tmp_path, run_cli):
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out),
        "--scheduler", "slurm", "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = _run_dir(out, "caopt-anfreq")
    assert not list(run_dir.glob("*-next-stage.json"))


def test_three_stages_chain_head_to_tail(tmp_path, run_cli):
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out), "--scheduler", "slurm",
        "--pre", "quick", "--pre", "quick-ca-fixed", "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    chain = ["quick", "quick-ca-fixed", "caopt-anfreq"]
    for previous, current in itertools.pairwise(chain):
        prev_id = layout.run_id_for(ID_NAME, previous)
        cur_id = layout.run_id_for(ID_NAME, current)
        geometry = _xyzfile_line(_run_dir(out, current) / f"{cur_id}.in").split()[-1]
        assert geometry == str(_run_dir(out, previous) / f"{prev_id}.xyz")


def test_a_repeated_mode_is_rejected_before_anything_is_written(tmp_path, run_cli):
    """Two stages in one mode would want the same run dir; the second would
    overwrite the first's input mid-chain."""
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out), "--scheduler", "slurm",
        "--pre", "caopt-anfreq", "--dry-run",
    )
    assert result.returncode == 1
    assert "appear twice" in result.stderr
    assert "caopt-anfreq -> caopt-anfreq" in result.stderr
    assert not (out / ID_NAME).exists()


def test_pre_is_rejected_for_a_mode_that_runs_no_orca(tmp_path, run_cli):
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out), "--scheduler", "slurm",
        "--pre", "quick-ca-fixed", "--interp-raw", "--dry-run",
    )
    assert result.returncode == 1
    assert "runs no ORCA stage" in result.stderr
    # Now validated up front (WIP 77e7365), so not even the pre-stage is written.
    assert not (out / ID_NAME).exists()


def test_an_unfillable_placeholder_fails_the_structure_and_writes_no_input(tmp_path, run_cli):
    """Regression (Phase 1): the Cys fixture has no COORD= tags, so xTB has no QM
    region. This used to write ``QMATOMS [INDICES]`` verbatim and exit 0."""
    out = tmp_path / "batch-out"
    result = run_cli(
        "prepare-orca.py", str(XYZ), "--out-dir", str(out), "--scheduler", "slurm",
        "--xtb-free", "--dry-run",
    )
    assert result.returncode == 1
    assert "[INDICES]" in result.stdout and XYZ.name in result.stdout
    assert "1 of 1 structure(s) failed" in result.stdout
    assert not list(out.rglob("*.in"))
    assert not list(out.rglob("*.script"))


def test_each_stage_run_dir_parses_back_to_its_own_mode(staged_dry_run):
    """The chain must not confuse the run-dir vocabulary every scan depends on."""
    out, _ = staged_dry_run
    for mode in ("quick-ca-fixed", "caopt-anfreq"):
        assert layout.mode_from_run_id(_run_dir(out, mode).name) == mode


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_second_stage_is_submitted_afterok_on_the_first(tmp_path, monkeypatch):
    """The one thing --dry-run cannot exercise.

    Stage 2 reads a geometry stage 1 has not written yet, so submitting it without
    the dependency means ORCA starts on a missing file. Patch the scheduler seam
    rather than skip it.
    """
    from xas_pipeline import scheduler
    from xas_pipeline.stages import orca_prep

    calls: list[dict] = []

    def fake_run_command(cmd, cwd=None):
        calls.append({"cmd": list(cmd), "cwd": Path(cwd).name})
        return _Result(f"Submitted batch job {1000 + len(calls)}\n")

    monkeypatch.setattr(scheduler, "run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "xas-prepare-orca", str(XYZ),
            "--out-dir", str(tmp_path / "batch-out"),
            "--scheduler", "slurm",
            "--pre", "quick-ca-fixed",
            "--no-anfreq",
        ],
    )
    assert orca_prep.main() == 0

    assert len(calls) == 2, calls
    first, second = calls
    assert first["cmd"][0] == "sbatch"
    assert first["cwd"] == layout.run_id_for(ID_NAME, "quick-ca-fixed")
    assert second["cwd"] == layout.run_id_for(ID_NAME, "caopt-anfreq")

    assert not [a for a in first["cmd"] if a.startswith("--dependency")]
    assert "--dependency=afterok:1001" in second["cmd"], second["cmd"]

    # And the sidecar records the real job id, so a rerun can cancel it.
    sidecar = (
        layout.run_dir_for(tmp_path / "batch-out", ID_NAME, "quick-ca-fixed")
        / f"{layout.run_id_for(ID_NAME, 'quick-ca-fixed')}-next-stage.json"
    )
    record = json.loads(sidecar.read_text())
    assert record["next_job_id"] == "1002"
    assert record["dry_run"] is False


def test_a_failed_first_submission_does_not_queue_the_second(tmp_path, monkeypatch):
    """Nothing would ever write the geometry stage 2 is pointed at.

    Behaviour change (WIP 77e7365): main() now returns 1 instead of raising
    SystemExit(1); the console scripts exit with that code either way.
    """
    from xas_pipeline import scheduler
    from xas_pipeline.stages import orca_prep

    calls: list[list[str]] = []

    def fake_run_command(cmd, cwd=None):
        calls.append(list(cmd))
        return _Result(returncode=1, stderr="sbatch: error: invalid partition\n")

    monkeypatch.setattr(scheduler, "run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "xas-prepare-orca", str(XYZ),
            "--out-dir", str(tmp_path / "batch-out"),
            "--scheduler", "slurm", "--pre", "quick-ca-fixed",
        ],
    )
    assert orca_prep.main() == 1
    assert len(calls) == 1, "stage 2 must not be submitted after stage 1 failed to queue"
