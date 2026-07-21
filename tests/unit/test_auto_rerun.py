"""Unit tests for the auto-rerun triage core.

Covers the three pure pieces that decide *whether* and *how* to resubmit a failed
ORCA run: log -> diagnosis (:mod:`xas_pipeline.diagnosis`), diagnosis -> remedy
(:mod:`xas_pipeline.remedy`), and remedy -> edited input
(:mod:`xas_pipeline.input_remedy`). No scheduler involved.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from xas_pipeline import diagnosis, input_remedy
from xas_pipeline.diagnosis import FailureKind
from xas_pipeline.remedy import MAX_ATTEMPTS, Remedy, select_remedy


TERMINATION = "****ORCA TERMINATED NORMALLY****"


def _make_run(tmp_path: Path, run_id: str, log_body: str, *, gbw: bool = False,
              last_geom: bool = False) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / f"{run_id}-orca.log").write_text(log_body, encoding="utf-8")
    if gbw:
        (run_dir / f"{run_id}.gbw").write_bytes(b"\x00" * 16)
    if last_geom:
        (run_dir / f"{run_id}.xyz").write_text("1\ncoords\nZn 0 0 0\n", encoding="utf-8")
    return run_dir


def _scf_iter_block(energies: list[float]) -> str:
    return "\n".join(
        f"  {i+1:>3}      {e:.12f}     6.4e-06   (NR   MAcro)"
        for i, e in enumerate(energies)
    )


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #

def test_diagnose_ok(tmp_path):
    run = _make_run(tmp_path, "OK1.S2N2_zn1", f"some output\n{TERMINATION}\n")
    d = diagnosis.diagnose(run)
    assert d.ok and d.kind is FailureKind.OK


def test_diagnose_no_log(tmp_path):
    run = tmp_path / "NOLOG.S2N2_zn1"
    run.mkdir()
    d = diagnosis.diagnose(run)
    assert not d.ok and d.kind is FailureKind.NO_LOG and not d.auto_remediable


def test_diagnose_charge_mult_not_remediable(tmp_path):
    body = "multiplicity (1) is odd and number of electrons (353) is odd -> impossible\n"
    run = _make_run(tmp_path, "BAD.S2N2_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.CHARGE_MULT
    assert not d.auto_remediable


def test_diagnose_oom(tmp_path):
    body = "No memory left for COSX RHS\nORCA finished by error termination in NUMFREQ\n"
    run = _make_run(tmp_path, "OOM.NC24_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.OOM and d.auto_remediable
    assert d.evidence.failed_module == "NUMFREQ"


def test_diagnose_scf_near_degeneracy(tmp_path):
    # Stable energy + small-gap warning + several opt cycles + a GBW present.
    body = "\n".join(
        ["GEOMETRY OPTIMIZATION CYCLE"] * 5
        + ["Warning: op=0 Small HOMO/LUMO gap ( -0.013) - skipping pre-diagonalization"]
        + [_scf_iter_block([-4839.898321 + 1e-7 * (i % 3) for i in range(12)])]
        + ["Error (ORCA_LEANSCF): unfortunately, the SCF has not converged"]
    )
    run = _make_run(tmp_path, "TEY.S2N2_zn1", body, gbw=True, last_geom=True)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.SCF_NEAR_DEGENERACY
    assert d.evidence.small_gap and d.evidence.gbw_present
    assert d.evidence.n_opt_cycles == 5


def test_diagnose_scf_stalled_no_gap(tmp_path):
    body = "\n".join(
        [_scf_iter_block([-1234.5678 + 1e-7 * (i % 2) for i in range(12)])]
        + ["Error: SCF has not converged"]
    )
    run = _make_run(tmp_path, "STALL.S4_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.SCF_STALLED
    assert d.evidence.energy_stable and not d.evidence.small_gap


def test_diagnose_scf_diverged(tmp_path):
    # Energies moving by >> 1e-3 across the tail -> not stable -> diverged.
    body = "\n".join(
        [_scf_iter_block([-1000.0 - i * 0.5 for i in range(12)])]
        + ["SCF has not converged"]
    )
    run = _make_run(tmp_path, "DIV.S4_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.SCF_DIVERGED
    assert not d.evidence.energy_stable


def test_diagnose_opt_nonconvergence(tmp_path):
    body = "The optimization did not converge but reached the maximum number of cycles\n"
    run = _make_run(tmp_path, "OPT.S2N2_zn1", body, gbw=True)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.OPT_NONCONVERGENCE and d.auto_remediable


def test_diagnose_post_opt_not_remediable(tmp_path):
    body = "ORCA finished by error termination in ANFREQ\n"
    run = _make_run(tmp_path, "POST.NC24_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.POST_OPT and not d.auto_remediable


# --------------------------------------------------------------------------- #
# remedy ladder
# --------------------------------------------------------------------------- #

def _ev(**kw):
    return diagnosis.Evidence(**kw)


def test_remedy_charge_mult_never():
    assert select_remedy(FailureKind.CHARGE_MULT, _ev(), 1) is None


def test_remedy_ladder_bounded():
    ev = _ev(small_gap=True, gbw_present=True, n_opt_cycles=5)
    assert select_remedy(FailureKind.SCF_NEAR_DEGENERACY, ev, MAX_ATTEMPTS + 1) is None


def test_remedy_near_degeneracy_attempt1_smear_moread():
    ev = _ev(small_gap=True, gbw_present=True, n_opt_cycles=5)
    r = select_remedy(FailureKind.SCF_NEAR_DEGENERACY, ev, 1)
    assert r is not None
    assert any("SmearTemp" in s for s in r.scf_lines)
    assert r.use_moread and r.opt_restart and r.stability_analysis


def test_remedy_near_degeneracy_attempt2_shift_fresh():
    ev = _ev(small_gap=True, gbw_present=True, n_opt_cycles=5)
    r = select_remedy(FailureKind.SCF_NEAR_DEGENERACY, ev, 2)
    assert r is not None
    assert "SlowConv" in r.keywords
    assert any("Shift" in s for s in r.scf_lines)
    assert r.use_moread is False  # fresh guess on the escalation


def test_remedy_oom_bumps_maxcore():
    r1 = select_remedy(FailureKind.OOM, _ev(), 1)
    r2 = select_remedy(FailureKind.OOM, _ev(), 2)
    assert r1.maxcore_mult > 1.0 and r2.maxcore_mult > r1.maxcore_mult


def test_remedy_no_moread_without_gbw():
    ev = _ev(small_gap=True, gbw_present=False, n_opt_cycles=1)
    r = select_remedy(FailureKind.SCF_NEAR_DEGENERACY, ev, 1)
    assert r.use_moread is False and r.opt_restart is False  # <2 cycles


# --------------------------------------------------------------------------- #
# input_remedy
# --------------------------------------------------------------------------- #

BASE_IN = textwrap.dedent(
    """\
    ! TightOPT PBE0 D3BJ RIJCOSX def2-TZVP def2/J CPCM(water)

    ! AnFreq

    %basis
      NewGTO Zn "def2-TZVPP" end
    end

    %maxcore 1800

    %pal nprocs 16
      end

    *xyzfile 0 1 /abs/path/TEY.S2N2_zn1/TEY.S2N2_zn1_clean.xyz
    """
)


def test_apply_smear_remedy_injects_all_cards():
    r = Remedy(label="scf-smear", scf_lines=["SmearTemp 5000", "MaxIter 300"],
               use_moread=True, opt_restart=True, stability_analysis=True)
    out = input_remedy.apply_remedy(
        BASE_IN, r, gbw_name="TEY.S2N2_zn1.gbw", last_geometry_name="TEY.S2N2_zn1.xyz"
    )
    assert "! MOREAD" in out
    assert '%moinp "TEY.S2N2_zn1.gbw"' in out
    assert "! SCFStabilityAnalysis" in out
    assert "%scf" in out and "  SmearTemp 5000" in out and "end" in out
    assert input_remedy.REMEDY_MARKER in out
    # opt-restart swapped the geometry, preserving the absolute directory.
    assert "/abs/path/TEY.S2N2_zn1/TEY.S2N2_zn1.xyz" in out
    assert "_clean.xyz" not in out
    # %scf block must be emitted before the *xyzfile line.
    assert out.index("%scf") < out.index("*xyzfile")


def test_apply_oom_remedy_bumps_maxcore_only():
    r = Remedy(label="oom-maxcore-x1.6", maxcore_mult=1.6)
    out = input_remedy.apply_remedy(BASE_IN, r)
    assert "%maxcore 2880" in out  # 1800 * 1.6
    assert "%scf" not in out and "MOREAD" not in out


def test_apply_no_moread_when_no_gbw():
    r = Remedy(label="scf-slowconv", keywords=["SlowConv"], scf_lines=["MaxIter 300"],
               use_moread=True)
    out = input_remedy.apply_remedy(BASE_IN, r, gbw_name=None)
    assert "MOREAD" not in out  # requested, but no gbw name -> omitted
    assert "! SlowConv" in out


# --------------------------------------------------------------------------- #
# CLI: mem bump + no card stacking across attempts
# --------------------------------------------------------------------------- #

def test_bump_job_script_mem_relative_to_base():
    from xas_pipeline.cli.rerun_orca import _bump_job_script_mem

    base = "#SBATCH --mem=16G\n"
    text, note = _bump_job_script_mem(base, 2.5)
    assert "--mem=40G" in text and note is not None


def _near_degeneracy_run(tmp_path: Path, run_id: str) -> Path:
    body = "\n".join(
        ["GEOMETRY OPTIMIZATION CYCLE"] * 4
        + ["Warning: op=0 Small HOMO/LUMO gap ( -0.010) - skipping pre-diagonalization"]
        + [_scf_iter_block([-100.0 + 1e-7 * (i % 3) for i in range(12)])]
        + ["Error (ORCA_LEANSCF): unfortunately, the SCF has not converged"]
    )
    run = _make_run(tmp_path, run_id, body, gbw=True, last_geom=True)
    (run / f"{run_id}.in").write_text(
        BASE_IN.replace("TEY.S2N2_zn1", run_id), encoding="utf-8"
    )
    (run / f"generated-{run_id}-orca.script").write_text(
        "#!/bin/bash\n#SBATCH --mem=16G\n", encoding="utf-8"
    )
    return run


def test_cli_attempts_do_not_stack_cards(tmp_path):
    import subprocess
    import sys

    run_id = "STK.S2N2_zn1"
    run = _near_degeneracy_run(tmp_path, run_id)
    cmd = [sys.executable, "-m", "xas_pipeline.cli.rerun_orca", str(run),
           "--scheduler", "slurm", "--no-submit"]
    for _ in range(2):  # two attempts on the same (unchanged) failing log
        assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0

    remedied = (run / f"{run_id}.in").read_text()
    # Exactly one remedy block despite two attempts (remedy is applied to the
    # pristine original each time, not the previously-remedied input).
    assert remedied.count("%scf") == 1
    assert remedied.count(input_remedy.REMEDY_MARKER) == 1
    # Attempt 2 escalated to shift + SlowConv from a fresh guess (no MOREAD).
    assert "! SlowConv" in remedied and "Shift Shift" in remedied
    assert "! MOREAD" not in remedied
    assert (run / f"{run_id}-rerun-history" / f"original-{run_id}.in").is_file()


def test_find_stale_corvus_job_id_picks_latest_submitted():
    from xas_pipeline.cli.rerun_orca import _find_stale_corvus_job_id

    log = (
        "job_name\tstatus\tjob_id\n"
        "orca-FOO.S4_zn1\tSUBMITTED\tjob_id=100\n"
        "corvus-xas-FOO.S4_zn1\tSUBMITTED\tjob_id=101\n"
        "corvus-xas-OTHER_zn1\tSUBMITTED\tjob_id=999\n"          # different structure
        "corvus-FOO.S4_zn1\tCANCELLED\treason=\"x\"\n"            # not SUBMITTED
        "corvus-xas-rerun1-FOO.S4_zn1\tSUBMITTED\tjob_id=202\n"   # latest wins
    )
    assert _find_stale_corvus_job_id(log, "FOO.S4_zn1") == "202"
    assert _find_stale_corvus_job_id(log, "MISSING_zn1") is None


def test_cli_cancels_dependent_corvus_on_orca_failure(tmp_path, monkeypatch):
    """On an ORCA failure, the dependent CORVUS job id is looked up and cancelled."""
    from xas_pipeline import orchestrate as bp
    from xas_pipeline.cli import rerun_orca

    run_id = "CAN.S2N2_zn1"
    run = _near_degeneracy_run(tmp_path, run_id)
    (tmp_path / "batch-jobs.log").write_text(
        "job_name\tstatus\tjob_id\n"
        f"orca-{run_id}\tSUBMITTED\tjob_id=500\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n",
        encoding="utf-8",
    )
    cancelled: list[tuple[str, str]] = []
    monkeypatch.setattr(bp, "_cancel_job", lambda jid, sched: cancelled.append((jid, sched)) or True)
    monkeypatch.setattr(bp, "_check_executable", lambda name: None)
    monkeypatch.setattr(bp, "_submit_job", lambda *a, **k: "600")
    monkeypatch.setattr(bp, "_write_corvus_wrapper_script", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["xas-rerun-orca", str(run), "--scheduler", "slurm"])

    assert rerun_orca.main() == 0
    assert cancelled == [("501", "slurm")]
    assert "CANCELLED" in (tmp_path / "batch-jobs.log").read_text()


def test_cli_escalates_to_canonical_channels_not_txt(tmp_path):
    """Exhausting the ladder records NEEDS_HUMAN in state + batch-jobs.log, no sidecar .txt."""
    import json
    import subprocess
    import sys

    run_id = "ESC.S2N2_zn1"
    run = _near_degeneracy_run(tmp_path, run_id)
    (tmp_path / "batch-jobs.log").write_text("job_name\tstatus\tjob_id\n", encoding="utf-8")

    cmd = [sys.executable, "-m", "xas_pipeline.cli.rerun_orca", str(run),
           "--scheduler", "slurm", "--no-submit"]
    for _ in range(MAX_ATTEMPTS + 1):  # two remedied attempts, then escalation
        assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0

    # No bespoke sidecar marker.
    assert not (run / f"{run_id}-needs-human.txt").exists()
    # Terminal resolution recorded in the structured per-run state.
    state = json.loads((run / f"{run_id}-rerun-state.json").read_text())
    assert state["resolution"] == "needs_human"
    assert len(state["attempts"]) == MAX_ATTEMPTS
    # And surfaced in the canonical batch-jobs.log.
    assert "NEEDS_HUMAN" in (tmp_path / "batch-jobs.log").read_text()
