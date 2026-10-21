"""Unit tests for the auto-rerun triage core.

Covers the pure pieces that decide *whether* and *how* to resubmit a failed
ORCA run: log -> diagnosis (:mod:`xas_pipeline.diagnosis`), diagnosis -> remedy
and remedy -> edited input (:mod:`xas_pipeline.remedy`, which absorbed
``input_remedy``), plus the pure helpers of :mod:`xas_pipeline.recovery`. No
scheduler involved: the xas-rerun-orca CLI paths that submit are exercised
against a fake scheduler in ``test_recovery_regressions.py``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from xas_pipeline import diagnosis, recovery, remedy
from xas_pipeline.cli.rerun_orca import bump_job_script_mem
from xas_pipeline.diagnosis import FailureKind
from xas_pipeline.remedy import MAX_ATTEMPTS, Remedy, apply_remedy, select_remedy

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
    # Still classified, but no longer auto-remediable (SCF solver edits retired).
    assert not d.auto_remediable


def test_diagnose_scf_stalled_no_gap(tmp_path):
    body = "\n".join(
        [_scf_iter_block([-1234.5678 + 1e-7 * (i % 2) for i in range(12)]), "Error: SCF has not converged"]
    )
    run = _make_run(tmp_path, "STALL.S4_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.SCF_STALLED
    assert d.evidence.energy_stable and not d.evidence.small_gap
    assert not d.auto_remediable


def test_diagnose_scf_diverged(tmp_path):
    # Energies moving by >> 1e-3 across the tail -> not stable -> diverged.
    body = "\n".join(
        [_scf_iter_block([-1000.0 - i * 0.5 for i in range(12)]), "SCF has not converged"]
    )
    run = _make_run(tmp_path, "DIV.S4_zn1", body)
    d = diagnosis.diagnose(run)
    assert d.kind is FailureKind.SCF_DIVERGED
    assert not d.evidence.energy_stable
    assert not d.auto_remediable


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
    # A remediable kind (OOM) still has no rung past MAX_ATTEMPTS.
    assert select_remedy(FailureKind.OOM, _ev(), MAX_ATTEMPTS + 1) is None


def test_scf_failures_are_never_auto_remedied():
    """SCF solver edits were retired (tetef01, 2026-09-21): SCF non-convergence
    escalates to a human instead of being retried with SlowConv/level-shift."""
    for kind in (FailureKind.SCF_NEAR_DEGENERACY, FailureKind.SCF_STALLED,
                 FailureKind.SCF_DIVERGED):
        assert kind not in diagnosis.AUTO_REMEDIABLE, kind
        for attempt in (1, 2):
            assert select_remedy(
                kind, _ev(small_gap=True, gbw_present=True, n_opt_cycles=5), attempt
            ) is None, (kind, attempt)


def test_only_oom_and_opt_are_remediable():
    """The whole remediable set is exactly the two physics-neutral kinds."""
    assert frozenset(
        {FailureKind.OOM, FailureKind.OPT_NONCONVERGENCE}
    ) == diagnosis.AUTO_REMEDIABLE


def test_remedy_oom_bumps_maxcore():
    r1 = select_remedy(FailureKind.OOM, _ev(), 1)
    r2 = select_remedy(FailureKind.OOM, _ev(), 2)
    assert r1.maxcore_mult > 1.0 and r2.maxcore_mult > r1.maxcore_mult
    # OOM never touches the SCF solver: a Remedy cannot carry solver edits any
    # more (the keywords/scf_lines fields were removed with input_remedy).
    assert not hasattr(r1, "scf_lines") and not hasattr(r1, "keywords")


def test_remedy_oom_no_moread_or_restart_without_evidence():
    ev = _ev(gbw_present=False, n_opt_cycles=1)
    r = select_remedy(FailureKind.OOM, ev, 1)
    assert r.use_moread is False and r.opt_restart is False  # no gbw, <2 cycles


def test_remedy_opt_nonconvergence_restart_then_gives_up():
    r1 = select_remedy(FailureKind.OPT_NONCONVERGENCE, _ev(gbw_present=True, n_opt_cycles=9), 1)
    assert r1 is not None and r1.opt_restart and r1.use_moread
    assert r1.maxcore_mult == 1.0  # geometry restart only, no other edits
    assert select_remedy(FailureKind.OPT_NONCONVERGENCE, _ev(n_opt_cycles=9), 2) is None


# --------------------------------------------------------------------------- #
# apply_remedy (was input_remedy)
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


def test_apply_moread_restart_remedy_injects_its_cards():
    """Was the SlowConv/shift test: the only cards a remedy can add now are
    MOREAD + %moinp; the geometry swap is unchanged; no %scf ever appears."""
    r = Remedy(label="opt-restart", use_moread=True, opt_restart=True)
    out = apply_remedy(
        BASE_IN, r, gbw_name="TEY.S2N2_zn1-moread.gbw", last_geometry_name="TEY.S2N2_zn1.xyz"
    )
    assert out.count("! MOREAD") == 1
    assert out.index("! MOREAD") < out.index("! AnFreq")  # right after the first ! line
    assert '%moinp "TEY.S2N2_zn1-moread.gbw"' in out
    assert "%scf" not in out and "SlowConv" not in out and "SCFStabilityAnalysis" not in out
    assert f"{remedy.REMEDY_MARKER} opt-restart ---" in out
    # opt-restart swapped the geometry, preserving the absolute directory.
    assert "/abs/path/TEY.S2N2_zn1/TEY.S2N2_zn1.xyz" in out
    assert "_clean.xyz" not in out
    # The remedy block is emitted before the *xyzfile line.
    assert out.index("%moinp") < out.index("*xyzfile")
    assert "%maxcore 1800" in out


def test_apply_oom_remedy_bumps_maxcore_only():
    r = Remedy(label="oom-maxcore-x1.6", maxcore_mult=1.6)
    out = apply_remedy(BASE_IN, r)
    assert "%maxcore 2880" in out  # 1800 * 1.6
    assert "%scf" not in out and "MOREAD" not in out
    assert remedy.REMEDY_MARKER not in out


def test_apply_no_moread_when_no_gbw():
    r = Remedy(label="opt-restart", use_moread=True)
    out = apply_remedy(BASE_IN, r, gbw_name=None)
    assert "MOREAD" not in out and "%moinp" not in out  # requested, but no gbw name -> omitted


# --------------------------------------------------------------------------- #
# job-script memory bump
# --------------------------------------------------------------------------- #

def test_bump_job_script_mem_relative_to_base():
    base = "#SBATCH --mem=16G\n"
    text, note = bump_job_script_mem(base, 2.5)
    assert "--mem=40G" in text and note == "mem 16G -> 40G"


def test_bump_job_script_mem_pbs():
    """PBS OOM remedy honours ``-l mem=`` (it was ignored before 77e7365)."""
    text, note = bump_job_script_mem("#!/bin/bash\n#PBS -l mem=64gb\n#PBS -l nodes=1:ppn=16\n", 1.6)
    assert "#PBS -l mem=102gb" in text and note == "mem 64gb -> 102gb"
    assert "nodes=1:ppn=16" in text


def test_bump_job_script_mem_without_request_is_a_no_op():
    assert bump_job_script_mem("#!/bin/bash\n", 2.5) == ("#!/bin/bash\n", None)


# --------------------------------------------------------------------------- #
# recovery helpers
# --------------------------------------------------------------------------- #

def test_latest_corvus_job_id_picks_latest_submitted(tmp_path):
    """Was rerun_orca._find_stale_corvus_job_id (log text); now reads the log file."""
    log = tmp_path / "batch-jobs.log"
    log.write_text(
        "job_name\tstatus\tjob_id\n"
        "orca-FOO.S4_zn1\tSUBMITTED\tjob_id=100\n"
        "corvus-xas-FOO.S4_zn1\tSUBMITTED\tjob_id=101\n"
        "corvus-xas-OTHER_zn1\tSUBMITTED\tjob_id=999\n"          # different structure
        "corvus-FOO.S4_zn1\tCANCELLED\treason=\"x\"\n"            # not SUBMITTED
        "corvus-xas-rerun1-FOO.S4_zn1\tSUBMITTED\tjob_id=202\n"   # latest wins
        "corvus-xas-FOO.S4_zn1-caopt-anfreq\tSUBMITTED\tjob_id=303\n",  # prefix sibling: not ours
        encoding="utf-8",
    )
    assert recovery.latest_corvus_job_id(log, "FOO.S4_zn1") == "202"
    assert recovery.latest_corvus_job_id(log, "MISSING_zn1") is None


# ---------------------------------------------------------------------------
# Staged runs (prepare-orca --pre)
# ---------------------------------------------------------------------------


def test_opt_restart_full_path_replaces_the_whole_geometry_spec():
    """A staged stage 2 reads from the PREVIOUS stage's directory.

    Swapping only the basename there would build
    ``<stage1_dir>/<stage2_run_id>.xyz`` -- a file that does not and will never
    exist, so the restart would fail on a missing geometry.
    """
    line = "*xyzfile -2 1 /batch/id/id-quick-ca-fixed/id-quick-ca-fixed.xyz"
    swapped = remedy._swap_geometry(line, "/batch/id/id-caopt-anfreq/id-caopt-anfreq.xyz")
    assert swapped == "*xyzfile -2 1 /batch/id/id-caopt-anfreq/id-caopt-anfreq.xyz"


def test_opt_restart_bare_filename_still_keeps_the_existing_directory():
    """The single-stage behaviour, unchanged."""
    line = "*xyzfile -2 1 /batch/id/id-caopt-anfreq/id-caopt-anfreq_clean.xyz"
    swapped = remedy._swap_geometry(line, "id-caopt-anfreq.xyz")
    assert swapped == "*xyzfile -2 1 /batch/id/id-caopt-anfreq/id-caopt-anfreq.xyz"


def test_next_stage_sidecar_is_ignored_when_absent_or_malformed(tmp_path):
    """A broken sidecar must not stop the rerun of the stage itself."""
    run_dir = tmp_path / "id-quick-ca-fixed"
    run_dir.mkdir()
    assert recovery.next_stage_record(run_dir, "id-quick-ca-fixed") is None

    (run_dir / "id-quick-ca-fixed-next-stage.json").write_text("{not json", encoding="utf-8")
    assert recovery.next_stage_record(run_dir, "id-quick-ca-fixed") is None

    (run_dir / "id-quick-ca-fixed-next-stage.json").write_text("{}", encoding="utf-8")
    assert recovery.next_stage_record(run_dir, "id-quick-ca-fixed") is None

    good = {"next_job_script": "generated-x-orca.script", "next_run_dir": str(tmp_path)}
    (run_dir / "id-quick-ca-fixed-next-stage.json").write_text(json.dumps(good), encoding="utf-8")
    assert recovery.next_stage_record(run_dir, "id-quick-ca-fixed") == good
