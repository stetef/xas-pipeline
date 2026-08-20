"""A XANES leg that produced no fine structure: detection, verdict, auto-rerun.

The failure: a CORVUS run finishes fine and writes a xanes ``xmu.dat`` whose chi
column is identically zero (``mu == mu0`` on every row). No header says so -- the
"0/0 paths used" line is meaningless for XANES -- so the gate used to pass these
as good spectra. Covered here, in the order the pipeline meets them:

1. :func:`xas_pipeline.chem.feff.scan_chi_column` -- the numeric check;
2. :mod:`xas_pipeline.corvus_diagnosis` -- kind + is-it-auto-remediable, and the
   gate (``feff_process.xas_is_valid``) agreeing with it by construction;
3. :mod:`xas_pipeline.cli.auto_rerun_corvus` -- which ids get recomputed, the
   bounded ladder, and the failed-manifest rewrite that keeps the download stage
   from quarantining a run dir a queued job is about to write into.

No scheduler: the submission mechanism (``rerun_corvus.rerun_ids``, already
covered by tests/cli/test_rerun_corvus_dryrun.py) is stubbed, so what is under
test here is the *policy* around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xas_pipeline import corvus_diagnosis, rerun_state
from xas_pipeline.chem import feff as chem_feff
from xas_pipeline.cli import auto_rerun_corvus as auto
from xas_pipeline.cli import rerun_corvus
from xas_pipeline.corvus_diagnosis import CorvusFailureKind

XMU_HEADER = """# # Once                                                         FEFF 10.0.0
#     0/   0 paths used
#  omega    e    k    mu    mu0     chi     @#
"""


def _write_table(path: Path, chis, *, header: str = XMU_HEADER) -> Path:
    """Write a 6-column FEFF-style table whose chi column is ``chis``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, chi in enumerate(chis):
        omega = 9650.0 + i
        mu0 = 1.0
        mu = mu0 + (chi if isinstance(chi, float) else 0.0)
        rows.append(
            f"  {omega:.3f}  {omega - 9660.0:.3f}  {0.1 * i:.3f}  {mu}  {mu0}  {chi}"
        )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the numeric check
# --------------------------------------------------------------------------- #

def test_all_zero_chi_is_flagged(tmp_path):
    scan = chem_feff.scan_chi_column(
        _write_table(tmp_path / "xmu.dat", ["0.0000000000E+00"] * 4)
    )
    assert scan.status == chem_feff.CHI_ALL_ZERO
    assert scan.is_all_zero and not scan.is_clean
    assert scan.n_rows == 4 and scan.n_zero == 4
    assert "no fine structure" in scan.detail


def test_real_spectrum_is_clean(tmp_path):
    scan = chem_feff.scan_chi_column(
        _write_table(tmp_path / "xmu.dat", ["1.234E-02", "-5.6E-03", "7.8E-04"])
    )
    assert scan.is_clean and scan.n_zero == 0 and scan.n_rows == 3


def test_partial_zeros_are_separated_from_all_zero(tmp_path):
    """A truncated/patched grid is suspicious, not dead -- a different bucket."""
    scan = chem_feff.scan_chi_column(
        _write_table(tmp_path / "xmu.dat", ["0.0", "0.0", "1.5E-02", "0.0"])
    )
    assert scan.status == chem_feff.CHI_PARTIAL_ZERO
    assert not scan.is_all_zero
    assert scan.n_zero == 3 and scan.n_rows == 4
    assert scan.zero_energy_range == (9650.0, 9653.0)


def test_non_finite_chi_is_its_own_bucket(tmp_path):
    scan = chem_feff.scan_chi_column(_write_table(tmp_path / "xmu.dat", ["nan", "1.0E-02"]))
    assert scan.status == chem_feff.CHI_NON_FINITE
    assert scan.n_nonfinite == 1


def test_header_only_file_is_unreadable(tmp_path):
    path = tmp_path / "xmu.dat"
    path.write_text(XMU_HEADER, encoding="utf-8")
    scan = chem_feff.scan_chi_column(path)
    assert scan.status == chem_feff.CHI_UNREADABLE
    assert scan.n_rows == 0


def test_fortran_overflow_rows_count_as_malformed_not_zero(tmp_path):
    """"****" from a Fortran overflow must not read as a chi value of any kind."""
    path = tmp_path / "xmu.dat"
    path.write_text(
        XMU_HEADER
        + "  9650.0  -10.0  0.1  1.0  1.0  ****\n"
        + "  9651.0   -9.0  0.2  1.01 1.0  1.0E-02\n",
        encoding="utf-8",
    )
    scan = chem_feff.scan_chi_column(path)
    assert scan.n_malformed == 1 and scan.n_rows == 1 and scan.n_zero == 0


def test_missing_file_is_unreadable(tmp_path):
    assert chem_feff.scan_chi_column(tmp_path / "nope.dat").status == chem_feff.CHI_UNREADABLE


# --------------------------------------------------------------------------- #
# the verdict, and the gate that shares it
# --------------------------------------------------------------------------- #

EXAFS_PATHS_HEADER = """# # Once                                                         FEFF 10.0.0
#    14/  14 paths used
#  omega    e    k    mu    mu0     chi     @#
"""


def _make_run(
    batch_root: Path,
    run_id: str,
    *,
    xanes_chi=("1.0E-02", "2.0E-02"),
    exafs_header: str = EXAFS_PATHS_HEADER,
    cfavg: bool = True,
    hess: bool = True,
) -> Path:
    """A post-processed run dir (split layout) with a combined-xas tree."""
    run_dir = batch_root / run_id
    working = run_dir / f"working-{run_id}"
    (run_dir / f"output-{run_id}").mkdir(parents=True, exist_ok=True)
    feff = working / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    _write_table(feff / "xanes" / "xmu.dat", list(xanes_chi))
    _write_table(feff / "exafs" / "xmu.dat", ["3.0E-02", "4.0E-02"], header=exafs_header)
    if cfavg:
        _write_table(working / "Corvus.cfavg_xas.out", ["1.0E-02", "2.0E-02"])
    if hess:
        (working / f"{run_id}.hess").write_text("", encoding="utf-8")
    return run_dir


def test_dead_xanes_leg_is_auto_remediable(tmp_path):
    run_dir = _make_run(tmp_path, "CL1-interp", xanes_chi=["0.0"] * 3)
    diag = corvus_diagnosis.diagnose_run_dir(run_dir)
    assert not diag.ok
    assert diag.kind is CorvusFailureKind.XANES_ZERO_CHI
    assert diag.auto_remediable
    assert "no fine structure" in diag.reason


def test_healthy_run_passes(tmp_path):
    diag = corvus_diagnosis.diagnose_run_dir(_make_run(tmp_path, "CL2-interp"))
    assert diag.ok and diag.kind is CorvusFailureKind.OK and not diag.warnings


def test_partly_zero_xanes_passes_but_warns(tmp_path):
    """Not dead output, so it must not fail the run -- but say something."""
    run_dir = _make_run(tmp_path, "CL3-interp", xanes_chi=["0.0", "1.0E-02", "0.0"])
    diag = corvus_diagnosis.diagnose_run_dir(run_dir)
    assert diag.ok
    assert diag.warnings and "partial-zero" in diag.warnings[0]


@pytest.mark.parametrize(
    "kwargs, kind",
    [
        ({"cfavg": False}, CorvusFailureKind.MISSING_SPECTRUM),
        ({"exafs_header": XMU_HEADER}, CorvusFailureKind.NO_EXAFS_PATHS),
    ],
)
def test_other_failures_are_never_auto_remediable(tmp_path, kwargs, kind):
    """Only the sporadic one is retried; the rest mean the inputs are wrong."""
    run_dir = _make_run(tmp_path, "CL4-interp", **kwargs)
    diag = corvus_diagnosis.diagnose_run_dir(run_dir)
    assert diag.kind is kind
    assert not diag.ok and not diag.auto_remediable


def test_postprocess_gate_shares_the_verdict(tmp_path):
    """feff_process.xas_is_valid must fail exactly what the diagnosis fails."""
    from xas_pipeline.stages import feff_process

    dead = _make_run(tmp_path, "CL5-interp", xanes_chi=["0.0"] * 3)
    working = dead / "working-CL5-interp"
    valid, reason = feff_process.xas_is_valid(
        feff_process.cfavg_xas_output(working), feff_process.mode_feff_dir(working, "xas")
    )
    assert not valid and "no fine structure" in reason

    live = _make_run(tmp_path, "CL6-interp")
    working = live / "working-CL6-interp"
    assert feff_process.xas_is_valid(
        feff_process.cfavg_xas_output(working), feff_process.mode_feff_dir(working, "xas")
    ) == (True, "ok")


# --------------------------------------------------------------------------- #
# the auto-rerun policy
# --------------------------------------------------------------------------- #

def _manifest(batch_root: Path, *run_ids: str) -> Path:
    path = batch_root / auto.FAILED_MANIFEST
    path.write_text("\n".join(run_ids) + "\n", encoding="utf-8")
    return path


class _FakeRerun:
    """Stands in for rerun_corvus.rerun_ids: records the call, submits nothing."""

    def __init__(self, *, runnable: set[str] | None = None):
        self.calls: list[dict] = []
        self.runnable = runnable

    def __call__(self, batch_root, **kwargs):
        self.calls.append({"batch_root": Path(batch_root), **kwargs})
        ids = sorted(kwargs["only_ids"])
        if self.runnable is not None:
            ids = [i for i in ids if i in self.runnable]
        records = [
            rerun_corvus.RerunRecord(
                run_id=run_id,
                run_dir=str(Path(batch_root) / run_id / f"working-{run_id}"),
                corvus_mode=kwargs.get("mode", "xas"),
                wrapper_script="wrapper.script",
                corvus_job_id=f"9000{n}",
                archived=["Corvus3_cfavg_xas -> Corvus3_cfavg_xas.rerun-TAG"],
            )
            for n, run_id in enumerate(ids, start=1)
        ]
        return rerun_corvus.RerunOutcome(
            tag="rerun-TAG", records=records, skipped=[],
            postprocess_job_id="99999", state_file=None,
        )


def _run_main(monkeypatch, batch_root: Path, *extra: str) -> tuple[int, _FakeRerun]:
    fake = _FakeRerun()
    monkeypatch.setattr(rerun_corvus, "rerun_ids", fake)
    monkeypatch.setattr(
        "sys.argv",
        ["xas-auto-rerun-corvus", str(batch_root), "--scheduler", "slurm", *extra],
    )
    monkeypatch.delenv(auto.AUTO_RERUN_ENV, raising=False)
    return auto.main(), fake


def _state(run_dir: Path, run_id: str) -> dict:
    return json.loads(
        rerun_state.state_path(run_dir, run_id, kind="corvus").read_text(encoding="utf-8")
    )


def test_only_the_dead_xanes_run_is_recomputed(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    _make_run(batch, "NOSPEC-interp", cfavg=False)
    _make_run(batch, "GOOD-interp")
    # The gate failed the first two; GOOD is not in the manifest at all.
    _manifest(batch, "DEAD-interp", "NOSPEC-interp")

    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0
    assert fake.calls and fake.calls[0]["only_ids"] == {"DEAD-interp"}
    assert fake.calls[0]["scheduler"] == "slurm"

    # The resubmitted id leaves the manifest so the download stage does NOT
    # quarantine a run dir the queued corvus job is about to write into; the
    # non-remediable one stays and is quarantined as before.
    assert auto.read_failed_ids(batch) == ["NOSPEC-interp"]


def test_attempt_is_recorded_with_the_job_id(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    run_dir = _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    _manifest(batch, "DEAD-interp")

    _run_main(monkeypatch, batch)

    state = _state(run_dir, "DEAD-interp")
    assert state["resolution"] is None
    assert len(state["attempts"]) == 1
    attempt = state["attempts"][0]
    assert attempt["attempt"] == 1
    assert attempt["kind"] == CorvusFailureKind.XANES_ZERO_CHI.value
    assert attempt["remedy"] == auto.REMEDY_LABEL
    assert attempt["corvus_job_id"] == "90001"
    # The ORCA ladder's own state file is untouched: separate counters.
    assert not rerun_state.state_path(run_dir, "DEAD-interp", kind="orca").exists()


def test_ladder_is_bounded_then_escalated(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    (batch / "batch-jobs.log").parent.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    (batch / "batch-jobs.log").write_text("# header\n", encoding="utf-8")

    for expected in (1, 2):
        _manifest(batch, "DEAD-interp")
        _run_main(monkeypatch, batch)
        assert len(_state(run_dir, "DEAD-interp")["attempts"]) == expected
        assert auto.read_failed_ids(batch) == []

    # Third dead recompute: the ladder is out of rungs.
    _manifest(batch, "DEAD-interp")
    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0
    assert not fake.calls, "a third recompute was submitted"
    state = _state(run_dir, "DEAD-interp")
    assert state["resolution"] == rerun_state.RESOLUTION_NEEDS_HUMAN
    assert len(state["attempts"]) == 2
    # Left in the manifest -> quarantined by the download stage, and recorded
    # next to every other outcome.
    assert auto.read_failed_ids(batch) == ["DEAD-interp"]
    log = (batch / "batch-jobs.log").read_text(encoding="utf-8")
    assert "NEEDS_HUMAN" in log and "ladder exhausted" in log


def test_escalated_run_is_not_re_escalated(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    run_dir = _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    state_file = rerun_state.state_path(run_dir, "DEAD-interp", kind="corvus")
    rerun_state.save_state(
        state_file,
        rerun_state.RerunState(
            run_id="DEAD-interp", attempts=[], resolution=rerun_state.RESOLUTION_NEEDS_HUMAN
        ),
    )
    _manifest(batch, "DEAD-interp")

    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0 and not fake.calls
    assert auto.read_failed_ids(batch) == ["DEAD-interp"]


def test_empty_manifest_is_a_no_op(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    _make_run(batch, "GOOD-interp")
    _manifest(batch)

    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0 and not fake.calls


def test_env_switch_disables_the_triage(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    _manifest(batch, "DEAD-interp")

    fake = _FakeRerun()
    monkeypatch.setattr(rerun_corvus, "rerun_ids", fake)
    monkeypatch.setattr("sys.argv", ["xas-auto-rerun-corvus", str(batch)])
    monkeypatch.setenv(auto.AUTO_RERUN_ENV, "0")

    assert auto.main() == 0
    assert not fake.calls
    assert auto.read_failed_ids(batch) == ["DEAD-interp"]


def test_dry_run_touches_neither_manifest_nor_state(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    run_dir = _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    _manifest(batch, "DEAD-interp")

    rc, fake = _run_main(monkeypatch, batch, "--no-submit")
    assert rc == 0
    assert fake.calls[0]["no_submit"] is True
    assert auto.read_failed_ids(batch) == ["DEAD-interp"]
    assert not rerun_state.state_path(run_dir, "DEAD-interp", kind="corvus").exists()


def test_unrunnable_candidate_stays_failed(tmp_path, monkeypatch):
    """No <id>.hess to recompute from is not something a retry fixes."""
    batch = tmp_path / "batch-out"
    run_dir = _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3, hess=False)
    _manifest(batch, "DEAD-interp")

    fake = _FakeRerun(runnable=set())  # rerun_ids skips it
    monkeypatch.setattr(rerun_corvus, "rerun_ids", fake)
    monkeypatch.setattr("sys.argv", ["xas-auto-rerun-corvus", str(batch)])
    monkeypatch.delenv(auto.AUTO_RERUN_ENV, raising=False)

    assert auto.main() == 0
    assert auto.read_failed_ids(batch) == ["DEAD-interp"]
    assert not rerun_state.state_path(run_dir, "DEAD-interp", kind="corvus").exists()


def test_sibling_mode_runs_are_not_dragged_in(tmp_path, monkeypatch):
    """A pre-suffix run dir also groups its later mode runs; only it gets rerun.

    ``iter_id_dirs`` matches ``only_ids`` by group name too (deliberate for the
    manual CLI: naming a structure takes every mode run for it). Here the failed
    id IS the group name, and the healthy ``-interp`` sibling inside it must not
    be archived and recomputed along with it.
    """
    batch = tmp_path / "batch-out"
    group = batch / "1a8h_ZN_cluster1"
    _make_run(batch, "1a8h_ZN_cluster1", xanes_chi=["0.0"] * 3)   # the original run
    _make_run(group, "1a8h_ZN_cluster1-interp")                    # a healthy mode run
    _manifest(batch, "1a8h_ZN_cluster1")

    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0
    assert fake.calls[0]["only_ids"] == {"1a8h_ZN_cluster1"}
    assert fake.calls[0]["match_groups"] is False


def test_manifest_id_with_no_run_dir_is_left_alone(tmp_path, monkeypatch):
    """Already quarantined (or moved by hand): never resurrect what we can't see."""
    batch = tmp_path / "batch-out"
    _make_run(batch, "DEAD-interp", xanes_chi=["0.0"] * 3)
    _manifest(batch, "DEAD-interp", "GONE-interp")

    rc, fake = _run_main(monkeypatch, batch)
    assert rc == 0
    assert fake.calls[0]["only_ids"] == {"DEAD-interp"}
    assert auto.read_failed_ids(batch) == ["GONE-interp"]
