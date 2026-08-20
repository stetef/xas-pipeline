"""CLI-level: `xas-process-feff` fails a run whose XANES chi is identically zero.

The unit tests pin the verdict; this pins what the *stage* does with it, because
that is what the rest of the pipeline reads: the id has to land in
``corvus-failed-ids.txt`` (which the download stage quarantines from and the
CORVUS auto-rerun triages) and get a ``FAILED`` outcome line in ``batch-jobs.log``
naming the reason. A run whose xanes chi is real must stay clean, so the gate
cannot be passing by failing everything.

Built in-tmp rather than from tests/fixtures/feff_run: the invalid path returns
before any plotting, so no larch/matplotlib output is involved and the fixture
only needs the tables the gate reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

XMU_HEADER = """# # Once                                                         FEFF 10.0.0
#    14/  14 paths used
#  omega    e    k    mu    mu0     chi     @#
"""
# XANES is an FMS calculation, so its own header legitimately says 0/0 paths --
# which is exactly why the zero-chi check has to look at the numbers.
XANES_HEADER = XMU_HEADER.replace("14/  14", " 0/   0")


def _table(path: Path, chis: list[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"  {9650.0 + i:.3f}  {i - 10.0:.3f}  {0.5 + 0.1 * i:.3f}  1.05  1.0  {chi}"
        for i, chi in enumerate(chis)
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _make_run(batch: Path, run_id: str, xanes_chi: list[str]) -> Path:
    run_dir = batch / run_id
    working = run_dir / f"working-{run_id}"
    (run_dir / f"output-{run_id}").mkdir(parents=True, exist_ok=True)
    feff = working / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    _table(feff / "xanes" / "xmu.dat", xanes_chi, XANES_HEADER)
    _table(feff / "exafs" / "xmu.dat", ["3.0E-02", "4.0E-02", "5.0E-02"], XMU_HEADER)
    _table(working / "Corvus.cfavg_xas.out", ["1.0E-02", "2.0E-02", "3.0E-02"], XMU_HEADER)
    (working / f"{run_id}.xyz").write_text("1\n\nZn 0.0 0.0 0.0\n", encoding="utf-8")
    return run_dir


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    batch = tmp_path_factory.mktemp("zerochi") / "batch-out"
    batch.mkdir(parents=True)
    (batch / "batch-jobs.log").write_text("# header\n", encoding="utf-8")
    _make_run(batch, "DEAD-interp", ["0.0000000000E+00"] * 3)
    _make_run(batch, "LIVE-interp", ["1.1E-02", "-2.2E-03", "3.3E-04"])

    from conftest import run_script

    result = run_script(
        "script-process-feff-output.py", str(batch), "--recursive", "--skip-fft"
    )
    return {"result": result, "batch": batch}


def test_stage_exits_clean(processed):
    result = processed["result"]
    assert result.returncode == 0, result.stderr


def test_dead_xanes_run_is_the_only_failed_id(processed):
    manifest = processed["batch"] / "corvus-failed-ids.txt"
    assert manifest.read_text(encoding="utf-8").split() == ["DEAD-interp"]


def test_failure_reason_names_the_dead_chi_column(processed):
    log = (processed["batch"] / "batch-jobs.log").read_text(encoding="utf-8")
    failed = [line for line in log.splitlines() if line.startswith("corvus-DEAD-interp")]
    assert failed, log
    assert "FAILED" in failed[-1]
    assert "no fine structure" in failed[-1]
    # ...and the healthy run is recorded OK, so the gate is not failing everything.
    assert any(
        line.startswith("corvus-LIVE-interp") and "OK" in line for line in log.splitlines()
    )


def test_healthy_run_still_produces_its_deliverables(processed):
    out = processed["batch"] / "LIVE-interp" / "output-LIVE-interp"
    assert (out / "xas-LIVE-interp.dat").is_file()
    assert (out / "xmu-xanes-LIVE-interp.dat").is_file()


def test_failed_run_keeps_its_geometry_but_no_spectrum(processed):
    """A failed id's output dir stays useful without pretending it has a spectrum."""
    out = processed["batch"] / "DEAD-interp" / "output-DEAD-interp"
    assert (out / "DEAD-interp.xyz").is_file()
    assert not (out / "xas-DEAD-interp.dat").exists()
