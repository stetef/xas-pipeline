"""FMS's gg.bin checked for the two FEFF 10 artifacts behind the v3 odd XANES (2026-11-20).

* group A: an uninitialised ``FlType`` puts 4 leftover bytes in every ``#DF#`` header line; a CR (0x0D) there makes
  MKGTR misread the file, an LF (0x0A) makes it stop. 1 run in 256 each.
* group B: FMS's uninitialised rank ``map`` makes rank 0 misfile a worker's results; the worker's last slot is never
  filled and stays all zeros. Seen at 16 MPI ranks.

Covered in the order the pipeline meets them: :func:`chem.feff.scan_gg_bin`; the node-side table
(``corvus_pack`` -> ``<id>-gg-check.tsv``); the diagnosis kinds (auto-remediable, ahead of every other verdict); the
auto-rerun recording the real kind; the rerun archiving the old table. See audits/odd-xanes/feff-bugs-report.html.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xas_pipeline import corvus_diagnosis, layout, recovery, rerun_state
from xas_pipeline.chem import feff as chem_feff
from xas_pipeline.cli import auto_rerun_corvus as auto
from xas_pipeline.cli import rerun_corvus
from xas_pipeline.corvus_diagnosis import CorvusFailureKind

XMU = """# # Once                                                         FEFF 10.0.0
#    14/  14 paths used
#  omega    e    k    mu    mu0     chi     @#
  9650.000  -10.000  0.000  1.01  1.0  1.0E-02
  9651.000   -9.000  0.100  1.02  1.0  2.0E-02
"""


def _gg_bytes(n_sections: int = 4, *, junk: bytes = b"H\x7f\x00\x00", zero: tuple[int, ...] = (), size: int = 2) -> bytes:
    """A gg.bin as FMS writes it: per section a #SN# line, the #DF# line with 4 junk bytes, headers, a size x size
    complex matrix (real, imag pairs). Sections in *zero* hold only zeros (the never-filled slot)."""
    out = []
    for n in range(1, n_sections + 1):
        out.append(f"#SN#   Section: {n:4d}\n".encode())
        out.append(b"#DF# This section written in " + junk + b".\n")
        out.append(b"#H#\n")
        out.append(f"#DT# 2D complex array with sizes {size:4d}{size:4d}\n".encode())
        for row in range(size):
            vals = [0.0 if n in zero else 0.1 * n + 0.01 * row + 0.001 * c for c in range(2 * size)]
            out.append(("".join(f"{v:16.7E}" for v in vals) + " \n").encode())
    return b"".join(out)


def _write_gg(path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_gg_bytes(**kwargs))
    return path


# --------------------------------------------------------------------------- #
# the scan
# --------------------------------------------------------------------------- #

def test_clean_gg_bin(tmp_path):
    scan = chem_feff.scan_gg_bin(_write_gg(tmp_path / "gg.bin"))
    assert scan.status == chem_feff.GG_OK and scan.is_clean and not scan.is_artifact
    assert scan.n_sections == 4


@pytest.mark.parametrize("byte, status", [(b"\r", chem_feff.GG_HEADER_CR), (b"\n", chem_feff.GG_HEADER_LF)])
def test_cr_or_lf_header_byte_is_flagged(tmp_path, byte, status):
    scan = chem_feff.scan_gg_bin(_write_gg(tmp_path / "gg.bin", junk=byte + b"\x7f\x00\x00"))
    assert scan.status == status and scan.is_artifact
    assert scan.header_byte == byte[0]


def test_other_junk_bytes_are_fine(tmp_path):
    for junk in (b"\x00\x7f\x00\x00", b"\xff\x7f\x00\x00", b"TXT "):
        assert chem_feff.scan_gg_bin(_write_gg(tmp_path / "gg.bin", junk=junk)).is_clean


def test_all_zero_section_is_flagged(tmp_path):
    scan = chem_feff.scan_gg_bin(_write_gg(tmp_path / "gg.bin", zero=(3,)))
    assert scan.status == chem_feff.GG_ZERO_SECTION and scan.is_artifact
    assert scan.zero_sections == (3,)


def test_header_byte_outranks_zero_section(tmp_path):
    scan = chem_feff.scan_gg_bin(_write_gg(tmp_path / "gg.bin", junk=b"\r\x7f\x00\x00", zero=(2,)))
    assert scan.status == chem_feff.GG_HEADER_CR


def test_missing_or_sectionless_file_is_unreadable_not_an_artifact(tmp_path):
    assert chem_feff.scan_gg_bin(tmp_path / "nope.bin").status == chem_feff.GG_UNREADABLE
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    scan = chem_feff.scan_gg_bin(empty)
    assert scan.status == chem_feff.GG_UNREADABLE and not scan.is_artifact


# --------------------------------------------------------------------------- #
# diagnosis: loose gg.bin (in-place run) and the node's table (node-scratch run)
# --------------------------------------------------------------------------- #

def _make_run(batch_root: Path, run_id: str, *, gg: dict | None = None, table: bool = False) -> Path:
    """A post-processed run dir. *gg*: component -> _gg_bytes kwargs. With *table*, the gg.bin files are scanned
    into ``<id>-gg-check.tsv`` (as corvus_pack does on the node) and removed, like a node-scratch run's slim tree."""
    run_dir = batch_root / run_id
    working = run_dir / f"working-{run_id}"
    (run_dir / f"output-{run_id}").mkdir(parents=True, exist_ok=True)
    feff = working / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    for comp in ("xanes", "exafs"):
        (feff / comp).mkdir(parents=True, exist_ok=True)
        (feff / comp / "xmu.dat").write_text(XMU, encoding="utf-8")
    (working / "Corvus.cfavg_xas.out").write_text(XMU, encoding="utf-8")
    (working / f"{run_id}.hess").write_text("", encoding="utf-8")
    for comp, kwargs in (gg or {}).items():
        _write_gg(feff / comp / "gg.bin", **kwargs)
    if table:
        corvus_diagnosis.write_gg_check(feff, working / corvus_diagnosis.GG_CHECK_TEMPLATE.format(run_id=run_id))
        for comp in (gg or {}):
            (feff / comp / "gg.bin").unlink()
    return run_dir


@pytest.mark.parametrize("table", [False, True], ids=["loose-gg.bin", "node-table"])
@pytest.mark.parametrize(
    "gg, kind",
    [
        ({"xanes": {"junk": b"\r\x7f\x00\x00"}, "exafs": {}}, CorvusFailureKind.FMS_GG_HEADER),
        ({"xanes": {}, "exafs": {"junk": b"\n\x7f\x00\x00"}}, CorvusFailureKind.FMS_GG_HEADER),
        ({"xanes": {"zero": (3,)}, "exafs": {}}, CorvusFailureKind.FMS_GG_ZERO_SECTION),
    ],
)
def test_artifacts_fail_the_run_and_are_auto_remediable(tmp_path, gg, kind, table):
    diag = corvus_diagnosis.diagnose_run_dir(_make_run(tmp_path, "CL1-interp", gg=gg, table=table))
    assert not diag.ok and diag.kind is kind and diag.auto_remediable
    assert "gg.bin" in diag.reason


@pytest.mark.parametrize("table", [False, True], ids=["loose-gg.bin", "node-table"])
def test_clean_gg_passes(tmp_path, table):
    diag = corvus_diagnosis.diagnose_run_dir(_make_run(tmp_path, "CL2-interp", gg={"xanes": {}, "exafs": {}},
                                                       table=table))
    assert diag.ok and diag.kind is CorvusFailureKind.OK


def test_runs_from_before_the_check_are_not_failed(tmp_path):
    """No gg.bin and no table (every v3 run): no verdict, the usual checks decide."""
    assert corvus_diagnosis.diagnose_run_dir(_make_run(tmp_path, "OLD-interp")).ok


def test_artifact_outranks_a_missing_spectrum(tmp_path):
    """An LF byte stops MKGTR; whatever CORVUS left behind, the remedy is a rerun."""
    run_dir = _make_run(tmp_path, "LF-interp", gg={"xanes": {"junk": b"\n\x7f\x00\x00"}})
    (run_dir / "working-LF-interp" / "Corvus.cfavg_xas.out").unlink()
    diag = corvus_diagnosis.diagnose_run_dir(run_dir)
    assert diag.kind is CorvusFailureKind.FMS_GG_HEADER and diag.auto_remediable


def test_postprocess_gate_fails_it_too(tmp_path):
    from xas_pipeline.stages import feff_process

    working = _make_run(tmp_path, "CL3-interp", gg={"xanes": {"zero": (2,)}}) / "working-CL3-interp"
    valid, reason = feff_process.xas_is_valid(corvus_diagnosis.cfavg_output(working),
                                              corvus_diagnosis.mode_feff_dir(working))
    assert not valid and "all zero" in reason


def test_rerun_archives_the_old_table(tmp_path):
    run_dir = _make_run(tmp_path, "CL4-interp", gg={"xanes": {"zero": (2,)}}, table=True)
    working = run_dir / "working-CL4-interp"
    recovery.archive_corvus_outputs(working, run_dir / "output-CL4-interp", "CL4-interp", "rerun-TAG")
    assert not list(working.glob(corvus_diagnosis.GG_CHECK_GLOB))
    assert (working / "CL4-interp-gg-check.tsv.rerun-TAG").is_file()


# --------------------------------------------------------------------------- #
# auto-rerun records the real kind
# --------------------------------------------------------------------------- #

class _FakeRerun:
    def __call__(self, batch_root, **kwargs):
        records = [rerun_corvus.RerunRecord(run_id=r, run_dir=Path(batch_root) / r, corvus_job_id=f"9000{n}",
                                            archived=[]) for n, r in enumerate(sorted(kwargs["only_ids"]), 1)]
        return rerun_corvus.RerunOutcome(tag="rerun-TAG", records=records, skipped=[], postprocess_job_id="99999",
                                         state_file=None)


def test_auto_rerun_resubmits_and_records_the_kind(tmp_path, monkeypatch):
    batch = tmp_path / "batch-out"
    a = _make_run(batch, "A-interp", gg={"xanes": {"junk": b"\r\x7f\x00\x00"}}, table=True)
    b = _make_run(batch, "B-interp", gg={"xanes": {"zero": (2,)}}, table=True)
    _make_run(batch, "GOOD-interp", gg={"xanes": {}}, table=True)
    (batch / layout.FAILED_CORVUS_MANIFEST).write_text("A-interp\nB-interp\n", encoding="utf-8")
    monkeypatch.setattr(rerun_corvus, "rerun_ids", _FakeRerun())
    monkeypatch.setattr("sys.argv", ["xas-auto-rerun-corvus", str(batch), "--scheduler", "slurm"])
    monkeypatch.delenv(auto.AUTO_RERUN_ENV, raising=False)

    assert auto.main() == 0
    assert layout.read_failed_corvus_ids(batch) == []
    for run_dir, kind in ((a, CorvusFailureKind.FMS_GG_HEADER), (b, CorvusFailureKind.FMS_GG_ZERO_SECTION)):
        state = json.loads(rerun_state.state_path(run_dir, run_dir.name, kind="corvus").read_text(encoding="utf-8"))
        assert [att["kind"] for att in state["attempts"]] == [kind.value]
