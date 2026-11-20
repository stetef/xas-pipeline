"""stages/corvus_pack: what a node-scratch CORVUS run sends back, and the path -> atom table."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from xas_pipeline.stages import corvus_pack

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures" / "feff_run" / "5a3w_ZN_homo_d2.60_cluster2"
           / "working-5a3w_ZN_homo_d2.60_cluster2")
RUN_ID = "toy-caopt-spring"
FEFF = "Corvus3_cfavg_xas/Corvus1Zn_3_FEFF"

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="needs zstd")

# Absorber Zn is atom 1 (not 0, and not at the origin) so the test checks the shift and the index lookup.
_XYZ = """4
toy
S   1.000000  0.000000  0.000000
Zn  1.000000  1.000000  1.000000
N   1.000000  1.000000  3.050000
C   3.000000  1.000000  1.000000
"""

# FEFF frame = absorber at the origin. Path 3 is not in the xmu.dat header below (FEFF dropped it).
_PATHS = """PATH  Rmax= 9.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%
 -----------------------------------------------------------------------
     1    2   1.000  index, nleg, degeneracy, r=  1.7321
      x           y           z     ipot  label      rleg      beta        eta
    0.000000   -1.000000   -1.000000   2 'S     '     1.4142  180.0000    0.0000
    0.000000    0.000000    0.000000   0 'Zn    '     1.4142  180.0000    0.0000
     2    3   2.000  index, nleg, degeneracy, r=  2.5000
      x           y           z     ipot  label      rleg      beta        eta
    0.000000    0.000000    2.050000   1 'N     '     2.0500  180.0000    0.0000
    2.000000    0.000000    0.000000   3 'C     '     2.8636   90.0000    0.0000
    0.000000    0.000000    0.000000   0 'Zn    '     2.0000   90.0000    0.0000
     3    2   1.000  index, nleg, degeneracy, r=  9.0000
      x           y           z     ipot  label      rleg      beta        eta
    9.000000    0.000000    0.000000   3 'C     '     9.0000  180.0000    0.0000
    0.000000    0.000000    0.000000   0 'Zn    '     9.0000  180.0000    0.0000
"""

_SIG2_HEADER = """#  S02=1.000  Temp=  10.00  Debye_temp= 315.00  Global_sig2= 0.00000
#     file         sig2 tot  cw amp ratio   deg  nlegs   reff  inp sig2
#           1       0.00246    100.00      1.00     2   1.7321
#           2       0.00400     50.00      2.00     3   2.5000
"""


def _scratch(tmp_path: Path) -> Path:
    s = tmp_path / "scratch"
    feff = s / FEFF
    for comp in ("xanes", "exafs"):
        (feff / comp).mkdir(parents=True)
        shutil.copy2(FIXTURE / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF" / comp / "xmu.dat", feff / comp / "xmu.dat")
    shutil.copy2(FIXTURE / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF" / "exafs" / "chi.dat", feff / "exafs" / "chi.dat")
    xmu = feff / "exafs" / "xmu.dat"
    # Replace the fixture's own header (a real run's path list) with the toy one.
    xmu.write_text(_SIG2_HEADER + "".join(row for row in xmu.read_text().splitlines(True) if not row.startswith("#")))
    (feff / "exafs" / "paths.dat").write_text(_PATHS)
    (feff / "exafs" / "dmdw.out").write_text("regenerable\n")
    (feff / "exafs" / "fort.bin").write_bytes(b"\0" * 64)
    (feff / "xanes" / "gg.bin").write_text("#SN#   Section:    1\n")  # FMS output: kept for the gg.bin check
    (feff / "exafs" / "log1.dat").write_text("feff log\n")
    shutil.copy2(FIXTURE / "Corvus.cfavg_xas.out", s / "Corvus.cfavg_xas.out")
    (s / f"corvus-begin-{RUN_ID}.xyz").write_text(_XYZ)
    (s / "Corvus.nest").write_text("nest\n")
    (s / ".heartbeat").write_text("")
    return s


def test_return_set_and_archive(tmp_path):
    s = _scratch(tmp_path)
    ret = corvus_pack.build_return(s, RUN_ID)

    loose = sorted(str(p.relative_to(ret)) for p in ret.rglob("*") if p.is_file())
    assert loose == sorted([
        f"{RUN_ID}-corvus-xas.tar.zst",
        f"{RUN_ID}-corvus-xas-node.txt",
        "Corvus.cfavg_xas.out",
        f"{FEFF}/xanes/xmu.dat",
        f"{FEFF}/exafs/xmu.dat",
        f"{FEFF}/exafs/chi.dat",
        f"feff-paths-{RUN_ID}.tsv",
    ])
    # The slim tree is byte-identical to what CORVUS wrote.
    for rel in (f"{FEFF}/xanes/xmu.dat", f"{FEFF}/exafs/xmu.dat", f"{FEFF}/exafs/chi.dat", "Corvus.cfavg_xas.out"):
        assert (ret / rel).read_bytes() == (s / rel).read_bytes()

    listed = subprocess.run(["tar", "-I", "zstd", "-tf", str(ret / f"{RUN_ID}-corvus-xas.tar.zst")],
                            check=True, capture_output=True, text=True).stdout.split()
    assert f"{FEFF}/exafs/paths.dat" in listed and f"{FEFF}/exafs/log1.dat" in listed and "Corvus.nest" in listed
    # Regenerable FEFF temporaries, the heartbeat and the return dir itself are not packed.
    assert not [m for m in listed if m.endswith(("dmdw.out", ".bin")) and not m.endswith("/gg.bin")
                or m.startswith((".return", ".heartbeat"))]
    # ...except gg.bin, the FMS output (odd-XANES groups A and B are only visible in it).
    assert f"{FEFF}/xanes/gg.bin" in listed


def test_pack_and_cleanup_keep_the_same_files():
    from xas_pipeline.stages import cleanup
    assert corvus_pack.PACK_KEEP == cleanup.FEFF_KEEP
    assert corvus_pack.PACK_EXCLUDE == cleanup.FEFF_SCRATCH_GLOBS


def test_paths_table_maps_atoms_and_keeps_used_paths(tmp_path):
    s = _scratch(tmp_path)
    ret = corvus_pack.build_return(s, RUN_ID)
    rows = [line.split("\t") for line in (ret / f"feff-paths-{RUN_ID}.tsv").read_text().splitlines()
            if not line.startswith("#")]
    header, body = rows[0], rows[1:]
    assert header == ["path", "nleg", "degeneracy", "reff_A", "sig2_A2", "amp_ratio", "atoms", "elements"]
    by_path = {r[0]: dict(zip(header, r, strict=True)) for r in body}
    assert sorted(by_path) == ["1", "2"]  # path 3 was not used by FEFF
    assert by_path["1"]["atoms"] == "0" and by_path["1"]["elements"] == "S" and by_path["1"]["sig2_A2"] == "0.00246"
    assert by_path["2"]["atoms"] == "2,3" and by_path["2"]["elements"] == "N,C"


def test_unmatched_leg_is_minus_one(tmp_path):
    s = _scratch(tmp_path)
    paths = s / FEFF / "exafs" / "paths.dat"
    paths.write_text(paths.read_text().replace("    2.000000    0.000000    0.000000   3 'C", "    2.500000    0.000000    0.000000   3 'C"))
    ret = corvus_pack.build_return(s, RUN_ID)
    table = (ret / f"feff-paths-{RUN_ID}.tsv").read_text()
    assert "\t2,-1\tN,C\n" in table


def test_failed_run_is_still_packed(tmp_path):
    """No spectra (CORVUS failed): the logs still come back, in the archive, so the failure can be read."""
    s = tmp_path / "scratch"
    (s / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF" / "xanes").mkdir(parents=True)
    (s / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF" / "xanes" / "log2.dat").write_text("MOVRLP-1\n")
    (s / f"corvus-{RUN_ID}-xas.in").write_text("input\n")
    ret = corvus_pack.build_return(s, RUN_ID)
    assert sorted(p.name for p in ret.iterdir()) == [f"{RUN_ID}-corvus-xas-node.txt", f"{RUN_ID}-corvus-xas.tar.zst"]


def test_main_reports_failure(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert corvus_pack.main([str(empty), "--run-id", RUN_ID]) == 1
