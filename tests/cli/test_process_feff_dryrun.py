"""Integration golden test: `script-process-feff-output.py`.

Processes a completed CORVUS run (split layout: <id>/working-<id>/ with
Corvus3_cfavg_<mode>/Corvus1Zn_FEFF trees) and populates output-<id>/ with the
per-mode xmu copies, the chi(R) FFT, the configurationally-averaged spectra, and
the structure xyz. It also renders PNGs.

What we freeze, and why:
- ``chi-R-<id>.dat`` is byte-snapshotted: it is the only *computed* output (the
  larch chi(k)->chi(R) FFT with default params), so it is the real regression
  target. np.savetxt gives deterministic %.18e columns + a fixed header.
- The other output files are plain copies, so we assert byte-identity to their
  sources instead of carrying redundant golden bytes.
- PNGs are non-deterministic (font/metadata), so we assert only that they exist.

Regenerate the chi-R golden after an intentional change with:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/test_process_feff_dryrun.py

larch and matplotlib must be importable (they are in the pinned .venv).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
RUN_ID = "2j6a_ZN_homo_d2.60_cluster1"
FIXTURE_SYS = FIXTURES / "feff_run" / RUN_ID
GOLDEN_DIR = FIXTURES / "golden" / "process-feff"
CHI_R_NAME = f"chi-R-{RUN_ID}.dat"

UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


@pytest.fixture(scope="module")
def feff_run(tmp_path_factory, repo_root):
    tmp = tmp_path_factory.mktemp("feff")
    sys_dir = tmp / RUN_ID
    shutil.copytree(FIXTURE_SYS, sys_dir)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["MPLBACKEND"] = "Agg"  # headless render

    result = subprocess.run(
        [sys.executable, "-m", "xas_pipeline.stages.feff_process",
         str(sys_dir), "--no-batch-log"],
        capture_output=True, text=True, env=env,
    )
    working = sys_dir / f"working-{RUN_ID}"
    return {
        "result": result,
        "sys_dir": sys_dir,
        "working": working,
        "output": sys_dir / f"output-{RUN_ID}",
    }


def test_exits_clean(feff_run):
    result = feff_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_output_dir_has_expected_files(feff_run):
    out = feff_run["output"]
    expected = {
        f"xmu-xanes-{RUN_ID}.dat",
        f"xmu-exafs-{RUN_ID}.dat",
        CHI_R_NAME,
        f"xanes-{RUN_ID}.dat",
        f"exafs-{RUN_ID}.dat",
        f"{RUN_ID}.xyz",
    }
    produced = {p.name for p in out.iterdir() if p.is_file()}
    assert expected <= produced, f"missing outputs: {expected - produced}"


def test_copied_outputs_are_byte_identical_to_sources(feff_run):
    out, working = feff_run["output"], feff_run["working"]
    pairs = [
        (out / f"xmu-xanes-{RUN_ID}.dat",
         working / "Corvus3_cfavg_xanes" / "Corvus1Zn_FEFF" / "xmu.dat"),
        (out / f"xmu-exafs-{RUN_ID}.dat",
         working / "Corvus3_cfavg_exafs" / "Corvus1Zn_FEFF" / "xmu.dat"),
        (out / f"xanes-{RUN_ID}.dat", working / "Corvus.cfavg_xanes.out"),
        (out / f"exafs-{RUN_ID}.dat", working / "Corvus.cfavg_exafs.out"),
        (out / f"{RUN_ID}.xyz", working / f"{RUN_ID}.xyz"),
    ]
    mismatches = [dst.name for dst, src in pairs if dst.read_bytes() != src.read_bytes()]
    assert not mismatches, f"copied output diverged from source: {mismatches}"


def test_pngs_rendered(feff_run):
    working = feff_run["working"]
    xanes_feff = working / "Corvus3_cfavg_xanes" / "Corvus1Zn_FEFF"
    exafs_feff = working / "Corvus3_cfavg_exafs" / "Corvus1Zn_FEFF"
    # xmu.dat present in both -> XANES + EXAFS plots in both; chi.dat only in
    # exafs -> chi_R.png only there.
    for feff in (xanes_feff, exafs_feff):
        assert (feff / "xanes_K.png").is_file()
        assert (feff / "exafs_K.png").is_file()
    assert (exafs_feff / "chi_R.png").is_file()


def test_chi_r_matches_golden(feff_run):
    produced = (feff_run["output"] / CHI_R_NAME).read_text(encoding="utf-8")
    golden = GOLDEN_DIR / CHI_R_NAME
    if UPDATE:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(produced, encoding="utf-8")
        pytest.skip("GOLDEN_UPDATE=1: wrote chi-R golden")
    assert golden.is_file(), "missing chi-R golden; run with GOLDEN_UPDATE=1 to create it"
    assert produced == golden.read_text(encoding="utf-8"), "chi(R) FFT output drifted from golden"
