"""Numerical snapshots of the spring-model route (interpolation and database build).

Pinned on two real clusters -- a Zn(Cys)4 site and a 2Cys2His site, so both the
cysteine and histidine ligand models are matched:

* ``spring.model`` (interpolated constants) and the raw spring Hessian: pure
  arithmetic, compared byte for byte;
* the default Hessian, whose eigenvalue floor goes through LAPACK: compared
  numerically, as are its frequencies;
* ``.interp`` + both spring files built from a real ORCA ``.hess`` pair.

Regenerate after an intentional change with ``GOLDEN_UPDATE=1``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, GOLDEN_ROOT, GOLDEN_UPDATE, assert_matches_golden, normalize

from xas_pipeline.chem import spring_model
from xas_pipeline.chem.hessian import read_orca_hessian
from xas_pipeline.stages.interp_hessian import SPRING_MODEL_NAME, build_interp_hessian

CLUSTERS = {
    "cys4": FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz",
    "cys2his2": FIXTURES / "xyz_his" / "4j1v_cluster2_Zn.xyz",
}
ORCA_HESS = FIXTURES / "corvus_run" / "2j6a_ZN_homo_d2.60_cluster1" / "2j6a_ZN_homo_d2.60_cluster1.hess"
GOLDEN = GOLDEN_ROOT / "springs"
RUN_ID = "site-asis-spring"


def _run(tmp_path: Path, xyz: Path, **options) -> tuple[Path, object]:
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    shutil.copy2(xyz, run_dir / f"{RUN_ID}.xyz")
    result = build_interp_hessian(run_dir, RUN_ID, **options)
    return run_dir, result


@pytest.mark.parametrize("cluster", list(CLUSTERS))
def test_interpolated_springs_and_raw_hessian_are_unchanged(tmp_path, cluster):
    run_dir, _ = _run(tmp_path, CLUSTERS[cluster], min_freq_scale=0.0, freq_check=False)
    assert_matches_golden(
        GOLDEN / cluster / "raw",
        {
            SPRING_MODEL_NAME: (run_dir / SPRING_MODEL_NAME).read_text(),
            f"{RUN_ID}.hess": (run_dir / f"{RUN_ID}.hess").read_text(),
        },
    )


@pytest.mark.parametrize("cluster", list(CLUSTERS))
def test_default_hessian_and_frequencies_are_unchanged(tmp_path, cluster):
    run_dir, result = _run(tmp_path, CLUSTERS[cluster])
    hessian, _ = read_orca_hessian(run_dir / f"{RUN_ID}.hess")
    summary = {
        "n_imaginary": result.n_imaginary,
        "n_imaginary_raw": result.n_imaginary_raw,
        "wavenumbers": [float(w) for w in result.wavenumbers],
    }
    golden_dir = GOLDEN / cluster / "default"
    if GOLDEN_UPDATE:
        golden_dir.mkdir(parents=True, exist_ok=True)
        np.save(golden_dir / "hessian.npy", hessian)
        (golden_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        pytest.skip("GOLDEN_UPDATE=1")

    expected = json.loads((golden_dir / "summary.json").read_text())
    assert summary["n_imaginary"] == expected["n_imaginary"]
    assert summary["n_imaginary_raw"] == expected["n_imaginary_raw"]
    np.testing.assert_allclose(summary["wavenumbers"], expected["wavenumbers"], atol=1e-6)
    golden_h = np.load(golden_dir / "hessian.npy")
    np.testing.assert_allclose(hessian, golden_h, rtol=0, atol=1e-12 * np.abs(golden_h).max())


def test_interp_built_from_orca_hessians_is_unchanged(tmp_path):
    # A real ORCA Hessian as the natural state and the spring Hessian of the same
    # cluster as the "modified" one: same atoms, same order, different constants.
    run_dir, _ = _run(tmp_path, CLUSTERS["cys4"], min_freq_scale=0.0, freq_check=False)
    db = tmp_path / "db"
    spring_model.build_interp_from_hess(ORCA_HESS, run_dir / f"{RUN_ID}.hess", db / "site.interp")

    paths = {"DB": db, "RUN": run_dir, "FIXTURES": FIXTURES}
    assert_matches_golden(
        GOLDEN / "build-interp",
        {path.name: normalize(path.read_text(), paths) for path in sorted(db.iterdir())},
    )
