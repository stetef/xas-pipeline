"""Characterization test: `script-prepare-files-for-download.py`.

Pins the survivor-copy / CORVUS-failed-quarantine behavior before the reorg
rewires this script through xas_pipeline.layout. Structural assertions (the
script moves/copies dirs; there is no text artifact to snapshot).

Run with cwd == the batch root, so the current cwd-relative `downloading-station`
and `failed-corvus` land under the batch (this also keeps the test stable under
the planned issue-#3 fix, which anchors failed-corvus/ at the batch root).
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def download_run(tmp_path_factory):
    from conftest import run_script

    batch = tmp_path_factory.mktemp("dl") / "batch-out"
    batch.mkdir(parents=True)
    (batch / "corvus-failed-ids.txt").write_text("run-failed\n", encoding="utf-8")

    out = batch / "run-ok" / "output-run-ok"
    out.mkdir(parents=True)
    (out / "xanes-run-ok.dat").write_text("1.0 2.0\n", encoding="utf-8")

    failed = batch / "run-failed" / "working-run-failed"
    failed.mkdir(parents=True)
    (failed / "note.txt").write_text("x", encoding="utf-8")

    result = run_script("script-prepare-files-for-download.py", str(batch), cwd=batch)
    return {"result": result, "batch": batch}


def test_exits_clean(download_run):
    result = download_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_survivor_copied_to_download_station(download_run):
    batch = download_run["batch"]
    copied = batch / "downloading-station" / "run-ok" / "xanes-run-ok.dat"
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == "1.0 2.0\n"


def test_failed_id_quarantined(download_run):
    batch = download_run["batch"]
    assert (batch / "failed-corvus" / "run-failed").is_dir()
    assert not (batch / "run-failed").exists()  # moved out of the batch
