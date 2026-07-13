"""Minimal characterization test: `script-count-imag-freq.py`.

count-imag-freq has no argparse (raw sys.argv) and no prior coverage; this pins
its core behavior (count the imaginary-frequency marker per id, write a CSV)
before the reorg modernizes it. Structural assertions, since id iteration order
is not guaranteed.
"""

import csv
from pathlib import Path

import pytest

MARKER = "Found imaginary frequency with large weight"


@pytest.fixture(scope="module")
def count_run(tmp_path_factory):
    from conftest import run_script

    family = tmp_path_factory.mktemp("imag") / "family"
    for run_id, hits in (("run1", 2), ("run2", 0)):
        working = family / run_id / f"working-{run_id}"
        working.mkdir(parents=True)
        body = "\n".join([f"line {i}" for i in range(3)] + [MARKER] * hits) + "\n"
        (working / f"corvus-{run_id}.out").write_text(body, encoding="utf-8")

    result = run_script("script-count-imag-freq.py", str(family))
    return {"result": result, "family": family}


def test_exits_clean(count_run):
    result = count_run["result"]
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_csv_counts(count_run):
    csv_path = count_run["family"] / "imaginary_frequencies.csv"
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["cluster", "imaginary_freq_count"]
    counts = {name: value for name, value in rows[1:]}
    assert counts["run1"] == "2"
    assert counts["run2"] == "0"
