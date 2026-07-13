"""Seed CLI/golden smoke test: proves the subprocess harness works.

Each script is a standalone CLI with argparse. `--help` exits 0 and prints a
usage line without importing heavy/optional deps. This is the thinnest slice of
the golden-test approach that will later snapshot real dry-run outputs; for now
it just verifies scripts are invokable and their CLIs parse.
"""

import pytest

SCRIPTS_WITH_CLI = [
    "prepare-orca.py",
    "prepare-corvus.py",
    "run-batch-pipeline.py",
    "submit-corvus-only.py",
    "script-cleanup-calc-artifacts.py",
    "script-prepare-files-for-download.py",
]


@pytest.mark.parametrize("script", SCRIPTS_WITH_CLI)
def test_help_exits_zero(run_cli, script):
    result = run_cli(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
