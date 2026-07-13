"""Shared pytest fixtures and the loader that makes the scripts importable.

Most scripts in this repo have hyphenated filenames (``prepare-orca.py``,
``script-process-feff-output.py``) which cannot be imported with a plain
``import`` statement (``-`` is not valid in an identifier). The repo already
works around this at runtime with ``importlib.util`` (see rerun-corvus.py and
submit-corvus-only.py); :func:`load_script` centralizes the same trick for
tests so we can unit-test the pure helper functions inside those scripts
*without* renaming any files first.

This keeps the initial test suite refactor-agnostic: the eventual reorg into a
proper package can rename modules freely, and only this loader needs to change.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(filename: str):
    """Import a (possibly hyphen-named) top-level script as a module object.

    The module is registered in ``sys.modules`` under a sanitized name before
    execution so that dataclasses / ``from __future__ import annotations``
    resolution works, mirroring the runtime loaders in the repo.
    """
    path = REPO_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"script not found: {path}")
    mod_name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # register before exec (see docstring)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def run_script(filename: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a script as a subprocess (CLI-level / golden test entry point).

    Uses the same interpreter running the tests. The directory of that
    interpreter is prepended to PATH so scripts that shell out to a bare
    ``python`` (e.g. run-batch-pipeline invoking prepare-orca) resolve to the
    test venv rather than failing on a login node that only has ``python3``.
    Returns the CompletedProcess with captured stdout/stderr; callers assert on
    returncode + output. Prefer dry-run / --no-submit flags so nothing hits the
    scheduler.
    """
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / filename), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def run_cli():
    """Fixture wrapper around :func:`run_script` for CLI/golden tests."""
    return run_script
