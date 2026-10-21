"""Shared pytest fixtures and the CLI runner used by the golden tests.

Post-reorg (phase 9) the pipeline is an installed package: unit tests import
``xas_pipeline.*`` directly, and the CLI/golden tests invoke the stages via
``python -m xas_pipeline...`` (see :data:`_SCRIPT_MODULES`). The old hyphen
scripts and the importlib-by-path loader they used to need are gone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Historical script filename -> package module invoked via `python -m`. The
# golden tests still refer to the old filenames; this keeps their call sites
# unchanged while routing to the package entry points. All entry points now live
# in the package; a filename not listed would fall back to a top-level file of
# that name (none remain).
_SCRIPT_MODULES = {
    "run-batch-pipeline.py": "xas_pipeline.orchestrate",
    "prepare-orca.py": "xas_pipeline.stages.orca_prep",
    "prepare-corvus.py": "xas_pipeline.stages.corvus_prep",
    "script-check-orca-convergence-and-extract-times.py": "xas_pipeline.stages.orca_check",
    "script-process-feff-output.py": "xas_pipeline.stages.feff_process",
    "script-prepare-files-for-download.py": "xas_pipeline.stages.download",
    "script-cleanup-calc-artifacts.py": "xas_pipeline.stages.cleanup",
    "rerun-corvus.py": "xas_pipeline.cli.rerun_corvus",
    "submit-corvus-only.py": "xas_pipeline.cli.submit_corvus",
    "script-count-imag-freq.py": "xas_pipeline.stages.count_imag_freq",
}


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
    module = _SCRIPT_MODULES.get(filename)
    if module is not None:
        cmd = [sys.executable, "-m", module, *args]
    else:
        cmd = [sys.executable, str(REPO_ROOT / filename), *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def run_cli():
    """Fixture wrapper around :func:`run_script` for CLI/golden tests."""
    return run_script


# ── Golden snapshots ──────────────────────────────────────────────────────────

FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_ROOT = FIXTURES / "golden"
GOLDEN_UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


def normalize(text: str, paths: dict[str, Path]) -> str:
    """Replace machine-specific paths with stable ``<TOKEN>`` placeholders.

    Longer paths are substituted first so a path nested in another keeps its own
    token.
    """
    for token, path in sorted(paths.items(), key=lambda item: -len(str(item[1]))):
        text = text.replace(str(path), f"<{token}>")
    return text


def assert_matches_golden(golden_dir: Path, produced: dict[str, str]) -> None:
    """Compare ``{relative path: text}`` with the snapshot tree at *golden_dir*.

    The file *set* must match as well as each file's content. With
    ``GOLDEN_UPDATE=1`` the snapshot tree is rewritten instead (review the diff:
    it is the behaviour change).
    """
    if GOLDEN_UPDATE:
        if golden_dir.exists():
            for old in golden_dir.rglob("*"):
                if old.is_file():
                    old.unlink()
        for rel, text in produced.items():
            path = golden_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        pytest.skip(f"GOLDEN_UPDATE=1: wrote {len(produced)} snapshot(s) to {golden_dir.name}")

    expected = {
        str(p.relative_to(golden_dir)): p.read_text(encoding="utf-8")
        for p in sorted(golden_dir.rglob("*"))
        if p.is_file()
    }
    assert expected, f"no golden snapshots under {golden_dir}; run with GOLDEN_UPDATE=1"
    assert sorted(produced) == sorted(expected), (
        f"file set drifted from {golden_dir}:\n"
        f"  only produced: {sorted(set(produced) - set(expected))}\n"
        f"  only golden:   {sorted(set(expected) - set(produced))}"
    )
    drift = [rel for rel in produced if produced[rel] != expected[rel]]
    assert not drift, f"content drift from {golden_dir}: {drift}"


def run_module(module: str, *args: str, cwd: Path | None = None, env: dict | None = None):
    """Run ``python -m <module> args`` with the test interpreter first on PATH."""
    full_env = dict(os.environ if env is None else env)
    full_env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), full_env.get("PATH", "")])
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
    )
