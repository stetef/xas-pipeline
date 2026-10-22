"""Shared pytest fixtures and the CLI runner used by the golden tests.

Unit tests import ``xas_pipeline.*`` directly; CLI and golden tests run the
entry points as ``python -m xas_pipeline...`` (:func:`run_script`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(autouse=True)
def _restore_environ():
    """CLI code may load the site .env into os.environ; keep each test's changes to itself."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def run_script(module: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m <module> args`` (CLI-level / golden test entry point).

    Uses the test interpreter, with its directory first on PATH. Returns the
    CompletedProcess with captured stdout/stderr; prefer dry-run / --no-submit
    flags so nothing reaches a scheduler.
    """
    return run_module(module, *args, cwd=cwd)


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
    token. The test interpreter, rendered into every job script, is always ``<PYTHON>``.
    """
    paths = {"PYTHON": Path(sys.executable), **paths}
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
