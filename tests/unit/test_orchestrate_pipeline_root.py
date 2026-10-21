"""resources.pipeline_root: where generated job scripts find .env and .venv.

Regression for project_root(), which assumed an editable install (the package
three levels below a checkout). Installed into a venv's site-packages, that
pointed into ``lib/pythonX.Y`` and every job script sourced a .env that does
not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

from xas_pipeline import jobscripts, resources
from xas_pipeline.scheduler import get_scheduler


def _site_packages_install(tmp_path: Path, monkeypatch) -> Path:
    """Make resources look installed (non-editable) into <tmp>/proj/.venv."""
    project = tmp_path / "proj"
    venv = project / ".venv"
    module = venv / "lib" / "python3.12" / "site-packages" / "xas_pipeline" / "resources.py"
    module.parent.mkdir(parents=True)
    monkeypatch.delenv("PIPELINE_ROOT", raising=False)
    monkeypatch.setattr(resources, "__file__", str(module))
    monkeypatch.setattr(sys, "prefix", str(venv))
    return project


def test_non_editable_install_uses_the_venvs_project(tmp_path, monkeypatch):
    project = _site_packages_install(tmp_path, monkeypatch)
    assert resources.pipeline_root() == project.resolve()
    assert resources.pipeline_env() == project.resolve() / ".env"


def test_editable_checkout_is_its_own_root(monkeypatch):
    monkeypatch.delenv("PIPELINE_ROOT", raising=False)
    root = resources.pipeline_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "xas_pipeline" / "resources.py").resolve() == Path(resources.__file__).resolve()


def test_environment_override_wins(tmp_path, monkeypatch):
    _site_packages_install(tmp_path, monkeypatch)
    monkeypatch.setenv("PIPELINE_ROOT", str(tmp_path / "site"))
    assert resources.pipeline_root() == (tmp_path / "site").resolve()


def test_job_scripts_carry_the_installed_root(tmp_path, monkeypatch):
    project = _site_packages_install(tmp_path, monkeypatch)
    batch = tmp_path / "batch"
    batch.mkdir()
    text = jobscripts.postprocess(get_scheduler("slurm"), batch, batch / "dl").read_text(encoding="utf-8")
    assert f'PIPELINE_ROOT="{project.resolve()}"\n' in text
    assert f'PIPELINE_ENV="{project.resolve() / ".env"}"' in text
    assert "site-packages" not in text
