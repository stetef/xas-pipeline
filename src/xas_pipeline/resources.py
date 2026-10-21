"""Where the packaged data lives, and where generated job scripts look for site config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "data"


def template_root() -> Path:
    """Packaged templates: orca-templates/, {slurm,pbs}-scripts/, corvus-*.in."""
    return DATA_ROOT


def interp_ligand_root() -> Path:
    """The packaged ligand ``.interp`` spring models (the spring-Hessian default)."""
    return DATA_ROOT / "interp-ligands"


def interp_ligand_files() -> list[Path]:
    """Every packaged ``.interp`` file, sorted. All are used: the two histidine
    files are not alternatives (each ring nitrogen coordinates differently)."""
    return sorted(interp_ligand_root().glob("*.interp"))


def pipeline_root() -> Path:
    """The directory whose ``.env`` (and ``.venv``) generated job scripts use.

    ``$PIPELINE_ROOT`` if set; else the source checkout this package runs from
    (an editable install); else the parent of the active virtualenv.
    """
    override = os.environ.get("PIPELINE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path(sys.prefix).resolve().parent


def pipeline_env() -> Path:
    """The site-config ``.env`` generated job scripts source."""
    return pipeline_root() / ".env"
