"""Locate the packaged bash templates (ORCA/Corvus input + job/wrapper scripts).

The templates ship as package data under ``xas_pipeline/data/`` so an installed
package (and the run-from-checkout src/ layout) finds them the same way, without
depending on a repo-root layout. Site config (``.env``) is *not* here -- that is
resolved separately from the cwd/parents (see :mod:`xas_pipeline.config`).
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "data"


def template_root() -> Path:
    """Directory holding orca-templates/, slurm-scripts/, pbs-scripts/, corvus-*.in."""
    return DATA_ROOT


def interp_ligand_root() -> Path:
    """Directory holding the pre-built per-ligand ``.interp`` constants files.

    These back the ``--interp`` optimization mode: instead of an ORCA analytic
    frequency run, the Hessian is interpolated from these ligand spring models
    (see :mod:`xas_pipeline.chem.springs`).
    """
    return DATA_ROOT / "interp-ligands"


def interp_ligand_files() -> list[Path]:
    """Every packaged ``.interp`` ligand file, sorted by name.

    All of them are used by default. The two histidine files are not
    alternatives: the two ring nitrogens coordinate differently, so each gets
    its own interpolation and both must be searched for.
    """
    return sorted(interp_ligand_root().glob("*.interp"))


def project_root() -> Path:
    """Repo checkout root -- holds ``.env`` and the top-level entry-point scripts.

    Transitional. Only meaningful in the run-from-checkout / editable-install
    layout (``src/xas_pipeline/`` under the repo). It exists so stage code that
    moved into the package can still emit the same ``<repo>/.env`` and sibling-
    script paths the flat scripts did, keeping the golden output byte-identical.
    Retired in phase 9, when ``.env`` is resolved via :func:`config.find_dotenv`
    and the sibling scripts become console_scripts.
    """
    return Path(__file__).resolve().parents[2]  # src/xas_pipeline -> src -> <repo>
