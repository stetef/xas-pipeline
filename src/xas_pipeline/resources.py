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
