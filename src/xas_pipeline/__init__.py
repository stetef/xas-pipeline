"""xas_pipeline: ORCA geometry optimization -> CORVUS/FEFF XANES/EXAFS spectra.

A batch pipeline that submits ORCA and CORVUS jobs to Slurm (or PBS) and
post-processes the results. :mod:`xas_pipeline.chem` holds the chemistry,
:mod:`xas_pipeline.stages` the per-step logic, :mod:`xas_pipeline.cli` the
batch-level commands; :mod:`xas_pipeline.orchestrate` ties them together.
"""

from __future__ import annotations

__version__ = "0.2.0"
