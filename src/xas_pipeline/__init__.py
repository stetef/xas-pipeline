"""xas_pipeline: ORCA geometry optimization -> CORVUS/FEFF XANES/EXAFS spectra.

Python-orchestrated batch pipeline that submits ORCA and CORVUS jobs to SLURM
(or PBS) and post-processes the results. This package is the refactor target of
the former flat collection of top-level scripts; modules are moved in here
incrementally while a characterization test suite keeps behavior fixed.
"""

from __future__ import annotations

__version__ = "0.1.0"
