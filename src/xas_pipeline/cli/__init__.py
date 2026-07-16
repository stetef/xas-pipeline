"""Command-line entry points that compose the pipeline stages/orchestrator.

rerun_corvus and submit_corvus reuse the orchestrator's tested wrapper-templating
and job-id parsing (``xas_pipeline.orchestrate``) rather than re-implementing it.
Exposed as console_scripts (xas-rerun-corvus, xas-submit-corvus) and runnable via
``python -m xas_pipeline.cli.<name>``.
"""

from __future__ import annotations
