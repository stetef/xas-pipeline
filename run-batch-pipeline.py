#!/usr/bin/env python3
"""Thin entry-point shim -> xas_pipeline.orchestrate.

The batch-pipeline orchestration core moved into the package (REORG.md phase 8c).
This shim keeps the old filename working as a CLI entry point and re-exports the
module's names so the importlib-by-path consumers keep resolving them:
  - rerun-corvus.py / submit-corvus-only.py load this file and reuse
    `_write_corvus_wrapper_script`, `_submit_job`, `_parse_submitted_job_id`, ...
  - the characterization tests reach `rbp._parse_submitted_job_id`.
Retired for a console_script in phase 9.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline import orchestrate as _stage

globals().update({k: v for k, v in vars(_stage).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_stage.main())
