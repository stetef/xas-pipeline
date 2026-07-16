#!/usr/bin/env python3
"""Thin entry-point shim -> xas_pipeline.stages.corvus_prep.

Stage logic moved into the package (REORG.md phase 8b). Keeps the old filename
working as a CLI entry point (the corvus wrapper + run-batch invoke it by path)
and re-exports the stage module's names for importlib-by-path consumers (e.g. the
characterization tests reach `corvus._atomic_number_from_token`). Retired for a
console_script in phase 9.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline.stages import corvus_prep as _stage

globals().update({k: v for k, v in vars(_stage).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_stage.main())
