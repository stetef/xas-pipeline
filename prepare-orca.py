#!/usr/bin/env python3
"""Thin entry-point shim -> xas_pipeline.stages.orca_prep.

Stage logic moved into the package (REORG.md phase 8b). Keeps the old filename
working as a CLI entry point and re-exports the stage module's names for
importlib-by-path consumers (e.g. the characterization unit tests reach
`orca._format_index_ranges`, `orca.orca_maxcore_mb`). Retired for a
console_script in phase 9.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline.stages import orca_prep as _stage

globals().update({k: v for k, v in vars(_stage).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_stage.main())
