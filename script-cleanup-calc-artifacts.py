#!/usr/bin/env python3
"""Thin entry-point shim -> xas_pipeline.stages.cleanup.

The stage logic moved into the package (REORG.md phase 8b). This shim keeps the
old filename working as a CLI entry point and re-exports the stage module's names
so the importlib-by-path test/consumers keep resolving them. Retired for a
console_script in phase 9.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline.stages import cleanup as _stage

# Re-export the stage module's public + private names (excluding dunders) so
# `importlib`-by-path consumers that reach into this file keep working.
globals().update({k: v for k, v in vars(_stage).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_stage.main())
