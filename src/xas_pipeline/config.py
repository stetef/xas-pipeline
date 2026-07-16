"""Site-config ``.env`` loading for the pipeline's Python entry points.

The bash job/wrapper templates source ``.env`` directly (``set -a; source .env``).
This module gives the Python entry points the same view of those variables when
run on a login node without the wrapper having exported them first (e.g. running
prepare-corvus by hand).

Format: plain ``KEY=VALUE`` lines (an optional leading ``export`` is tolerated),
``#`` comments and blank lines ignored. Existing environment values win, so a
variable already exported by the shell/scheduler is never clobbered.

(Formerly the top-level ``pipeline_env.py``; moved into the package during the
reorg. The default-path behavior changed from "the .env next to this module" to
"search cwd upward" so it works when installed rather than run from a checkout;
the transitional top-level scripts pass an explicit repo-root path.)
"""

from __future__ import annotations

import os
from pathlib import Path

# Regenerable ORCA scratch artifacts (glob patterns). Single source of truth for
# fix #8: the ORCA job script excludes these from its copy-back out of scratch
# (injected as the [SCRATCH_EXCLUDE] placeholder) AND the cleanup stage deletes
# them, so they are never dragged back only to be removed later. `.bin` is ORCA
# scratch (MO integrals etc.), not the kept .gbw restart.
SCRATCH_EXCLUDE_GLOBS = (
    "*.densities",
    "*.densitiesinfo",
    "*.cpcm",
    "*.cpcm_corr",
    "*.engrad",
    "*.bin",
)


def find_dotenv(start: Path | None = None) -> Path | None:
    """Search ``start`` (default: cwd) and its parents for a ``.env`` file."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(env_path: "str | Path | None" = None) -> Path | None:
    """Load KEY=VALUE pairs from a ``.env`` into ``os.environ`` (without override).

    With ``env_path`` given, that file is loaded. Otherwise the nearest ``.env``
    at or above the current working directory is used. Returns the path that was
    loaded, or ``None`` if no file was found.
    """
    if env_path is not None:
        path = Path(env_path)
        if not path.is_file():
            return None
    else:
        path = find_dotenv()
        if path is None:
            return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (len(val) >= 2) and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)
    return path
