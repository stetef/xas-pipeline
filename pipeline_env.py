"""Tiny, dependency-free loader for the pipeline's site-config .env file.

The bash job/wrapper templates source ``.env`` directly (``set -a; source .env``).
This module gives the Python entry points the same view of those variables when
they are run on a login node without the wrapper having exported them first
(e.g. running prepare-corvus.py by hand).

Format: plain ``KEY=VALUE`` lines (an optional leading ``export`` is tolerated),
``#`` comments and blank lines ignored. Existing environment values win, so a
variable already exported by the shell/scheduler is never clobbered.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: "str | Path | None" = None) -> Path | None:
    """Load KEY=VALUE pairs from `.env` into os.environ (without overriding).

    Returns the path that was loaded, or None if no file was found.
    """
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not path.is_file():
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
