"""Logging setup for the command-line entry points.

Library code logs to the ``xas_pipeline`` logger and never prints. Each CLI
``main()`` calls :func:`setup_cli_logging` so those messages reach stdout (the
job log) as plain lines; a library caller configures logging as it likes.
"""

from __future__ import annotations

import logging
import sys


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Send ``xas_pipeline`` log records to stdout as bare messages (idempotent)."""
    logger = logging.getLogger("xas_pipeline")
    if any(getattr(h, "_xas_cli", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._xas_cli = True
    logger.addHandler(handler)
    logger.setLevel(level)
