"""Logging setup for the command-line entry points.

Library code logs to ``xas_pipeline.*`` loggers and never prints. Each CLI
``main()`` calls :func:`setup_cli_logging` so those messages reach stdout (the
job log) as plain lines; a library caller configures logging as it likes.
"""

from __future__ import annotations

import logging
import sys

# Run as ``python -m xas_pipeline.<module>`` (as the job scripts do), the entry
# module logs under ``__main__``, not its package name, so both get the handler.
CLI_LOGGERS = ("xas_pipeline", "__main__")


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Send pipeline log records to stdout as bare messages (idempotent)."""
    for name in CLI_LOGGERS:
        logger = logging.getLogger(name)
        if any(getattr(h, "_xas_cli", False) for h in logger.handlers):
            continue
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._xas_cli = True
        logger.addHandler(handler)
        logger.setLevel(level)
