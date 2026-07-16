"""Pipeline stages: the I/O shells that drive one step of the workflow.

Each module owns one stage's file-producing logic (ORCA prep, Corvus prep, ORCA
convergence check, FEFF post-processing, download staging, cleanup) plus a
``main()`` argparse entry point. The top-level hyphen scripts are thin shims that
delegate here (see REORG.md phase 8); they are retired for console_scripts in
phase 9.
"""

from __future__ import annotations
