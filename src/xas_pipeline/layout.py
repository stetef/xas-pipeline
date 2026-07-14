"""Batch directory layout conventions, shared across the scanning stages.

Before the reorg each stage (process-feff, orca-convergence-check, download,
cleanup, rerun-corvus) re-implemented the same primitives: which sibling dirs to
skip, how to iterate the per-id run directories, how to detect the split
``working-<id>/`` + ``output-<id>/`` layout, and the move-aside-with-stale-
replacement used for quarantining. They live here now.

Conventions (unchanged): the run id is the directory name; a batch may be
"flat" (``<id>/`` holds everything) or "split" (``<id>/working-<id>/`` for the
live calc, ``<id>/output-<id>/`` for captured spectra).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Iterator

# Sibling directories under a batch root that are never per-id run directories.
SKIP_DIR_NAMES = frozenset(
    {
        "failed-orca",
        "failed-corvus",
        "downloading-station",
        "xyz_files",
        "optimized_xyz_files",
    }
)


def iter_id_dirs(
    parent_dir: Path,
    *,
    skip: Iterable[str] = SKIP_DIR_NAMES,
    only_ids: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield first-level per-id directories under ``parent_dir`` (sorted).

    Skips ``skip`` (the helper/quarantine dirs) and, when ``only_ids`` is given,
    restricts to those id names. Callers apply their own "is this really a run
    dir?" predicate to the yielded candidates.
    """
    skip = set(skip)
    only = set(only_ids) if only_ids else None
    for child in sorted(Path(parent_dir).iterdir()):
        if not child.is_dir() or child.name in skip:
            continue
        if only is not None and child.name not in only:
            continue
        yield child


def has_working_output_pair(system_dir: Path) -> bool:
    """True when ``system_dir`` is a split layout (both working-/output- exist)."""
    name = Path(system_dir).name
    return (system_dir / f"working-{name}").is_dir() and (system_dir / f"output-{name}").is_dir()


def working_roots(system_dir: Path) -> list[Path]:
    """Roots under which Corvus3_cfavg_<mode> dirs may live (flat or split)."""
    system_dir = Path(system_dir)
    roots = [system_dir]
    if system_dir.is_dir():
        roots.extend(
            child
            for child in system_dir.iterdir()
            if child.is_dir() and child.name.startswith("working")
        )
    return roots


def quarantine_move(src_dir: Path, dest_parent: Path) -> Path:
    """Move ``src_dir`` into ``dest_parent/<name>``, replacing any stale copy."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    destination = dest_parent / Path(src_dir).name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(src_dir), str(destination))
    return destination
