"""Batch directory layout conventions, shared across the scanning stages.

Before the reorg each stage (process-feff, orca-convergence-check, download,
cleanup, rerun-corvus) re-implemented the same primitives: which sibling dirs to
skip, how to iterate the per-id run directories, how to detect the split
``working-<id>/`` + ``output-<id>/`` layout, and the move-aside-with-stale-
replacement used for quarantining. They live here now.

Conventions: the run id is the run directory's name; a run dir may be "flat"
(everything directly inside) or "split" (``working-<run_id>/`` for the live calc,
``output-<run_id>/`` for captured spectra).

Run dirs are grouped by starting structure. Each ORCA optimization mode gets its
own run dir, named ``<id>-<mode>``, nested under a *group* directory named for
the structure, so several modes can be run from one XYZ without overwriting each
other and land side by side::

    batch-root/
      2j6a_ZN_cluster1/                        <- group dir (the structure)
        2j6a_ZN_cluster1-ca-fixed/             <- run dir; its name is the run id
        2j6a_ZN_cluster1-free/
        2j6a_ZN_cluster1-interp/
      downloading-station/

Older batches, whose run dirs sit directly under the batch root with no mode
suffix, still scan correctly: :func:`iter_id_dirs` descends into group dirs but
also yields flat run dirs unchanged.
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

# The ORCA optimization modes, and so the vocabulary of run-dir suffixes. Naming
# lives here because it is what makes "<id>-<mode>" parseable back into a mode
# (see mode_from_run_id); the mode -> template mapping stays in
# stages.orca_prep.TEMPLATE_FILE_BY_MODE, and a test asserts the two agree.
KNOWN_MODES = (
    "ca-fixed",
    "quick",
    "quick-ca-fixed",
    "h-only",
    "single-point",
    "no-constraints",
    "backbone",
    "xtb-free",
    "xtb-constrained",
    "interp",
)

# Historical spelling. h-only was the one mode that already suffixed its run dirs,
# as "<id>-H-only", before every mode got a suffix; batches on disk use that
# casing, so keep producing it rather than silently orphaning them.
_LEGACY_MODE_SUFFIX = {"h-only": "H-only"}


def mode_suffix(mode: str) -> str:
    """The run-dir/run-id suffix for an ORCA optimization mode."""
    return _LEGACY_MODE_SUFFIX.get(mode, mode)


def run_id_for(id_name: str, mode: str) -> str:
    """The run id (== run dir name, == artifact basename) for one structure+mode."""
    return f"{id_name}-{mode_suffix(mode)}"


def mode_from_run_id(run_id: str) -> str | None:
    """Recover the ORCA mode from a run id, or ``None`` if it encodes none.

    ``None`` means a run dir predating mode suffixes (a bare ``<id>``), where the
    mode is genuinely unknown -- callers should fall back to whatever is safe for
    an already-computed run rather than guessing.

    Longest suffix wins, so ``<id>-quick-ca-fixed`` resolves to ``quick-ca-fixed``
    rather than the ``ca-fixed`` it also ends with.
    """
    for mode in sorted(KNOWN_MODES, key=len, reverse=True):
        if run_id.endswith(f"-{mode_suffix(mode)}"):
            return mode
    return None


def group_dir_for(batch_root: Path, id_name: str) -> Path:
    """The per-structure group dir holding every mode's run dir."""
    return Path(batch_root) / id_name


def run_dir_for(batch_root: Path, id_name: str, mode: str) -> Path:
    """The run dir for one structure+mode: ``<batch_root>/<id>/<id>-<mode>``."""
    return group_dir_for(batch_root, id_name) / run_id_for(id_name, mode)


def is_group_dir(candidate: Path) -> bool:
    """True when ``candidate`` holds per-mode run dirs rather than being one.

    Deliberately structural rather than name-based, so it also classifies dirs
    this pipeline did not create. A group dir holds *only* subdirectories, at
    least one of which is prefixed with the group's own name, and none of which
    is a ``working-``/``output-`` pair member -- those belong to a run dir, and a
    cleaned-up split run dir can otherwise look empty of files too.
    """
    candidate = Path(candidate)
    if not candidate.is_dir():
        return False
    try:
        children = list(candidate.iterdir())
    except OSError:
        return False

    if any(child.is_file() for child in children):
        return False
    subdirs = [child for child in children if child.is_dir()]
    if not subdirs:
        return False
    if any(child.name.startswith(("working-", "output-")) for child in subdirs):
        return False
    return any(child.name.startswith(f"{candidate.name}-") for child in subdirs)


def iter_id_dirs(
    parent_dir: Path,
    *,
    skip: Iterable[str] = SKIP_DIR_NAMES,
    only_ids: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield the per-run directories under ``parent_dir`` (sorted).

    Descends one level into per-structure group dirs (see :func:`is_group_dir`),
    yielding their ``<id>-<mode>`` run dirs; a first-level child that is itself a
    run dir is yielded as-is, so pre-grouping batches keep working. Skips
    ``skip`` (the helper/quarantine dirs).

    ``only_ids`` matches a run dir by its own name (``<id>-<mode>``) *or* by its
    group name (``<id>``), so naming a structure selects every mode run for it.
    Callers apply their own "is this really a run dir?" predicate to the results.
    """
    skip = set(skip)
    only = set(only_ids) if only_ids else None
    for child in sorted(Path(parent_dir).iterdir()):
        if not child.is_dir() or child.name in skip:
            continue

        if is_group_dir(child):
            group_selected = only is not None and child.name in only
            for run_dir in sorted(child.iterdir()):
                if not run_dir.is_dir() or run_dir.name in skip:
                    continue
                if not run_dir.name.startswith(f"{child.name}-"):
                    continue
                if only is not None and not group_selected and run_dir.name not in only:
                    continue
                yield run_dir
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
