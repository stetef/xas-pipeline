"""Batch directory layout: run ids, run dirs, and scanning a batch for them.

A run id is ``<structure id>-<mode>`` and names both the run dir and its files.
Run dirs are grouped by structure, so several modes of one structure sit side
by side::

    batch-root/
      batch-jobs.log
      2j6a_ZN_cluster1/                        <- group dir (the structure)
        2j6a_ZN_cluster1-caopt-anfreq/         <- run dir = run id
        2j6a_ZN_cluster1-hopt-spring/
      downloading-station/  failed-orca/  failed-corvus/

A run dir is "flat" (everything inside it) or "split" by the postprocess into
``working-<run_id>/`` (the calculation) and ``output-<run_id>/`` (the spectra).
Batches from before mode suffixes have bare ``<id>/`` run dirs directly under the
root; they still scan, and a mode added to such a batch nests inside the old
run dir -- :func:`iter_id_dirs` yields both.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import modes

# Directories under a batch root that never hold runs.
SKIP_DIR_NAMES = frozenset(
    {"failed-orca", "failed-corvus", "downloading-station", "xyz_files", "optimized_xyz_files"}
)

# Every run-dir suffix this code writes.
RUN_DIR_MODES = tuple(modes.MODES)

# Suffixes of older batches, mapped to the mode they meant. Recognized, never
# written: such a dir does not round-trip through run_id_for, which is the
# signal that it predates the rename.
LEGACY_SUFFIX_TO_MODE = {
    "ca-fixed": "caopt-anfreq",
    "H-only": "hopt-anfreq",
    "h-only": "hopt-anfreq",
    "single-point": "carved-anfreq",
    "interp": "carved-spring",
    "interp-hopt": "hopt-spring",
    "opt-interp": "caopt-spring",
    "interp-raw": "asis-spring",
}


def run_id_for(id_name: str, mode: str) -> str:
    """The run id (== run dir name, == artifact basename) for one structure+mode."""
    return f"{id_name}-{mode}"


def mode_from_run_id(run_id: str) -> str | None:
    """The mode a run id encodes (legacy suffixes mapped), or None for a bare ``<id>``.

    Longest suffix wins, so ``-quick-ca-fixed`` is not read as legacy ``-ca-fixed``.
    """
    candidates = list(RUN_DIR_MODES) + list(LEGACY_SUFFIX_TO_MODE)
    for suffix in sorted(candidates, key=len, reverse=True):
        if run_id.endswith(f"-{suffix}"):
            return LEGACY_SUFFIX_TO_MODE.get(suffix, suffix)
    return None


def group_dir_for(batch_root: Path, id_name: str) -> Path:
    """The per-structure group dir holding every mode's run dir."""
    return Path(batch_root) / id_name


def run_dir_for(batch_root: Path, id_name: str, mode: str) -> Path:
    """The run dir for one structure+mode: ``<batch_root>/<id>/<id>-<mode>``."""
    return group_dir_for(batch_root, id_name) / run_id_for(id_name, mode)


def nested_mode_run_dirs(candidate: Path) -> list[Path]:
    """The ``<name>-<mode>`` run dirs nested directly inside ``candidate``.

    Only names ending in a *known* mode suffix count, so an ordinary subdirectory
    of a run dir (a Corvus working tree, an archive snapshot) is never mistaken
    for a run.
    """
    candidate = Path(candidate)
    if not candidate.is_dir():
        return []
    try:
        children = sorted(candidate.iterdir())
    except OSError:
        return []
    return [
        child
        for child in children
        if child.is_dir()
        and child.name.startswith(f"{candidate.name}-")
        and mode_from_run_id(child.name) is not None
    ]


def is_own_run_dir(candidate: Path) -> bool:
    """True when ``candidate`` holds a run itself, rather than only grouping them.

    The rule is deliberately generous: a directory is its own run dir unless it
    is a *pure* group, i.e. it contains nothing but ``<id>-<mode>`` run dirs.
    Anything else -- files, a ``working-``/``output-`` pair, a bare
    ``Corvus3_cfavg_xas/`` tree with no top-level files yet -- counts as run
    content. Callers still apply their own "is this really a run dir?" predicate,
    so being generous here costs nothing, whereas being strict silently drops
    real runs from every scan.
    """
    candidate = Path(candidate)
    if not candidate.is_dir():
        return False
    try:
        children = list(candidate.iterdir())
    except OSError:
        return False

    nested_names = {run_dir.name for run_dir in nested_mode_run_dirs(candidate)}
    if not nested_names:
        return True
    return any(child.name not in nested_names for child in children)


def is_group_dir(candidate: Path) -> bool:
    """True when ``candidate`` only groups per-mode run dirs and is not one itself."""
    return bool(nested_mode_run_dirs(candidate)) and not is_own_run_dir(candidate)


def iter_id_dirs(
    parent_dir: Path,
    *,
    skip: Iterable[str] = SKIP_DIR_NAMES,
    only_ids: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield the per-run directories under ``parent_dir`` (sorted).

    A first-level child contributes itself when it holds a run of its own, and
    additionally any ``<id>-<mode>`` run dirs nested inside it. The two are not
    exclusive, which is what lets a new mode be added to an *existing* batch: a
    pre-grouping ``<id>/`` (already split into ``working-``/``output-``) that
    gains an ``<id>-interp/`` yields both the original run and the new one.

    ``only_ids`` matches a run dir by its own name (``<id>-<mode>``) *or* by its
    group name (``<id>``), so naming a structure selects every mode run for it.
    None selects every run; an empty collection selects none.
    Callers apply their own "is this really a run dir?" predicate to the results.
    """
    skip = set(skip)
    only = set(only_ids) if only_ids is not None else None
    for child in sorted(Path(parent_dir).iterdir()):
        if not child.is_dir() or child.name in skip:
            continue

        selected_by_group = only is None or child.name in only

        if is_own_run_dir(child) and selected_by_group:
            yield child

        for run_dir in nested_mode_run_dirs(child):
            if run_dir.name in skip:
                continue
            if only is not None and not selected_by_group and run_dir.name not in only:
                continue
            yield run_dir


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
    """Move a run dir into ``dest_parent/<name>``, replacing any stale copy.

    A legacy run dir that also hosts newer ``<id>-<mode>`` runs keeps those: only
    its own contents move, so a failed old run never drags healthy new ones
    into quarantine.
    """
    src_dir, dest_parent = Path(src_dir), Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    destination = dest_parent / src_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    nested = set(nested_mode_run_dirs(src_dir))
    if not nested:
        shutil.move(str(src_dir), str(destination))
        return destination
    destination.mkdir()
    for child in sorted(src_dir.iterdir()):
        if child not in nested:
            shutil.move(str(child), str(destination / child.name))
    return destination


# Written by xas-process-feff at the batch root: one CORVUS-failed run id per line.
FAILED_CORVUS_MANIFEST = "corvus-failed-ids.txt"


def read_failed_corvus_ids(batch_root: Path) -> list[str]:
    """The run ids in ``<batch_root>/corvus-failed-ids.txt``, in file order (none if absent)."""
    manifest = Path(batch_root) / FAILED_CORVUS_MANIFEST
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return []
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def write_failed_corvus_ids(batch_root: Path, run_ids) -> Path:
    """Rewrite ``<batch_root>/corvus-failed-ids.txt`` (sorted, one id per line)."""
    manifest = Path(batch_root) / FAILED_CORVUS_MANIFEST
    manifest.write_text("".join(f"{run_id}\n" for run_id in sorted(set(run_ids))), encoding="utf-8")
    return manifest
