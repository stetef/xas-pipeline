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

The two are not exclusive, which is what lets a mode be added to a batch that
already ran. Pointing ``--interp`` at an existing batch gives::

    first-set/
      2j6a_ZN_cluster1/                 <- the original run, still a run dir
        working-2j6a_ZN_cluster1/
        output-2j6a_ZN_cluster1/
        2j6a_ZN_cluster1-interp/        <- the newly added mode run

and :func:`iter_id_dirs` yields both, so the postprocess stages see the original
results and the new mode side by side.
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

# The ORCA optimization modes: every mode that has an ORCA input template. Naming
# lives here because it is what makes "<id>-<mode>" parseable back into a mode
# (see mode_from_run_id); the mode -> template mapping stays in
# stages.orca_prep.TEMPLATE_FILE_BY_MODE, and a test asserts the two agree.
#
# Four of these form the geometry x Hessian comparison family, and are named on
# those two axes: "<geometry>-<hessian>". The geometry token says what was allowed
# to move (caopt = CA-fixed optimization, hopt = hydrogens only, carved =
# untouched, asis = whatever was handed in); the Hessian token says where the
# force constants came from (anfreq = ORCA analytic, spring = interpolated from
# ligand spring models). So caopt-anfreq vs caopt-spring differs only in the
# Hessian, and hopt-spring vs caopt-spring only in the geometry -- the comparison
# structure is readable off the names.
#
# The remaining modes vary something else entirely (cheaper theory, point charges,
# QM/MM partitioning) and are left under their own names rather than forced into a
# two-axis scheme that does not describe them.
KNOWN_MODES = (
    "caopt-anfreq",
    "quick",
    "quick-ca-fixed",
    "hopt-anfreq",
    "carved-anfreq",
    "no-constraints",
    "backbone",
    "xtb-free",
    "xtb-constrained",
    "carved-spring",
    "hopt-spring",
)

# Run-dir suffixes that are NOT ORCA modes: no ORCA stage runs for them and there
# is no input template. They are still part of the suffix vocabulary, because
# their run dirs must parse back into a mode like any other.
#
# caopt-spring is "CA-fixed-optimized geometry + interpolated Hessian, CORVUS
# only": its run dirs hold spring.model and <id>.hess but no ORCA log at all. Under its old
# name (opt-interp) it was absent from the suffix vocabulary entirely, which made
# mode_from_run_id() fall through to the "-interp" suffix it also ends with and
# report every one as plain interp -- silently mislabelling 302 run dirs in the
# clustering-validation tree alone. Keeping these out of KNOWN_MODES preserves
# that tuple's meaning (modes with templates) and the test asserting it matches
# TEMPLATE_FILE_BY_MODE. The two-axis naming also removes the trap that caused
# that bug: no name here is a suffix of another.
#
# asis-spring is the same idea with no claim about the geometry: whatever is handed
# to it goes straight to the spring Hessian and FEFF, with no ORCA stage at all.
# Pointed at a carved cluster it reproduces what carved-spring computed (a single
# point moves no atoms) for none of the cost; pointed at a CA-fixed-optimized
# geometry it *is* caopt-spring, reached through xas-run-batch rather than by
# hand.
SUFFIX_ONLY_MODES = ("caopt-spring", "asis-spring")

# The full run-dir suffix vocabulary: what mode_from_run_id can recognize.
RUN_DIR_MODES = KNOWN_MODES + SUFFIX_ONLY_MODES

# Modes with no ORCA stage: the orchestrator submits their CORVUS job with no
# ORCA job to depend on, and prepare-orca scaffolds the run dir without writing
# an ORCA input or job script.
#
# This is SUFFIX_ONLY_MODES by construction rather than a second list to keep in
# sync: a mode is in that tuple precisely because it has no entry in
# TEMPLATE_FILE_BY_MODE, and a mode with no template cannot run ORCA. Named
# separately because callers care about the behaviour, not about why the suffix
# needed registering.
NO_ORCA_MODES = frozenset(SUFFIX_ONLY_MODES)

# Run-dir suffixes this code no longer *emits* but must still recognize, mapped to
# the canonical mode they meant. Two sources:
#
#   - the pre-2026-08 names, before modes were named on the geometry x Hessian
#     axes. Batches on disk (and any tree not migrated) still carry them.
#   - "H-only", the one mode that suffixed its run dirs before the others did, so
#     its dirs use that casing.
#
# Recognized, never produced: run_id_for() always emits the canonical name. A
# legacy dir therefore does not round-trip (mode_from_run_id -> run_id_for gives
# the new spelling), which is deliberate -- the mismatch is the signal that the dir
# predates the rename and should be migrated, not silently re-derived.
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


def mode_suffix(mode: str) -> str:
    """The run-dir/run-id suffix for an ORCA optimization mode.

    Identity now: the canonical mode name *is* the suffix. Kept as a function
    because callers spell the relationship through it, and because the legacy
    casing used to live here.
    """
    return mode


def run_id_for(id_name: str, mode: str) -> str:
    """The run id (== run dir name, == artifact basename) for one structure+mode."""
    return f"{id_name}-{mode_suffix(mode)}"


def mode_from_run_id(run_id: str) -> str | None:
    """Recover the ORCA mode from a run id, or ``None`` if it encodes none.

    ``None`` means a run dir predating mode suffixes (a bare ``<id>``), where the
    mode is genuinely unknown -- callers should fall back to whatever is safe for
    an already-computed run rather than guessing.

    Longest suffix wins, so ``<id>-quick-ca-fixed`` resolves to ``quick-ca-fixed``
    rather than the ``ca-fixed`` it also ends with. Under the geometry x Hessian
    names no canonical suffix is a suffix of another, so that hazard now only
    applies to the legacy spellings below.

    Matches against RUN_DIR_MODES, not KNOWN_MODES: some suffixes name a run that
    skips ORCA entirely and so has no template (see SUFFIX_ONLY_MODES). Legacy
    spellings resolve to the canonical mode they meant, so an un-migrated dir is
    reported as what it *is* rather than as unknown.
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
    Callers apply their own "is this really a run dir?" predicate to the results.
    """
    skip = set(skip)
    only = set(only_ids) if only_ids else None
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
    """Move ``src_dir`` into ``dest_parent/<name>``, replacing any stale copy."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    destination = dest_parent / Path(src_dir).name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(src_dir), str(destination))
    return destination
