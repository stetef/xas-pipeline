#!/usr/bin/env python3
"""Reclaim disk space from finished ORCA + Corvus/FEFF calculation directories.

This is a *deny-list* cleaner: it deletes only explicitly enumerated files that
are either regenerable intermediates or superseded rerun duplicates. Anything not
on the list is kept, so an unexpected/new file type is never silently removed.

What it deletes, per cluster ``<id>/`` directory:

  ORCA scratch (in ``working-<id>/`` root):
    *.densities, *.densitiesinfo   -> density-plot data, regenerable from the .gbw
    *.cpcm, *.cpcm_corr            -> CPCM solvation restart, regenerable
    *.engrad                       -> converged gradient, intermediate

  FEFF scratch (in each live ``Corvus3_cfavg_*/*_FEFF/``):
    dmdw.out                       -> 40+ MB DMDW dump, regenerable from .dym + dmdw.inp
    *.bin                          -> FEFF module restart/scratch binaries
    gg.dat                         -> FMS Green's-function text dump

  Rerun duplicates (anything with ``.rerun-`` in its name, file or dir, at the
  cluster or working-dir level) -> superseded snapshots from earlier reruns.

What it always KEEPS (never a delete target): .log, _trj.xyz, .opt, .gbw, .hess,
.dym, .in / _clean.xyz / _comments.txt, .property.txt, .timing, .bibtex, and in
FEFF dirs xmu.dat / chi.dat / chi_R.dat / *.inp / dmdw.inp / list.dat / paths.dat
/ *.png / geom.dat / atoms.dat / stdout / stderr.

Safety model:
  * DRY RUN IS THE DEFAULT. Nothing is deleted unless you pass --execute.
  * A cluster with no ``output-*`` directory (results not yet captured) is skipped
    unless you pass --force.
  * Batch-level files (e.g. rerun-state-*.json) are left untouched.

Usage:
    # preview a whole batch (no deletion):
    python cleanup-calc-artifacts.py /path/to/test-VIII
    # actually delete:
    python cleanup-calc-artifacts.py /path/to/test-VIII --execute
    # a single cluster:
    python cleanup-calc-artifacts.py /path/to/test-VIII/2j6a_ZN_homo_d2.60_cluster1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # run-from-checkout bootstrap
from xas_pipeline import layout


# Regenerable ORCA intermediates, matched at the top level of working-<id>/.
ORCA_SCRATCH_GLOBS = (
    "*.densities",
    "*.densitiesinfo",
    "*.cpcm",
    "*.cpcm_corr",
    "*.engrad",
)

# Regenerable FEFF intermediates, matched at the top level of each live FEFF dir.
FEFF_SCRATCH_GLOBS = (
    "dmdw.out",
    "*.bin",
    "gg.dat",
)

# Substring that marks a superseded rerun snapshot (file or directory).
RERUN_MARKER = ".rerun-"

# Directories under a batch root that are not cluster/run directories.
SKIP_DIR_NAMES = layout.SKIP_DIR_NAMES


def human(nbytes: int) -> str:
    """Format a byte count as a short human-readable string."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}TB"


def path_size(path: Path) -> int:
    """Total size in bytes of a file, or of every file under a directory."""
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            try:
                total += child.lstat().st_size
            except OSError:
                pass
    return total


class Cleaner:
    def __init__(self, execute: bool, force: bool,
                 do_orca: bool, do_feff: bool, do_reruns: bool):
        self.execute = execute
        self.force = force
        self.do_orca = do_orca
        self.do_feff = do_feff
        self.do_reruns = do_reruns
        self.total_bytes = 0
        self.total_items = 0

    # -- discovery -----------------------------------------------------------

    @staticmethod
    def is_cluster_dir(path: Path) -> bool:
        """A cluster dir has a working-* and/or output-* child directory."""
        if not path.is_dir():
            return False
        for child in path.iterdir():
            if child.is_dir() and (
                child.name.startswith("working") or child.name.startswith("output")
            ):
                return True
        return False

    def find_clusters(self, target: Path) -> list[Path]:
        if self.is_cluster_dir(target):
            return [target]
        clusters = []
        for child in sorted(target.iterdir()):
            if not child.is_dir() or child.name in SKIP_DIR_NAMES:
                continue
            if self.is_cluster_dir(child):
                clusters.append(child)
        return clusters

    @staticmethod
    def working_dirs(cluster: Path) -> list[Path]:
        return [c for c in sorted(cluster.glob("working*")) if c.is_dir()]

    @staticmethod
    def live_feff_dirs(working: Path) -> list[Path]:
        """FEFF dirs to prune, excluding any under a .rerun- snapshot."""
        feff_dirs = []
        for mode_dir in sorted(working.glob("Corvus3_cfavg_*")):
            if not mode_dir.is_dir() or RERUN_MARKER in mode_dir.name:
                continue
            for feff in sorted(mode_dir.glob("*FEFF")):
                if feff.is_dir():
                    feff_dirs.append(feff)
        return feff_dirs

    # -- deletion ------------------------------------------------------------

    def _remove(self, path: Path, reason: str) -> None:
        size = path_size(path)
        kind = "dir " if path.is_dir() and not path.is_symlink() else "file"
        action = "DELETE " if self.execute else "would delete"
        print(f"    [{reason:<12}] {action} {kind} {human(size):>8}  {path}")
        self.total_bytes += size
        self.total_items += 1
        if not self.execute:
            return
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            print(f"      !! failed to delete {path}: {exc}")

    def _glob_delete(self, root: Path, globs, reason: str) -> None:
        for pattern in globs:
            for match in sorted(root.glob(pattern)):
                # Never touch rerun snapshots here; they are handled wholesale.
                if RERUN_MARKER in match.name:
                    continue
                self._remove(match, reason)

    def clean_reruns(self, scope: Path) -> None:
        """Delete top-level files/dirs whose name marks a rerun snapshot."""
        for entry in sorted(scope.iterdir()):
            if RERUN_MARKER in entry.name:
                self._remove(entry, "rerun")

    def clean_cluster(self, cluster: Path) -> None:
        print(f"\n=== {cluster.name} ===")

        has_output = any(
            c.is_dir() and c.name.startswith("output") for c in cluster.iterdir()
        )
        if not has_output and not self.force:
            print("    SKIP: no output-* directory (results not captured); "
                  "use --force to clean anyway")
            return

        # Rerun snapshots at the cluster level (e.g. xanes-archive.rerun-*/).
        if self.do_reruns:
            self.clean_reruns(cluster)

        for working in self.working_dirs(cluster):
            # Rerun snapshots inside the working dir (dirs and files).
            if self.do_reruns:
                self.clean_reruns(working)
            # ORCA scratch at the working-dir root.
            if self.do_orca:
                self._glob_delete(working, ORCA_SCRATCH_GLOBS, "orca-scratch")
            # FEFF scratch in each live FEFF dir.
            if self.do_feff:
                for feff in self.live_feff_dirs(working):
                    self._glob_delete(feff, FEFF_SCRATCH_GLOBS, "feff-scratch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reclaim disk from finished ORCA/Corvus calc dirs by deleting "
            "regenerable intermediates and rerun duplicates (deny-list). "
            "Dry-run by default; pass --execute to actually delete."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="A batch root (scanned for cluster dirs) or a single cluster dir.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete. Without this flag the script only previews (dry run).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clean clusters even if they have no output-* directory.",
    )
    parser.add_argument("--keep-orca-scratch", action="store_true",
                        help="Do not delete ORCA scratch (.densities/.cpcm/.engrad).")
    parser.add_argument("--keep-feff-scratch", action="store_true",
                        help="Do not delete FEFF scratch (dmdw.out/*.bin/gg.dat).")
    parser.add_argument("--keep-reruns", action="store_true",
                        help="Do not delete .rerun- snapshots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target.expanduser().resolve()
    if not target.is_dir():
        print(f"error: not a directory: {target}")
        return 2

    cleaner = Cleaner(
        execute=args.execute,
        force=args.force,
        do_orca=not args.keep_orca_scratch,
        do_feff=not args.keep_feff_scratch,
        do_reruns=not args.keep_reruns,
    )

    clusters = cleaner.find_clusters(target)
    if not clusters:
        print(f"No cluster directories found under {target} "
              "(expected working-*/output-* subdirs).")
        return 0

    banner = "EXECUTING (files will be deleted)" if args.execute else "DRY RUN (no changes)"
    print(f"{banner}  --  {len(clusters)} cluster dir(s) under {target}")

    for cluster in clusters:
        cleaner.clean_cluster(cluster)

    verb = "Deleted" if args.execute else "Would delete"
    print(f"\n{verb} {cleaner.total_items} item(s), "
          f"reclaiming {human(cleaner.total_bytes)}.")
    if not args.execute and cleaner.total_items:
        print("Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
