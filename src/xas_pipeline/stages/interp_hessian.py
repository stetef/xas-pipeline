#!/usr/bin/env python3
"""Build ``<run_id>.hess`` by interpolating ligand spring models (``--interp`` mode).

This stage stands in for ORCA's analytic-frequency (AnFreq) step. The ``interp``
ORCA template runs a single point for the energy only -- no ``! AnFreq``, so ORCA
writes no ``.hess`` -- and this stage produces the Hessian instead, from
pre-built per-ligand spring models:

1. Resolve the run's geometry exactly as :mod:`~xas_pipeline.stages.corvus_prep`
   does (:func:`xas_pipeline.chem.xyz.select_run_xyz`), so the Hessian and the
   ``.dym`` describe the same atoms in the same order.
2. Locate every occurrence of each packaged ligand (``data/interp-ligands/*.interp``)
   in that geometry by subgraph isomorphism and interpolate its spring constants
   to the observed bond lengths -> ``spring.model``.
3. Build the 3N x 3N Hessian from those constants and the geometry's bond
   vectors, and write it in ORCA ``.hess`` format -> ``<run_id>.hess``.

The output is the same filename the CORVUS stage already expects, and is parsed
by the same reader as a real ORCA Hessian, so nothing downstream needs to know
which route produced it.

Run by the generated corvus wrapper immediately before prepare-corvus; also
usable by hand:

    python -m xas_pipeline.stages.interp_hessian <run_dir> --run-id <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xas_pipeline import resources
from xas_pipeline.chem import spring_hessian, springs
from xas_pipeline.chem import xyz as _chem_xyz

# Written next to the Hessian for provenance: the merged, interpolated spring
# constants the Hessian was built from. Kept (not a temp file) because it is the
# only record of what the subgraph search actually matched.
SPRING_MODEL_NAME = "spring.model"


def _resolve_dir(path_str: str) -> Path:
    run_dir = Path(path_str).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")
    return run_dir


def build_interp_hessian(
    run_dir: Path,
    run_id: str,
    *,
    ligand_files: list[Path] | None = None,
    bond_factor: float = 1.2,
    bond_cutoff: float | None = None,
    no_extrapolate: bool = False,
    zero_negative: bool = False,
    freq_check: bool = True,
    skip_modes: int = 6,
) -> spring_hessian.HessianResult:
    """Interpolate a Hessian for one run directory; returns the build result."""
    ligand_files = list(ligand_files) if ligand_files else resources.interp_ligand_files()
    if not ligand_files:
        raise FileNotFoundError(
            "No .interp ligand files found. Expected packaged files under "
            f"{resources.interp_ligand_root()}, or pass --ligand explicitly."
        )

    source_xyz = _chem_xyz.select_run_xyz(run_dir, run_id)
    print(f"Selected source XYZ: {source_xyz.name}")

    spring_model_path = run_dir / SPRING_MODEL_NAME
    springs.interpolate_to_spring_model(
        source_xyz,
        ligand_files,
        spring_model_path,
        options=springs.InterpOptions(
            bond_factor=bond_factor,
            bond_cutoff=bond_cutoff,
            no_extrapolate=no_extrapolate,
        ),
    )
    print(f"Wrote interpolated spring model: {spring_model_path.name}")

    hess_path = run_dir / f"{run_id}.hess"
    result = spring_hessian.build_hess_from_spring_model(
        source_xyz,
        spring_model_path,
        hess_path,
        zero_negative=zero_negative,
        freq_check=freq_check,
        skip_modes=skip_modes,
    )

    if result.n_imaginary:
        # Not fatal: FEFF's Debye-Waller step still runs, but the DW factors from
        # an unstable Hessian are suspect, so make it findable in the job log.
        print(
            f"WARNING: interpolated Hessian for {run_id} has {result.n_imaginary} "
            "imaginary mode(s) beyond the 6 trans/rot modes; the Debye-Waller "
            "factors derived from it are suspect."
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate ligand spring models onto a run's geometry and write "
            "<run_id>.hess, replacing ORCA's AnFreq step for --interp runs."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory; its name is used as the run ID by default.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Explicit run ID (the basename used for <run_id>.hess and <run_id>.xyz). "
            "Defaults to the run directory's name."
        ),
    )
    parser.add_argument(
        "--ligand",
        action="append",
        dest="ligands",
        default=None,
        metavar="FILE",
        help=(
            "A .interp ligand constants file; repeat for several. Defaults to every "
            "file packaged under data/interp-ligands/."
        ),
    )
    parser.add_argument(
        "--bond-factor",
        type=float,
        default=1.2,
        help="Pair (i,j) is bonded when d <= FACTOR * (r_i + r_j) (default: 1.2).",
    )
    parser.add_argument(
        "--bond-cutoff",
        type=float,
        default=None,
        help="Fixed bond cutoff in Angstrom; overrides --bond-factor when set.",
    )
    parser.add_argument(
        "--no-extrapolate",
        action="store_true",
        help="Clamp the interpolation parameter to [0, 1] instead of extrapolating.",
    )
    parser.add_argument(
        "--zero-negative",
        action="store_true",
        help="Set negative interpolated spring constants to zero before building H.",
    )
    parser.add_argument(
        "--no-freq-check",
        action="store_true",
        help="Skip diagonalization and the imaginary-mode report (large systems).",
    )
    parser.add_argument(
        "--skip-modes",
        type=int,
        default=6,
        help="Lowest-|freq| modes excluded from the imaginary-mode warning (default: 6).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_dir = _resolve_dir(args.run_dir)
    run_id = args.run_id if args.run_id else run_dir.name
    ligand_files = [Path(p).expanduser() for p in args.ligands] if args.ligands else None

    if ligand_files:
        missing = [str(p) for p in ligand_files if not p.is_file()]
        if missing:
            print(f"ERROR: ligand file(s) not found: {', '.join(missing)}", file=sys.stderr)
            return 1

    try:
        build_interp_hessian(
            run_dir,
            run_id,
            ligand_files=ligand_files,
            bond_factor=args.bond_factor,
            bond_cutoff=args.bond_cutoff,
            no_extrapolate=args.no_extrapolate,
            zero_negative=args.zero_negative,
            freq_check=not args.no_freq_check,
            skip_modes=args.skip_modes,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not build the interpolated Hessian for {run_id}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
