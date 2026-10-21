#!/usr/bin/env python3
"""Build one ligand's ``.interp`` spring-model file from two ORCA Hessians.

Database building, not inference: nothing in the batch DAG runs this. Give it
the natural-charge ``.hess`` first and the modified-charge one second -- the
natural-charge geometry becomes the ``.interp`` reference, which is what the
inference stage builds the ligand's bond graph from. The geometry is read from
each ``.hess`` file's own ``$atoms`` block.

    xas-build-interp natural.hess modified.hess -o ZnCys.interp

Writes ``<stem>.natural.springs`` and ``<stem>.modified.springs`` next to the
output (or into ``--springs-dir``); the ``.interp`` header names them.
Point ``xas-run-batch --ligand-dir`` (or ``xas-interp-hessian --ligand-dir``) at
the directory holding the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xas_pipeline.chem import spring_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ligand .interp file from its natural- and modified-charge "
            "ORCA Hessians (natural first)."
        )
    )
    parser.add_argument("natural_hess", type=Path, help="Natural-charge ORCA .hess (model 1)")
    parser.add_argument("modified_hess", type=Path, help="Modified-charge ORCA .hess (model 2)")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output .interp file"
    )
    parser.add_argument(
        "--springs-dir",
        type=Path,
        default=None,
        help="Where to write the two intermediate spring models (default: next to --output).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Keep only pairs within this distance in Angstrom (default: every pair).",
    )
    parser.add_argument(
        "--bonded-only",
        action="store_true",
        help="Keep only covalently bonded pairs.",
    )
    parser.add_argument(
        "--bond-factor",
        type=float,
        default=1.2,
        help=(
            "Pair (i,j) is bonded when d <= FACTOR * (r_i + r_j) (default: 1.2). Used "
            "for --bonded-only and for the bond-graph comparison of the two charge states."
        ),
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the Hessian reconstruction (MAE) report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the two charge states have different bond graphs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for path in (args.natural_hess, args.modified_hess):
        if not path.is_file():
            print(f"ERROR: Hessian not found: {path}", file=sys.stderr)
            return 1

    try:
        result = spring_model.build_interp_from_hess(
            args.natural_hess,
            args.modified_hess,
            args.output,
            springs_dir=args.springs_dir,
            options=spring_model.SpringModelOptions(
                cutoff=args.cutoff,
                bonded_only=args.bonded_only,
                bond_factor=args.bond_factor,
            ),
            check=not args.no_check,
            bond_factor=args.bond_factor,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.strict and (result.bonds_only_natural or result.bonds_only_modified):
        print(
            "ERROR: the two charge states have different bond graphs (see warnings above).",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
