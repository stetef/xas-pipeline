"""The optimization modes: one registry every mode table is derived from.

A mode fixes how a structure's geometry is prepared and where its Hessian comes
from. Its name is also the run-dir suffix (``<id>-<mode>``). The geometry x
Hessian family is named on both axes -- ``<geometry>-<hessian>`` with geometry
caopt (CA atoms fixed), hopt (hydrogens only), carved (untouched) or asis (as
handed in), and Hessian anfreq (ORCA analytic) or spring (interpolated from
ligand spring models) -- so e.g. caopt-anfreq vs caopt-spring differ only in the
Hessian. The other modes vary something else (cheaper theory, point charges,
QM/MM partitioning) and keep their own names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str
    #: Command-line flag selecting it, or None (the default mode, or one kept only
    #: so existing run dirs still parse).
    flag: str | None
    #: ORCA input template under ``data/orca-templates/``, or None: no ORCA stage.
    template: str | None
    #: Whether the ORCA input runs ``! AnFreq`` unless overridden.
    anfreq: bool
    #: Hessian interpolated from ligand spring models before CORVUS (else ORCA's).
    spring_hessian: bool
    #: Atoms frozen via [CA_ATOM]: "ca" (CA atoms), "backbone" (COORD=TRUE
    #: backbone groups), "xtb" (COORD=FALSE + backbone groups), or None.
    constraints: str | None = None
    #: xTB QM region via [INDICES]: "non-backbone" or "all" COORD=TRUE atoms.
    qm_region: str | None = None
    #: Needs a ``<xyz stem>.pc`` point-charge file next to each input.
    needs_point_charges: bool = False
    #: One line for --help.
    help: str = ""


DEFAULT_MODE = "caopt-anfreq"

MODES: dict[str, Mode] = {
    m.name: m
    for m in (
        Mode("caopt-anfreq", None, "orca-template-caopt-anfreq.in", True, False, constraints="ca",
             help="CA atoms fixed, everything else optimized, ORCA AnFreq Hessian (default)."),
        Mode("quick", "--quick", "orca-template-quick.in", False, False,
             help="Cheap unconstrained optimization, no frequencies."),
        Mode("quick-ca-fixed", "--quick-ca-fixed", "orca-template-quick-ca-fixed.in", False, False,
             constraints="ca", help="Cheap CA-fixed optimization, no frequencies."),
        Mode("hopt-anfreq", "--H", "orca-template-hopt-anfreq.in", True, False,
             help="Optimize hydrogens only, ORCA AnFreq Hessian."),
        Mode("carved-anfreq", "--single", "orca-template-carved-anfreq.in", True, False,
             help="No optimization (carved geometry), ORCA AnFreq Hessian."),
        Mode("no-constraints", "--free", "orca-template-no-constraints.in", True, False,
             help="Unconstrained optimization, ORCA AnFreq Hessian."),
        Mode("backbone", "--backbone", "orca-template-backbone-charges.in", True, False,
             constraints="backbone", needs_point_charges=True,
             help="Backbone point charges (needs <stem>.pc per XYZ), ORCA AnFreq Hessian."),
        Mode("xtb-free", "--xtb-free", "orca-template-xtb-free.in", True, False,
             qm_region="non-backbone",
             help="QM/xTB: COORD=TRUE non-backbone atoms in QM, ORCA AnFreq Hessian."),
        Mode("xtb-constrained", "--xtb-constrained", "orca-template-xtb-constrained.in", False, False,
             constraints="xtb", qm_region="all",
             help="QM/xTB: all COORD=TRUE atoms in QM, backbone frozen, no frequencies."),
        Mode("carved-spring", None, "orca-template-carved-spring.in", False, True,
             help="Single point, spring Hessian (superseded by hopt-spring)."),
        Mode("hopt-spring", "--interp", "orca-template-hopt-spring.in", False, True,
             help="Optimize hydrogens only (relaxes the H contacts that break FEFF's "
                  "muffin tins), spring Hessian."),
        Mode("asis-spring", "--interp-raw", None, False, True,
             help="No ORCA: the geometry as handed in, spring Hessian. Cheapest; nothing "
                  "checks the geometry before FEFF."),
        Mode("caopt-spring", "--caopt-spring", None, False, True,
             help="No ORCA, for geometries already CA-fixed-optimized (e.g. a caopt-anfreq "
                  "result): spring Hessian. Only sets the label; the claim is not checked."),
    )
}

#: Modes with an ORCA stage (an input template).
ORCA_MODES = tuple(name for name, m in MODES.items() if m.template)
#: Modes with no ORCA stage: the CORVUS job is submitted with nothing to wait on.
NO_ORCA_MODES = frozenset(name for name, m in MODES.items() if not m.template)
#: Modes whose Hessian is interpolated from ligand spring models.
SPRING_HESSIAN_MODES = frozenset(name for name, m in MODES.items() if m.spring_hessian)


def get(name: str) -> Mode:
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(f"Unknown mode {name!r}; known: {', '.join(MODES)}") from None


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive mode flags (and ``--mode NAME``); result in ``args.mode``."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--mode", choices=list(MODES), default=DEFAULT_MODE,
        help=f"Optimization mode by name (default: {DEFAULT_MODE}). The flags below are shorthands.",
    )
    for mode in MODES.values():
        if mode.flag:
            group.add_argument(mode.flag, dest="mode", action="store_const", const=mode.name,
                               help=f"{mode.name}: {mode.help}")
