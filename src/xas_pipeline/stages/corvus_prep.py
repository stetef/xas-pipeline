"""Prepare a finished ORCA (or spring-Hessian) run dir for CORVUS.

Writes, in the run dir:

- ``corvus-begin-<run_id>.xyz``: the geometry of record, cleaned and with
  canonical element symbols (checked against the last ORCA trajectory frame);
- ``<run_id>.dym``: the dynamical matrix (``.hess`` + masses + Bohr coordinates)
  and ``corvus-<run_id>.dym``, the same centered on the first Zn (the absorber),
  which FEFF's DMDW step needs because it matches atoms by coordinate;
- ``corvus-<run_id>-xas.in`` (+ the ``xanes.in``/``exafs.in`` cards it reads) and
  ``corvus-job-xas.script``, which the CORVUS wrapper runs inline.

Centering uses FEFF's ``dym2feffinp`` ($DYM2FEFFINP_BIN, /opt/feff10, or PATH),
else Corvus's Python DYM writer.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xas_pipeline import config, jobscripts, resources, templates
from xas_pipeline.chem import periodic
from xas_pipeline.chem import xyz as chem_xyz
from xas_pipeline.chem.hessian import read_orca_hessian
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import Scheduler, add_scheduler_argument, get_scheduler

log = logging.getLogger(__name__)

CORVUS_TEMPLATE = "corvus-template-xas.in"
# Sub-input cards the xas template reads by relative name (xanes_input{xanes.in}).
CORVUS_SUBINPUTS = ("xanes.in", "exafs.in")
DYM2FEFFINP_FALLBACK_PATHS = ["/opt/feff10/bin/MPI/dym2feffinp"]


def write_clean_corvus_xyz(source_xyz: Path, dest_xyz: Path) -> None:
    """Copy an XYZ with comments stripped and element symbols canonicalized."""
    lines = source_xyz.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Input does not look like XYZ (missing header lines): {source_xyz}")
    try:
        natoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid XYZ atom count header in {source_xyz}: {lines[0]!r}") from exc

    cleaned = [lines[0].strip(), lines[1].strip()]
    for raw in lines[2:]:
        if len(cleaned) - 2 >= natoms:
            break
        if not raw.strip():
            continue
        tokens = raw.split("#", 1)[0].split()
        if len(tokens) < 4:
            raise ValueError(f"Invalid atom line in {source_xyz}: {raw!r}")
        symbol = periodic.canonical_symbol_from_token(tokens[0])
        cleaned.append("{:<2} {:>12} {:>12} {:>12}".format(symbol, *tokens[1:4]))
    if len(cleaned) - 2 != natoms:
        raise ValueError(
            f"XYZ atom count mismatch in {source_xyz}: header={natoms}, parsed={len(cleaned) - 2}"
        )
    dest_xyz.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def validate_against_trajectory(run_dir: Path, corvus_xyz: Path, tolerance_angstrom: float = 1e-4) -> None:
    """Raise if the newest ``*_trj.xyz`` last frame disagrees with *corvus_xyz*.

    Guards against handing CORVUS a geometry other than the one ORCA's Hessian
    was computed for. No trajectory (no ORCA optimization) means nothing to check.
    """
    trajectories = [p for p in run_dir.glob("*_trj.xyz") if p.is_file()]
    if not trajectories:
        return
    latest = max(trajectories, key=lambda p: (p.stat().st_mtime, p.name))
    trj_z, trj_coords = chem_xyz.read_last_xyz_frame(latest)
    xyz_z, xyz_coords = chem_xyz.read_last_xyz_frame(corvus_xyz)
    if len(trj_z) != len(xyz_z):
        raise ValueError(
            f"Trajectory/Corvus XYZ atom-count mismatch: {latest.name} has {len(trj_z)}, "
            f"{corvus_xyz.name} has {len(xyz_z)}"
        )
    if not np.array_equal(trj_z, xyz_z):
        raise ValueError(f"Trajectory/Corvus XYZ element-order mismatch between {latest.name} and {corvus_xyz.name}")
    max_diff = float(np.max(np.abs(trj_coords - xyz_coords)))
    if max_diff > tolerance_angstrom:
        raise ValueError(
            f"Trajectory/Corvus XYZ coordinate mismatch: max|delta|={max_diff:.6e} A exceeds "
            f"tolerance {tolerance_angstrom:.1e} A (trj={latest.name}, corvus={corvus_xyz.name})"
        )
    log.info("Validated %s (last frame) against %s within %.1e A", latest.name, corvus_xyz.name, tolerance_angstrom)


def write_dym(path: Path, atomic_numbers, masses_amu, coords_bohr, hess: np.ndarray) -> None:
    """Write a type-1 FEFF dynamical-matrix file.

    Layout: ``1``; N; N atomic numbers; N masses (amu); N ``x y z`` (Bohr); then
    for every atom pair (i, j), 1-based, the line ``i j`` and the 3x3 Hessian
    block (Hartree/Bohr^2), one row per line.
    """
    natoms = len(atomic_numbers)
    if hess.shape != (3 * natoms, 3 * natoms):
        raise ValueError(f"Hessian shape {hess.shape} does not match 3N x 3N with N={natoms}")
    lines = ["1", str(natoms)]
    lines += [str(int(z)) for z in atomic_numbers]
    lines += [f"{float(m):.10f}" for m in masses_amu]
    lines += [f"{x:.10f} {y:.10f} {z:.10f}" for x, y, z in coords_bohr]
    for i in range(natoms):
        for j in range(natoms):
            lines.append(f"{i + 1} {j + 1}")
            for row in range(3):
                lines.append(" ".join(f"{h:12.6e}" for h in hess[3 * i + row, 3 * j:3 * j + 3]))
    Path(path).write_text("\n".join(lines) + "\n")


def find_executable(name: str, env_var: str, fallback_paths: list[str]) -> str:
    """$env_var (must be executable if set), else the first fallback path, else PATH."""
    override = os.environ.get(env_var, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise FileNotFoundError(f"{env_var} is set but not executable: {candidate}")
    for raw in fallback_paths:
        candidate = Path(raw).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find '{name}'. Set {env_var} to its full path. "
        f"Searched: {', '.join([*fallback_paths, f'$PATH via {name!r}'])}"
    )


# dym2feffinp keeps filenames in a fixed Fortran buffer: past this length it
# silently writes to a truncated name and still exits 0 (FEFF10 10.0.0: 49
# characters round-trip, 51 do not). Run ids with mode suffixes get there, so
# the conversion runs on these short scratch names and is renamed into place.
DYM2FEFFINP_MAX_FILENAME = 50
_DYM_SCRATCH_IN = "dym2feffinp-in.dym"
_DYM_SCRATCH_OUT = "dym2feffinp-out.dym"


def _run_dym2feffinp(dym2feffinp_bin: str, run_dir: Path, dym_path: Path, feff_dym_path: Path,
                     center_index_1based: int) -> None:
    """Center ``dym_path`` on the absorber into ``feff_dym_path`` with dym2feffinp."""
    scratch_in, scratch_out = run_dir / _DYM_SCRATCH_IN, run_dir / _DYM_SCRATCH_OUT
    scratch_out.unlink(missing_ok=True)
    shutil.copyfile(dym_path, scratch_in)
    try:
        subprocess.run(
            [dym2feffinp_bin, "--c", str(center_index_1based), "--d", scratch_out.name, scratch_in.name],
            check=True, cwd=run_dir,
        )
        # Its exit code says nothing about whether it wrote the file.
        if not scratch_out.is_file():
            raise RuntimeError(
                f"dym2feffinp exited 0 but produced no centered DYM for {dym_path.name}; check "
                f"that it is a valid dynamical matrix and that the absorber index "
                f"({center_index_1based}) is in range."
            )
        scratch_out.replace(feff_dym_path)
        log.info("Wrote centered DYM: %s", feff_dym_path.name)
    finally:
        scratch_in.unlink(missing_ok=True)
        scratch_out.unlink(missing_ok=True)


def _write_centered_dym_with_corvus(run_dir, run_id, feff_dym_path, atomic_numbers, masses_amu,
                                    coords_bohr, hess, center_index_1based) -> None:
    """Center the DYM with Corvus's Python writer (when dym2feffinp is unavailable)."""
    try:
        from corvus.dmdw import writeDym
    except Exception as exc:
        raise FileNotFoundError(
            "Found neither dym2feffinp nor the Python fallback corvus.dmdw.writeDym. Install "
            "Corvus in this environment or set DYM2FEFFINP_BIN."
        ) from exc
    natoms = len(atomic_numbers)
    center = coords_bohr.tolist()[center_index_1based - 1]
    # writeDym never shifts a type-1 DYM, so the coordinates are centered here
    # (identical to dym2feffinp's output); atoms print nearest-first.
    coords = [[a - b for a, b in zip(coord, center, strict=True)] for coord in coords_bohr.tolist()]
    order = sorted(range(natoms), key=lambda i: float(np.sqrt(np.sum(np.square(coords[i])))))
    writeDym({
        "dymType": 1,
        "nAt": natoms,
        "atNums": [int(z) for z in atomic_numbers.tolist()],
        "atMasses": [float(m) for m in masses_amu.tolist()],
        "atCoords": coords,
        "dm": [[hess[3 * i:3 * i + 3, 3 * j:3 * j + 3].tolist() for j in range(natoms)] for i in range(natoms)],
        "printOrder": order,
    }, str(feff_dym_path))
    (run_dir / f"corvus-{run_id}.feff.inp").write_text(
        "# Generated by prepare-corvus.py Python fallback to document the centered DYM conversion.\n",
        encoding="utf-8",
    )
    log.info("Wrote centered DYM with corvus.dmdw.writeDym: %s", feff_dym_path.name)


@dataclass
class CorvusPrep:
    xyz: Path
    dym: Path
    centered_dym: Path
    corvus_input: Path
    job_script: Path


def prepare_corvus(run_dir: Path, run_id: str | None = None, *, sched: Scheduler, num_procs: int = 16) -> CorvusPrep:
    """Write a run dir's CORVUS inputs (see the module docstring)."""
    run_dir = Path(run_dir)
    run_id = run_id or run_dir.name
    hess_path = run_dir / f"{run_id}.hess"
    if not hess_path.exists():
        raise FileNotFoundError(f"Missing Hessian file: {hess_path}")

    source_xyz = chem_xyz.select_run_xyz(run_dir, run_id)
    xyz_path = run_dir / f"corvus-begin-{run_id}.xyz"
    write_clean_corvus_xyz(source_xyz, xyz_path)
    log.info("Selected source XYZ: %s; wrote %s", source_xyz.name, xyz_path.name)
    validate_against_trajectory(run_dir, xyz_path)

    hess, natoms_hess = read_orca_hessian(hess_path)
    atomic_numbers, masses_amu, coords_bohr = chem_xyz.read_xyz(xyz_path)
    if natoms_hess != len(atomic_numbers):
        raise ValueError(f"Atom count mismatch: Hessian implies {natoms_hess} atoms, XYZ has {len(atomic_numbers)}")
    dym_path = run_dir / f"{run_id}.dym"
    write_dym(dym_path, atomic_numbers, masses_amu, coords_bohr, hess)

    zn = np.where(atomic_numbers == periodic.ATOMIC_SYMBOLS["ZN"])[0]
    if zn.size == 0:
        raise ValueError("No Zn atoms found in XYZ; cannot center DYM on absorber.")
    absorber = int(zn[0]) + 1  # the first Zn
    feff_dym_path = run_dir / f"corvus-{run_id}.dym"
    try:
        dym2feffinp = find_executable("dym2feffinp", "DYM2FEFFINP_BIN", DYM2FEFFINP_FALLBACK_PATHS)
    except FileNotFoundError as exc:
        # A configured but unusable path is a site mistake worth seeing in the job log.
        report = log.warning if os.environ.get("DYM2FEFFINP_BIN", "").strip() else log.info
        report("%s; using the Corvus Python DYM writer instead.", exc)
        _write_centered_dym_with_corvus(run_dir, run_id, feff_dym_path, atomic_numbers, masses_amu,
                                        coords_bohr, hess, absorber)
    else:
        log.info("Using dym2feffinp: %s", dym2feffinp)
        _run_dym2feffinp(dym2feffinp, run_dir, dym_path, feff_dym_path, absorber)

    template_root = resources.template_root()
    for name in CORVUS_SUBINPUTS:
        shutil.copy2(template_root / name, run_dir / name)
    corvus_input = templates.render(template_root / CORVUS_TEMPLATE, run_dir / f"corvus-{run_id}-xas.in", {
        "DIRECTORY": f"{run_dir}/",
        "ID": run_id,
        "PROCS": num_procs,
        "XYZ_FILE": xyz_path.name,
    })
    job_script = jobscripts.corvus_job(sched, run_dir, run_id)
    log.info("CORVUS input %s, job %s", corvus_input.name, job_script.name)
    return CorvusPrep(xyz_path, dym_path, feff_dym_path, corvus_input, job_script)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a run dir for CORVUS (.dym, inputs, job script).")
    parser.add_argument("run_dir", type=Path, help="The run dir; its name is the run id unless --run-id is given")
    parser.add_argument("--run-id", default=None,
                        help="Basename of <run_id>.hess etc., when the dir is named otherwise (e.g. working-<id>)")
    add_scheduler_argument(parser)
    parser.add_argument("--num-procs", type=int, default=16, help="[PROCS] in the CORVUS input (default: 16)")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    # Site config (e.g. DYM2FEFFINP_BIN) for runs outside the wrapper; values
    # already in the environment win.
    config.load_env(resources.pipeline_env())
    args = build_parser().parse_args()
    setup_cli_logging()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")
    prepare_corvus(run_dir, args.run_id, sched=get_scheduler(args.scheduler), num_procs=args.num_procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
