#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from xas_pipeline import config, scheduler as _sched, templates, resources
from xas_pipeline.chem import periodic as _periodic, xyz as _chem_xyz, hessian as _chem_hess

SCHEDULER_SUBMIT_COMMAND = _sched.SUBMIT_COMMAND
_default_scheduler = _sched.default_scheduler_name

CORVUS_TEMPLATE_BY_MODE = {
    "xas": "corvus-template-xas.in",
}

# Sub-input cards the combined xas template reads via xanes_input{}/exafs_input{}.
# They carry no [CAPS] placeholders, so they are copied into the run dir verbatim
# under the exact names the template references.
CORVUS_SUBINPUTS = ("xanes.in", "exafs.in")


def _resolve_dir(path_str: str) -> Path:
    run_dir = Path(path_str).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")
    return run_dir


# Chemistry parsers/tables now live in xas_pipeline.chem; these module-level
# aliases keep the old private names working for internal callers below (and for
# the characterization tests that reach them via importlib). Retired in phase 9.
ANGSTROM_PER_BOHR = _periodic.ANGSTROM_PER_BOHR
_ATOMIC_SYMBOLS = _periodic.ATOMIC_SYMBOLS
_ATOMIC_MASSES_AMU = _periodic.ATOMIC_MASSES_AMU
_ATOMIC_NUM_TO_SYMBOL = _periodic.ATOMIC_NUM_TO_SYMBOL
_atomic_number_from_token = _periodic.atomic_number_from_token
_atomic_mass_amu = _periodic.atomic_mass_amu
_canonical_symbol_from_token = _periodic.canonical_symbol_from_token
_read_orca_hessian = _chem_hess.read_orca_hessian
_read_xyz = _chem_xyz.read_xyz
_read_last_xyz_frame = _chem_xyz.read_last_xyz_frame


# Geometry selection moved to xas_pipeline.chem.xyz so the interp-Hessian stage
# resolves the same file this stage does; aliased here for internal callers (and
# the characterization tests that reach it via importlib).
_select_latest_xyz = _chem_xyz.select_run_xyz


def _write_clean_corvus_xyz(source_xyz: Path, dest_xyz: Path) -> None:
    lines = source_xyz.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Input does not look like XYZ (missing header lines): {source_xyz}")

    try:
        natoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid XYZ atom count header in {source_xyz}: {lines[0]!r}") from exc

    cleaned = [lines[0].strip(), lines[1].strip()]
    parsed_atoms = 0
    for raw in lines[2:]:
        if parsed_atoms >= natoms:
            break
        if not raw.strip():
            continue
        main_part = raw.split("#", 1)[0]
        tokens = main_part.split()
        if len(tokens) < 4:
            raise ValueError(f"Invalid atom line in {source_xyz}: {raw!r}")
        atom_symbol = _canonical_symbol_from_token(tokens[0])
        cleaned.append("{:<2} {:>12} {:>12} {:>12}".format(atom_symbol, *tokens[1:4]))
        parsed_atoms += 1

    if parsed_atoms != natoms:
        raise ValueError(
            f"XYZ atom count mismatch in {source_xyz}: header={natoms}, parsed={parsed_atoms}"
        )

    dest_xyz.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _validate_latest_trj_matches_corvus_xyz(
    run_dir: Path, corvus_xyz_path: Path, tolerance_angstrom: float = 1e-4
) -> None:
    trj_files = [path for path in run_dir.glob("*_trj.xyz") if path.is_file()]
    if not trj_files:
        return

    latest_trj = max(trj_files, key=lambda path: (path.stat().st_mtime, path.name))

    trj_atomic_numbers, trj_coords = _read_last_xyz_frame(latest_trj)
    xyz_atomic_numbers, xyz_coords = _read_last_xyz_frame(corvus_xyz_path)

    if len(trj_atomic_numbers) != len(xyz_atomic_numbers):
        raise ValueError(
            "Trajectory/Corvus XYZ atom-count mismatch: "
            f"{latest_trj.name} has {len(trj_atomic_numbers)}, "
            f"{corvus_xyz_path.name} has {len(xyz_atomic_numbers)}"
        )
    if not np.array_equal(trj_atomic_numbers, xyz_atomic_numbers):
        raise ValueError(
            "Trajectory/Corvus XYZ element-order mismatch between "
            f"{latest_trj.name} and {corvus_xyz_path.name}"
        )

    max_abs_diff = float(np.max(np.abs(trj_coords - xyz_coords)))
    if max_abs_diff > tolerance_angstrom:
        raise ValueError(
            "Trajectory/Corvus XYZ coordinate mismatch: "
            f"max|delta|={max_abs_diff:.6e} A exceeds tolerance {tolerance_angstrom:.1e} A "
            f"(trj={latest_trj.name}, corvus={corvus_xyz_path.name})"
        )

    print(
        "Validated trajectory consistency: "
        f"{latest_trj.name} (last frame) matches {corvus_xyz_path.name} "
        f"within {tolerance_angstrom:.1e} A"
    )


def _print_atom_pair_blocks(hessian: np.ndarray, natoms: int, stream=None) -> None:
    if stream is None:
        stream = sys.stdout

    for i in range(natoms):
        for j in range(natoms):
            print(i + 1, j + 1, file=stream)
            ii = 0
            while ii < 3:
                print(
                    " ".join(
                        f"{h:12.6e}" for h in hessian[3 * i + ii, 3 * j: 3 * j + 3]
                    ),
                    file=stream,
                )
                ii += 1


def _write_dym_file(
    dymout_filename: Path,
    atomic_numbers: np.ndarray,
    masses_amu: np.ndarray,
    coords_bohr: np.ndarray,
    hess: np.ndarray,
) -> None:
    """
    Write dynamic file from ORCA hessian and xyz files. Formatted as follows:
    Line 1 - dym_Type: Dynamical matrix file type (integer)
        This value is for future use. Set to 1 for now.
    Line 2 - nAt: Number of atoms (integer)
        Number of atoms in the system.
    Lines 2..2+nAt - Atomic numbers (integer)
        Atomic numbers of atoms in the system.
    Lines 2+nAt+1..2+2*nAt - Atomic masses (real, in AMU)
        Atomic masses of the atoms in the system.
    Lines 2+2*nAt+1..2+3*nAt - Atomic coordinates (real, in Bohr)
        Cartesian coordinates ("x y z") of the atoms in the system.
        Directly from xyz file, but with conversion from angstrom to Bohr
    Lines 2+3*nAt+1..End - Dynamical matrix in atom pair block format (integer and
        real, see below,in atomic units)
        From `print_atom_pair_blocks` function
    """
    natoms = int(len(atomic_numbers))
    if hess.shape != (3 * natoms, 3 * natoms):
        raise ValueError(
            f"Hessian shape {hess.shape} does not match 3N x 3N with N={natoms}"
        )

    with open(dymout_filename, "w") as f:
        f.write("1\n")
        f.write(f"{natoms}\n")
        for z in atomic_numbers:
            f.write(f"{int(z)}\n")
        for m in masses_amu:
            f.write(f"{float(m):.10f}\n")
        for (x, y, z) in coords_bohr:
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")
        _print_atom_pair_blocks(hess, natoms, stream=f)


def _copy_and_replace_job_script(
    template_path: Path,
    dest_path: Path,
    run_dir: Path,
    run_id: str,
    corvus_mode: str,
    corvus_input_basename: str,
    corvus_output_basename: str,
) -> None:
    env_path = resources.project_root() / ".env"
    templates.render(template_path, dest_path, {
        "DIRECTORY": f"{run_dir}/",
        "ID": run_id,
        "CORVUS_MODE": corvus_mode,
        "CORVUS_INPUT_BASENAME": corvus_input_basename,
        "CORVUS_OUTPUT_BASENAME": corvus_output_basename,
        "PIPELINE_ENV": env_path,
    })


def _copy_corvus_subinputs(template_dir: Path, run_dir: Path) -> None:
    """Copy the xanes.in/exafs.in sub-input cards into the run dir verbatim.

    The combined xas template references them by literal relative name
    (xanes_input{xanes.in}, exafs_input{exafs.in}), so CORVUS (run with cwd set
    to the run dir) reads them from alongside the main input.
    """
    for name in CORVUS_SUBINPUTS:
        source = template_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing Corvus sub-input card: {source}")
        shutil.copy2(source, run_dir / name)
        print(f"Copied Corvus sub-input: {name}")


def _copy_and_replace_corvus(
    template_path: Path,
    dest_path: Path,
    run_dir: Path,
    run_id: str,
    num_procs: str,
    xyz_filename: str,
) -> None:
    templates.render(template_path, dest_path, {
        "DIRECTORY": f"{run_dir}/",
        "ID": run_id,
        "PROCS": num_procs,
        "XYZ_FILE": xyz_filename,
    })


def _resolve_executable(name: str, env_var: str, fallback_paths: list[str]) -> str:
    override = os.environ.get(env_var, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise FileNotFoundError(
            f"{env_var} is set but not executable: {candidate}"
        )

    for raw_path in fallback_paths:
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    discovered = shutil.which(name)
    if discovered:
        return discovered

    searched = ", ".join([*fallback_paths, f"$PATH via '{name}'"])
    raise FileNotFoundError(
        f"Could not find executable '{name}'. Set {env_var} to its full path. Searched: {searched}"
    )


def _write_centered_dym_with_legacy_corvus(
    run_dir: Path,
    run_id: str,
    feff_dym_path: Path,
    atomic_numbers: np.ndarray,
    masses_amu: np.ndarray,
    coords_bohr: np.ndarray,
    hess: np.ndarray,
    center_index_1based: int,
) -> None:
    try:
        from corvus.dmdw import writeDym
    except Exception as exc:
        raise FileNotFoundError(
            "Could not find 'dym2feffinp' and failed to import the Python fallback "
            "'corvus.dmdw.writeDym'. Install Corvus in the active environment "
            "or set DYM2FEFFINP_BIN to the external executable."
        ) from exc

    natoms = int(len(atomic_numbers))
    dm_blocks = [
        [
            hess[3 * i : 3 * i + 3, 3 * j : 3 * j + 3].tolist()
            for j in range(natoms)
        ]
        for i in range(natoms)
    ]
    dym = {
        "dymType": 1,
        "nAt": natoms,
        "atNums": [int(z) for z in atomic_numbers.tolist()],
        "atMasses": [float(m) for m in masses_amu.tolist()],
        "atCoords": coords_bohr.tolist(),
        "dm": dm_blocks,
    }

    center_index = center_index_1based - 1
    center_coords = dym["atCoords"][center_index]
    # >>> EDIT BY CLAUDE (2026-06-22) >>>
    # Re-center the .dym on the absorber (origin). FEFF's feff.inp ATOMS list is
    # centered on the absorber, and ff2x's ab-initio DMDW Debye-Waller step (idwopt=5)
    # matches path atoms to dynamical-matrix atoms by coordinate, so the .dym MUST be
    # centered too. corvus.dmdw.writeDym force-disables its own shift for dymType==1,
    # so the old fallback emitted an UN-centered .dym -> empty EXAFS spectrum. We now
    # shift atCoords explicitly (verified byte-identical to dym2feffinp output).
    dym["atCoords"] = [
        [a - b for a, b in zip(coord, center_coords)] for coord in dym["atCoords"]
    ]
    centered = []
    for i in range(natoms):
        distance = float(np.sqrt(np.sum(np.square(dym["atCoords"][i]))))
        centered.append((i, distance))
    centered.sort(key=lambda item: item[1])
    dym["printOrder"] = [idx for idx, _distance in centered]
    # <<< END EDIT BY CLAUDE <<<

    writeDym(dym, str(feff_dym_path))
    legacy_feffinp = run_dir / f"corvus-{run_id}.feff.inp"
    legacy_feffinp.write_text(
        "# Generated by prepare-corvus.py Python fallback to document the centered DYM conversion.\n",
        encoding="utf-8",
    )
    print(
        "Used Python fallback based on corvus.dmdw.writeDym "
        f"to write {feff_dym_path.name}"
    )


# dym2feffinp holds filenames in a fixed-length Fortran buffer and silently
# TRUNCATES to this length, then writes the output under the truncated name and
# exits 0. Measured against FEFF10 10.0.0: 49 characters round-trips, 51 lands in
# a file whose name is cut at 50. We never hand it a long name (see
# _run_dym2feffinp), so this is documentation plus the check's rationale.
DYM2FEFFINP_MAX_FILENAME = 50

# Short, collision-free scratch names used for the dym2feffinp call itself. The
# run dir holds exactly one conversion, so fixed names are safe here.
_DYM_SCRATCH_IN = "dym2feffinp-in.dym"
_DYM_SCRATCH_OUT = "dym2feffinp-out.dym"


def _run_dym2feffinp(
    dym2feffinp_bin: str,
    run_dir: Path,
    dym_path: Path,
    feff_dym_path: Path,
    center_index_1based: int,
) -> None:
    """Convert ``dym_path`` to the absorber-centered ``feff_dym_path``.

    The conversion runs on short scratch filenames and the result is renamed into
    place, because dym2feffinp truncates names past
    :data:`DYM2FEFFINP_MAX_FILENAME` and writes to the truncated name instead --
    while still exiting 0, so the caller sees success and no file where it asked
    for one. Run ids long enough to trip that are ordinary here once a
    mode suffix is appended (``<pdb>_ZN_homo_d2.60_cluster3_altlocLIGA-interp``
    is 44 characters before ``corvus-`` and ``.dym`` are added), and the symptom
    was a CORVUS failure several minutes later complaining about a missing
    centered DYM, with nothing pointing at the real cause.
    """
    scratch_in = run_dir / _DYM_SCRATCH_IN
    scratch_out = run_dir / _DYM_SCRATCH_OUT
    scratch_out.unlink(missing_ok=True)
    shutil.copyfile(dym_path, scratch_in)

    try:
        subprocess.run(
            [
                dym2feffinp_bin,
                "--c",
                str(center_index_1based),
                "--d",
                scratch_out.name,
                scratch_in.name,
            ],
            check=True,
            cwd=run_dir,
        )
        # dym2feffinp reports success by exit code even when it wrote nothing, so
        # the only reliable check is whether the file is there.
        if not scratch_out.is_file():
            raise RuntimeError(
                f"dym2feffinp exited 0 but produced no centered DYM for {dym_path.name}. "
                "The FEFF DYM conversion silently truncates long output names; check "
                f"that {dym_path.name} is a valid dynamical matrix and that the "
                f"absorber index ({center_index_1based}) is in range."
            )
        scratch_out.replace(feff_dym_path)
        print(f"Wrote centered DYM: {feff_dym_path.name}")
    finally:
        scratch_in.unlink(missing_ok=True)
        scratch_out.unlink(missing_ok=True)


def main() -> int:
    # Pull site config (e.g. DYM2FEFFINP_BIN) from .env for direct/login-node runs;
    # values already exported by the wrapper/scheduler are preserved.
    config.load_env(resources.project_root() / ".env")

    parser = argparse.ArgumentParser(
        description="Prepare Corvus run directory (dym + templates)."
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory; its name is used as the run ID by default.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Explicit run ID (the basename used for <run_id>.hess, corvus-<run_id>-*.in, "
            "etc.). Defaults to the run directory's name. Needed when the run directory is "
            "named differently from the ID, e.g. a post-processed 'working-<ID>' directory."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(SCHEDULER_SUBMIT_COMMAND),
        default=_default_scheduler(),
        help="Scheduler backend used for job template lookup (default: pbs).",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=16,
        help="Processor count used to populate [PROCS] in Corvus templates (default: 16).",
    )
    parser.add_argument(
        "--corvus-mode",
        choices=sorted(CORVUS_TEMPLATE_BY_MODE),
        default="xas",
        help=(
            "Corvus target to prepare. Only 'xas' is supported: renders "
            "corvus-<id>-xas.in and copies the xanes.in/exafs.in sub-input cards."
        ),
    )
    args = parser.parse_args()

    run_dir = _resolve_dir(args.run_dir)
    run_id = args.run_id if args.run_id else run_dir.name
    corvus_modes = [args.corvus_mode]

    template_dir = resources.template_root()
    scheduler_dir = template_dir / f"{args.scheduler}-scripts"
    job_template_path = scheduler_dir / "corvus-job.script"

    template_paths = {
        mode: template_dir / CORVUS_TEMPLATE_BY_MODE[mode] for mode in corvus_modes
    }
    for mode, template_path in template_paths.items():
        if not template_path.exists():
            raise FileNotFoundError(
                f"Missing Corvus template for mode '{mode}': {template_path}"
            )
    if not job_template_path.exists():
        raise FileNotFoundError(f"Missing corvus-job.script at {job_template_path}")

    hess_path = run_dir / f"{run_id}.hess"
    source_xyz_path = _select_latest_xyz(run_dir, run_id)
    xyz_path = run_dir / f"corvus-begin-{run_id}.xyz"
    dym_path = run_dir / f"{run_id}.dym"

    if not hess_path.exists():
        raise FileNotFoundError(f"Missing Hessian file: {hess_path}")

    _write_clean_corvus_xyz(source_xyz_path, xyz_path)
    print(f"Selected source XYZ: {source_xyz_path.name}")
    print(f"Wrote standardized CORVUS XYZ: {xyz_path.name}")
    _validate_latest_trj_matches_corvus_xyz(run_dir, xyz_path)

    hess, natoms_hess = _read_orca_hessian(hess_path)
    atomic_numbers, masses_amu, coords_bohr = _read_xyz(xyz_path)
    if natoms_hess != len(atomic_numbers):
        raise ValueError(
            f"Atom count mismatch: Hessian implies {natoms_hess} atoms, XYZ has {len(atomic_numbers)}"
        )

    _write_dym_file(dym_path, atomic_numbers, masses_amu, coords_bohr, hess)

    zn_indices = np.where(atomic_numbers == _ATOMIC_SYMBOLS["ZN"])[0]
    if zn_indices.size == 0:
        raise ValueError("No Zn atoms found in XYZ; cannot center DYM on absorber.")

    zn_index_1based = int(zn_indices[0]) + 1
    feff_dym_path = run_dir / f"corvus-{run_id}.dym"
    try:
        # The site-specific FEFF10 path now lives in .env (DYM2FEFFINP_BIN); it is
        # loaded into the environment above (pipeline_env.load_env) or exported by
        # the corvus wrapper. Without it, dym2feffinp must be on PATH or at the
        # generic /opt/feff10 location, else we fall back to the Python DYM writer.
        dym2feffinp_bin = _resolve_executable(
            "dym2feffinp",
            "DYM2FEFFINP_BIN",
            [
                "/opt/feff10/bin/MPI/dym2feffinp",
            ],
        )
    except FileNotFoundError as exc:
        print(f"{exc}")
        print("Falling back to corvus.feff.legacy_dym2feffinp().")
        _write_centered_dym_with_legacy_corvus(
            run_dir,
            run_id,
            feff_dym_path,
            atomic_numbers,
            masses_amu,
            coords_bohr,
            hess,
            zn_index_1based,
        )
    else:
        print(f"Using dym2feffinp executable: {dym2feffinp_bin}")
        _run_dym2feffinp(dym2feffinp_bin, run_dir, dym_path, feff_dym_path, zn_index_1based)

    _copy_corvus_subinputs(template_dir, run_dir)

    num_procs = args.num_procs
    generated_job_scripts = []
    for mode in corvus_modes:
        corvus_in_basename = f"corvus-{run_id}-{mode}.in"
        corvus_out_basename = f"corvus-{run_id}-{mode}.out"
        corvus_job_basename = f"corvus-job-{mode}.script"

        corvus_in_dest = run_dir / corvus_in_basename
        _copy_and_replace_corvus(
            template_paths[mode],
            corvus_in_dest,
            run_dir,
            run_id,
            num_procs,
            xyz_path.name,
        )

        job_script_dest = run_dir / corvus_job_basename
        _copy_and_replace_job_script(
            job_template_path,
            job_script_dest,
            run_dir,
            run_id,
            mode,
            corvus_in_basename,
            corvus_out_basename,
        )
        generated_job_scripts.append(job_script_dest)

    submit_command = SCHEDULER_SUBMIT_COMMAND[args.scheduler]
    for job_script in generated_job_scripts:
        print(f"Job can be submitted with: {submit_command} {job_script}")
    return 0

if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
