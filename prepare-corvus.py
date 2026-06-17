#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


SCHEDULER_SUBMIT_COMMAND = {
    "pbs": "qsub",
    "slurm": "sbatch",
}


def _default_scheduler() -> str:
    scheduler = os.environ.get("PIPELINE_SCHEDULER", "pbs").strip().lower()
    if scheduler not in SCHEDULER_SUBMIT_COMMAND:
        supported = ", ".join(sorted(SCHEDULER_SUBMIT_COMMAND))
        raise SystemExit(
            f"Invalid PIPELINE_SCHEDULER={scheduler!r}. Supported values: {supported}"
        )
    return scheduler


def _resolve_dir(path_str: str) -> Path:
    run_dir = Path(path_str).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")
    return run_dir


ANGSTROM_PER_BOHR = 0.52917724899

_ATOMIC_SYMBOLS = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
}

_ATOMIC_MASSES_AMU = {
    1: 1.00794,
    2: 4.002602,
    3: 6.941,
    4: 9.012182,
    5: 10.811,
    6: 12.0107,
    7: 14.0067,
    8: 15.9994,
    9: 18.9984032,
    10: 20.1797,
    11: 22.98976928,
    12: 24.3050,
    13: 26.9815386,
    14: 28.0855,
    15: 30.973762,
    16: 32.065,
    17: 35.453,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955912,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938045,
    26: 55.845,
    27: 58.933195,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.64,
    33: 74.92160,
    34: 78.96,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90585,
    40: 91.224,
    41: 92.90638,
    42: 95.96,
    43: 98.0,
    44: 101.07,
    45: 102.90550,
    46: 106.42,
    47: 107.8682,
    48: 112.411,
    49: 114.818,
    50: 118.710,
    51: 121.760,
    52: 127.60,
    53: 126.90447,
    54: 131.293,
}

_ATOMIC_NUM_TO_SYMBOL = {
    z: sym[0] + sym[1:].lower() for sym, z in _ATOMIC_SYMBOLS.items()
}


def _atomic_number_from_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("Empty atom token")
    if token.isdigit():
        return int(token)
    sym = token.upper()
    if sym not in _ATOMIC_SYMBOLS:
        raise ValueError(
            f"Unknown element symbol '{token}'. Extend _ATOMIC_SYMBOLS/_ATOMIC_MASSES_AMU."
        )
    return _ATOMIC_SYMBOLS[sym]


def _atomic_mass_amu(z: int) -> float:
    if z not in _ATOMIC_MASSES_AMU:
        raise ValueError(
            f"Unknown atomic mass for Z={z}. Extend _ATOMIC_MASSES_AMU for this element."
        )
    return float(_ATOMIC_MASSES_AMU[z])


def _canonical_symbol_from_token(token: str) -> str:
    z = _atomic_number_from_token(token)
    if z not in _ATOMIC_NUM_TO_SYMBOL:
        raise ValueError(f"Missing canonical symbol for atomic number {z}")
    return _ATOMIC_NUM_TO_SYMBOL[z]


def _read_orca_hessian(filename: Path) -> tuple[np.ndarray, int]:
    with open(filename, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "$HESSIAN" in line.upper():
            start = i + 1
            break

    if start is None:
        raise ValueError("HESSIAN section not found.")

    size = int(lines[start].strip())
    natoms = int(size / 3)
    hess_start = start + 1

    hessian = np.zeros((size, size))
    row = 0
    i = hess_start

    while row < size:
        header = lines[i].split()
        ncols = len(header)
        cols = [int(x) for x in header]
        i += 1
        for r in range(size):
            parts = lines[i + r].split()
            values = list(map(float, parts[1:1 + ncols]))
            for c, val in zip(cols, values):
                hessian[r, c] = val
        if c >= size - 1:
            break
        i += size
        row += 1

    return hessian, natoms


def _read_xyz(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    lines = [ln for ln in raw_lines if ln]
    if not lines:
        raise ValueError("Empty XYZ file")

    atom_lines = None
    try:
        n = int(lines[0].split()[0])
        if len(lines) < 2 + n:
            raise ValueError("XYZ file too short for declared atom count")
        atom_lines = lines[2:2 + n]
    except ValueError:
        atom_lines = [ln for ln in lines if len(ln.split()) >= 4]

    atomic_numbers = []
    masses_amu = []
    coords_bohr = []

    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        z = _atomic_number_from_token(parts[0])
        x_a, y_a, z_a = map(float, parts[1:4])
        atomic_numbers.append(z)
        masses_amu.append(_atomic_mass_amu(z))
        coords_bohr.append(
            [x_a / ANGSTROM_PER_BOHR, y_a / ANGSTROM_PER_BOHR, z_a / ANGSTROM_PER_BOHR]
        )

    if not atomic_numbers:
        raise ValueError("No atom records found in XYZ")

    return (
        np.array(atomic_numbers, dtype=int),
        np.array(masses_amu, dtype=float),
        np.array(coords_bohr, dtype=float),
    )


def _select_latest_xyz(run_dir: Path) -> Path:
    xyz_files = [path for path in run_dir.glob("*.xyz") if path.is_file()]
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {run_dir}")

    # Prefer single-geometry outputs and ignore any prior standardized CORVUS copies.
    preferred = [
        path
        for path in xyz_files
        if not path.stem.lower().endswith("_trj")
        and not path.name.startswith("corvus-begin-")
    ]
    pool = preferred if preferred else xyz_files

    return max(pool, key=lambda path: (path.stat().st_mtime, path.name))


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


def _read_last_xyz_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the final XYZ frame as (atomic_numbers, coords_angstrom)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    last_atomic_numbers = None
    last_coords = None

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            natoms = int(line.split()[0])
        except (IndexError, ValueError):
            i += 1
            continue

        if natoms <= 0:
            raise ValueError(f"Invalid atom count in XYZ frame ({natoms}) for {path}")
        if i + 2 + natoms > len(lines):
            raise ValueError(f"Truncated XYZ frame in {path} near line {i + 1}")

        atom_lines = lines[i + 2 : i + 2 + natoms]
        atomic_numbers = []
        coords = []
        for raw in atom_lines:
            main_part = raw.split("#", 1)[0]
            tokens = main_part.split()
            if len(tokens) < 4:
                raise ValueError(f"Invalid atom line in {path}: {raw!r}")
            z = _atomic_number_from_token(tokens[0])
            x_a, y_a, z_a = map(float, tokens[1:4])
            atomic_numbers.append(z)
            coords.append([x_a, y_a, z_a])

        last_atomic_numbers = np.array(atomic_numbers, dtype=int)
        last_coords = np.array(coords, dtype=float)
        i = i + 2 + natoms

    if last_atomic_numbers is None or last_coords is None:
        raise ValueError(f"No XYZ frames parsed from {path}")

    return last_atomic_numbers, last_coords


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


def _copy_and_replace_job_script(template_path: Path, dest_path: Path, run_dir: Path, run_id: str) -> None:
    content = template_path.read_text(encoding="utf-8")
    content = content.replace("[directory]", f"{run_dir}/")
    content = content.replace("[ID]", run_id)
    dest_path.write_text(content, encoding="utf-8")


def _copy_and_replace_corvus(
    template_path: Path,
    dest_path: Path,
    run_dir: Path,
    run_id: str,
    num_procs: str,
    xyz_filename: str,
) -> None:
    content = template_path.read_text(encoding="utf-8")
    content = content.replace("[DIRECTORY]", f"{run_dir}/")
    content = content.replace("[ID]", run_id)
    content = content.replace("[PROCS]", str(num_procs))
    content = content.replace("[XYZ_FILE]", xyz_filename)
    dest_path.write_text(content, encoding="utf-8")


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
    centered = []
    center_coords = dym["atCoords"][center_index]
    for i in range(natoms):
        shifted = [a - b for a, b in zip(dym["atCoords"][i], center_coords)]
        distance = float(np.sqrt(np.sum(np.square(shifted))))
        centered.append((i, distance))
    centered.sort(key=lambda item: item[1])
    dym["printOrder"] = [idx for idx, _distance in centered]

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Corvus run directory (dym + templates)."
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory; its name is used as the run ID.",
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
        help="Processor count used to populate [PROCS] in corvus-template.in (default: 16).",
    )
    args = parser.parse_args()

    run_dir = _resolve_dir(args.run_dir)
    run_id = run_dir.name

    script_dir = Path(__file__).resolve().parent
    scheduler_dir = script_dir / f"{args.scheduler}-scripts"
    corvus_template_path = script_dir / "corvus-template.in"
    job_template_path = scheduler_dir / "corvus-job.script"

    if not corvus_template_path.exists():
        raise FileNotFoundError(f"Missing corvus-template.in at {corvus_template_path}")
    if not job_template_path.exists():
        raise FileNotFoundError(f"Missing corvus-job.script at {job_template_path}")

    hess_path = run_dir / f"{run_id}.hess"
    source_xyz_path = _select_latest_xyz(run_dir)
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
        dym2feffinp_bin = _resolve_executable(
            "dym2feffinp",
            "DYM2FEFFINP_BIN",
            ["/opt/feff10/bin/MPI/dym2feffinp"],
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
        dym2feffinp_cmd = [
            dym2feffinp_bin,
            "--c",
            str(zn_index_1based),
            "--d",
            str(feff_dym_path.name),
            str(dym_path.name),
        ]
        print(f"Using dym2feffinp executable: {dym2feffinp_bin}")
        subprocess.run(dym2feffinp_cmd, check=True, cwd=run_dir)

    num_procs = args.num_procs
    corvus_in_dest = run_dir / f"corvus-{run_id}.in"
    _copy_and_replace_corvus(
        corvus_template_path,
        corvus_in_dest,
        run_dir,
        run_id,
        num_procs,
        xyz_path.name,
    )

    job_script_dest = run_dir / "corvus-job.script"
    _copy_and_replace_job_script(job_template_path, job_script_dest, run_dir, run_id)

    submit_command = SCHEDULER_SUBMIT_COMMAND[args.scheduler]
    print(f"Job can be submitted with: {submit_command} {run_dir}/corvus-job.script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
