#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pipeline_batch_log import append_outcomes, find_batch_log

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
    }
)


# CORVUS modes the pipeline can produce, each as Corvus3_cfavg_<mode>/Corvus1Zn_FEFF.
CORVUS_MODES = ("xanes", "exafs", "xas")
# Configurationally-averaged spectrum components copied per id (xanes-<id>.dat, exafs-<id>.dat).
CFAVG_COMPONENTS = ("xanes", "exafs")
# Directories under the batch root that are never id/run directories.
SKIP_DIR_NAMES = {
    "failed-orca",
    "failed-corvus",
    "downloading-station",
    "xyz_files",
    "optimized_xyz_files",
}


def load_feff_table(path: Path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns in {path}, got {data.shape[1]}")
    omega = data[:, 0]
    energy = data[:, 1]
    k = data[:, 2]
    mu = data[:, 3]
    mu0 = data[:, 4]
    chi = data[:, 5]
    return omega, energy, k, mu, mu0, chi


def load_xmu_columns(path: Path):
    # FEFF xmu.dat uses the same 6-column numeric layout as other FEFF tables.
    return load_feff_table(path)


def xmu_reports_zero_paths(path: Path) -> bool:
    pattern = re.compile(r"^#\s*0\s*/\s*0\s+paths\s+used\b", re.IGNORECASE)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if pattern.match(raw_line.strip()):
                return True
    return False


def load_chi_dat(path: Path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {data.shape[1]}")
    k = data[:, 0]
    chi = data[:, 1]
    return k, chi


def xftf_larch(k, chi, kmin, kmax, dk, kweight, kstep, rmax_out, window):
    from larch import Group
    from larch.xafs import xftf

    grp = Group()
    grp.k = k
    grp.chi = chi
    xftf(
        grp.k,
        grp.chi,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        kweight=kweight,
        kstep=kstep,
        rmax_out=rmax_out,
        window=window,
        group=grp,
    )
    return grp.r, grp.chir


def apply_plot_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(direction="in", width=2, length=8)


def resolve_xanes_path(feff_dir: Path) -> Path | None:
    path = feff_dir / "xanes_K.dat"
    return path if path.is_file() else None


def resolve_exafs_path(feff_dir: Path) -> Path | None:
    # Accept both legacy and mode-specific filename variants.
    candidates = [feff_dir / "exafs_K.dat", feff_dir / "exafs_k.dat"]
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_xmu_path(feff_dir: Path) -> Path | None:
    path = feff_dir / "xmu.dat"
    return path if path.is_file() else None


def run_for_feff_dir(feff_dir: Path, args: argparse.Namespace):
    """Generate XANES/EXAFS plots and the chi(R) FFT (chi_R.dat) inside a FEFF dir."""
    xanes_path = resolve_xanes_path(feff_dir)
    exafs_path = resolve_exafs_path(feff_dir)
    xmu_path = resolve_xmu_path(feff_dir)
    if xanes_path is None and exafs_path is None:
        if xmu_path is None and not (feff_dir / "chi.dat").is_file():
            raise FileNotFoundError(
                f"No supported FEFF outputs in {feff_dir} "
                "(expected one of xanes_K.dat, exafs_K.dat, exafs_k.dat, xmu.dat, chi.dat)"
            )
        print(
            f"warning: no xanes_K.dat/exafs_K.dat/exafs_k.dat in {feff_dir}; "
            "plotting from xmu.dat where available and continuing with FFT outputs"
        )

    saved_outputs = []

    if xanes_path is not None:
        x_omega, _, _, x_mu, _, _ = load_feff_table(xanes_path)
    elif xmu_path is not None:
        x_omega, _, _, x_mu, _, _ = load_xmu_columns(xmu_path)
    else:
        x_omega = None
        x_mu = None

    if x_omega is not None and x_mu is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_omega, x_mu, lw=2)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"$\mu$")
        ax.set_title("XANES")
        apply_plot_style(ax)
        fig.tight_layout()
        xanes_png = feff_dir / "xanes_K.png"
        fig.savefig(xanes_png, dpi=300)
        saved_outputs.append(xanes_png)
        if not args.show:
            plt.close(fig)

    if exafs_path is not None:
        _, _, ex_k, _, _, ex_chi = load_feff_table(exafs_path)
    elif xmu_path is not None:
        _, _, ex_k, _, _, ex_chi = load_xmu_columns(xmu_path)
    else:
        ex_k = None
        ex_chi = None

    if ex_k is not None and ex_chi is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ex_k, ex_chi, lw=2)
        ax.set_xlabel(r"$k\ (1/\AA)$")
        ax.set_ylabel(r"$\chi(k)$")
        ax.set_title("EXAFS")
        apply_plot_style(ax)
        fig.tight_layout()
        exafs_png = feff_dir / "exafs_K.png"
        fig.savefig(exafs_png, dpi=300)
        saved_outputs.append(exafs_png)
        if not args.show:
            plt.close(fig)

    if not args.skip_fft:
        chi_path = feff_dir / "chi.dat"
        if not chi_path.exists():
            print(f"warning: missing chi.dat for FFT step: {chi_path}")
        else:
            k, chi = load_chi_dat(chi_path)

            r, chir = xftf_larch(
                k,
                chi,
                kmin=args.kmin,
                kmax=args.kmax,
                dk=args.dk,
                kweight=args.kweight,
                kstep=args.kstep,
                rmax_out=args.rmax,
                window=args.window,
            )

            chir_mag = np.abs(chir)
            chir_re = chir.real
            chir_im = chir.imag

            out_dat = feff_dir / "chi_R.dat"
            header = "r  chir_mag  chir_re  chir_im"
            np.savetxt(out_dat, np.column_stack([r, chir_mag, chir_re, chir_im]), header=header)
            saved_outputs.append(out_dat)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(r, chir_mag, lw=2)
            ax.set_xlabel(r"$R\ (\AA)$")
            ax.set_ylabel(r"$|\chi(R)|$")
            ax.set_title("EXAFS FT")
            apply_plot_style(ax)
            fig.tight_layout()

            out_png = feff_dir / "chi_R.png"
            fig.savefig(out_png, dpi=300)
            saved_outputs.append(out_png)
            if not args.show:
                plt.close(fig)

    if args.show:
        plt.show()
    else:
        for out_path in saved_outputs:
            print(f"Saved: {out_path}")


def parse_cfavg_mode_from_input(input_path: Path) -> str | None:
    pattern = re.compile(r"cfavg_target\s*\{\s*(xas|xanes|exafs)\s*\}", re.IGNORECASE)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).lower()


def detect_cfavg_modes(base: Path) -> List[str]:
    if not base.is_dir():
        return []

    modes: List[str] = []
    seen = set()
    search_roots = [base] + [
        child for child in base.iterdir() if child.is_dir() and child.name.startswith("working")
    ]

    for root in search_roots:
        for pattern in ("corvus-*.in", "*.in"):
            for input_path in sorted(root.glob(pattern)):
                mode = parse_cfavg_mode_from_input(input_path)
                if mode and mode not in seen:
                    seen.add(mode)
                    modes.append(mode)

    return modes


def mode_feff_dir(working_root: Path, mode: str) -> Path:
    return working_root / f"Corvus3_cfavg_{mode}" / "Corvus1Zn_FEFF"


def is_feff_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    # Accept both post-processed spectra and raw FEFF outputs.
    # Some EXAFS runs provide chi/xmu tables but no exafs_K.dat file.
    return (
        (path / "xanes_K.dat").is_file()
        or (path / "exafs_K.dat").is_file()
        or (path / "exafs_k.dat").is_file()
        or (path / "xmu.dat").is_file()
        or (path / "chi.dat").is_file()
    )


def feff_dir_is_valid(feff_dir: Path, mode: str) -> tuple[bool, str]:
    """Return (valid, reason) for a mode's FEFF directory.

    The "0/0 paths used" check only applies to EXAFS: XANES is a full
    multiple-scattering calculation that legitimately reports 0/0 paths while
    still producing a valid xmu.dat spectrum, so it must not be flagged as failed.
    """
    if not feff_dir.is_dir():
        return False, "FEFF directory missing (CORVUS produced no output for this mode)"
    if not is_feff_dir(feff_dir):
        return False, "FEFF directory has no spectral output files"
    if mode == "exafs":
        xmu = feff_dir / "xmu.dat"
        if xmu.is_file() and xmu_reports_zero_paths(xmu):
            return False, "xmu.dat reports 0/0 paths used (FEFF found no EXAFS scattering paths)"
    return True, "ok"


def _read_cfavg_dict(path: Path):
    """Parse a combined Corvus.cfavg(.xas).out file written as a Python dict literal."""
    try:
        data = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError):
        return None
    return data if isinstance(data, dict) else None


def copy_cfavg_component(search_dir: Path, component: str, dest: Path) -> bool:
    """Copy the configurationally-averaged <component> spectrum to dest.

    Prefers the per-mode file Corvus.cfavg_<component>.out; falls back to the
    combined xas output (Corvus.cfavg_xas.out / Corvus.cfavg.out) when present.
    """
    direct = search_dir / f"Corvus.cfavg_{component}.out"
    if direct.is_file():
        shutil.copy2(direct, dest)
        return True

    for cand_name in ("Corvus.cfavg_xas.out", "Corvus.cfavg.out"):
        cand = search_dir / cand_name
        if not cand.is_file():
            continue
        data = _read_cfavg_dict(cand)
        if not data or component not in data:
            continue
        try:
            arr = np.array(data[component], dtype=float)
        except (ValueError, TypeError):
            continue
        # Stored as [[x...], [y...]]; transpose to two columns.
        if arr.ndim == 2 and arr.shape[0] == 2:
            arr = arr.T
        if arr.ndim == 2 and arr.shape[1] >= 2:
            np.savetxt(dest, arr[:, :2])
            return True
    return False


def move_unprocessed_contents_to_working(system_dir: Path, working_dir: Path, output_dir: Path):
    skip_names = {working_dir.name, output_dir.name}
    for entry in list(system_dir.iterdir()):
        if entry.name in skip_names:
            continue
        shutil.move(str(entry), str(working_dir / entry.name))


def copy_if_exists(src: Path, dst: Path, label: str):
    if src.is_file():
        shutil.copy2(src, dst)
        print(f"Copied {label}: {src} -> {dst}")
    else:
        print(f"warning: missing {label}: {src}")


def has_working_output_pair(system_dir: Path) -> bool:
    name = system_dir.name
    return (system_dir / f"working-{name}").is_dir() and (system_dir / f"output-{name}").is_dir()


def working_roots(system_dir: Path) -> List[Path]:
    """Roots under which Corvus3_cfavg_<mode> dirs may live (flat or split layout)."""
    roots = [system_dir]
    roots.extend(
        child
        for child in (system_dir.iterdir() if system_dir.is_dir() else [])
        if child.is_dir() and child.name.startswith("working")
    )
    return roots


def is_process_target(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False
    if has_working_output_pair(system_dir):
        return True
    for root in working_roots(system_dir):
        if any(is_feff_dir(mode_feff_dir(root, mode)) for mode in CORVUS_MODES):
            return True
    return False


def resolve_system_targets(parent_dir: Path, recursive: bool) -> List[Path]:
    if not recursive:
        if not is_process_target(parent_dir):
            raise FileNotFoundError(
                f"No processing target found in {parent_dir}. "
                "Expected working/output directories or Corvus3_cfavg_<mode>/Corvus1Zn_FEFF dirs."
            )
        return [parent_dir]

    if is_process_target(parent_dir):
        return [parent_dir]

    targets = []
    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIR_NAMES:
            continue
        if is_process_target(child):
            targets.append(child)
    return targets


def process_system_dir(system_dir: Path, args: argparse.Namespace) -> tuple[bool, List[str]]:
    """Process every CORVUS mode for one id.

    Returns (ok, failed_modes). ok is False (the whole id is treated as a CORVUS
    failure) when any expected mode failed or no mode produced usable output.
    """
    name = system_dir.name
    output_dir = system_dir / f"output-{name}"
    working_dir = system_dir / f"working-{name}"

    already_processed = output_dir.is_dir() and working_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    working_dir.mkdir(exist_ok=True)

    if not already_processed:
        move_unprocessed_contents_to_working(system_dir, working_dir, output_dir)

    # Expected modes come from the CORVUS input files; fall back to whichever mode
    # FEFF dirs are physically present.
    expected_modes = detect_cfavg_modes(working_dir)
    present_modes = [m for m in CORVUS_MODES if is_feff_dir(mode_feff_dir(working_dir, m))]
    if not expected_modes:
        expected_modes = present_modes

    failed_modes: List[str] = []
    processed_any = False

    modes_to_check = expected_modes if expected_modes else list(CORVUS_MODES)
    for mode in modes_to_check:
        feff_dir = mode_feff_dir(working_dir, mode)
        valid, reason = feff_dir_is_valid(feff_dir, mode)
        if not valid:
            failed_modes.append(f"{mode}: {reason}")
            continue
        try:
            run_for_feff_dir(feff_dir, args)
        except Exception as exc:  # noqa: BLE001 - record and continue with other modes
            failed_modes.append(f"{mode}: processing error ({exc})")
            continue

        processed_any = True
        xmu_src = resolve_xmu_path(feff_dir)
        if xmu_src is not None:
            copy_if_exists(xmu_src, output_dir / f"xmu-{mode}-{name}.dat", f"xmu.dat ({mode})")
        chi_r_src = feff_dir / "chi_R.dat"
        if chi_r_src.is_file():
            copy_if_exists(chi_r_src, output_dir / f"chi-R-{name}.dat", f"chi_R.dat ({mode})")

    # Configurationally-averaged spectra (one per component per id).
    for component in CFAVG_COMPONENTS:
        dest = output_dir / f"{component}-{name}.dat"
        if copy_cfavg_component(working_dir, component, dest):
            print(f"Wrote cfavg {component}: {dest}")

    # Structure file.
    xyz_src_candidates = [working_dir / f"{name}.xyz", system_dir / f"{name}.xyz"]
    xyz_src = next((path for path in xyz_src_candidates if path.is_file()), xyz_src_candidates[0])
    copy_if_exists(xyz_src, output_dir / f"{name}.xyz", "xyz")

    if not modes_to_check:
        failed_modes.append("no CORVUS cfavg output found")

    ok = processed_any and not failed_modes
    return ok, failed_modes


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Process FEFF output(s): for each id, process every CORVUS mode "
            "(Corvus3_cfavg_xanes/_exafs/_xas), generate plots and chi_R.dat, and copy "
            "xmu-<mode>-<id>.dat, chi-R-<id>.dat, xanes-<id>.dat/exafs-<id>.dat and the xyz "
            "into output-<id>. Ids with any failed CORVUS mode are recorded in "
            "corvus-failed-ids.txt for the download stage."
        )
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help=(
            "Directory to process. In recursive mode, this is scanned for child system directories "
            "when it is not itself a processing target."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all matching child directories when parent_dir is not itself a target.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after saving.",
    )
    parser.add_argument(
        "--skip-fft",
        action="store_true",
        help="Skip chi(k) to chi(R) Fourier transform.",
    )
    parser.add_argument("--kmin", type=float, default=3.0)
    parser.add_argument("--kmax", type=float, default=11.0)
    parser.add_argument("--dk", type=float, default=3.0)  # dk=1
    parser.add_argument("--kweight", type=int, default=2)
    parser.add_argument("--kstep", type=float, default=0.05)
    parser.add_argument("--rmax", type=float, default=6.0)
    parser.add_argument("--window", type=str, default="kaiser")  # window="hanning"
    parser.add_argument(
        "--no-batch-log",
        action="store_true",
        help="Do not append authoritative CORVUS outcomes to batch-jobs.log.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    parent_dir = args.parent_dir
    if not parent_dir.is_dir():
        print(f"error: directory not found: {parent_dir}")
        return 2

    targets = resolve_system_targets(parent_dir, recursive=args.recursive)
    manifest = parent_dir.resolve() / "corvus-failed-ids.txt"

    if not targets:
        print(
            f"warning: no processable directories found under {parent_dir}. "
            "Expected working/output dirs or Corvus3_cfavg_<mode>/Corvus1Zn_FEFF dirs."
        )
        manifest.write_text("", encoding="utf-8")
        return 0

    failed_ids: List[str] = []
    # (job_name, status, reason) tuples appended to batch-jobs.log after processing.
    batch_outcomes: list[tuple[str, str, str | None]] = []
    for target in targets:
        print(f"\nProcessing: {target}")
        try:
            ok, failed_modes = process_system_dir(target, args)
        except Exception as exc:  # noqa: BLE001 - treat as CORVUS failure, keep going
            failed_ids.append(target.name)
            print(f"error: failed processing {target}: {exc}")
            batch_outcomes.append(
                (f"corvus-{target.name}", "FAILED", f"error processing FEFF output: {exc}")
            )
            continue
        if not ok:
            failed_ids.append(target.name)
            detail = "; ".join(failed_modes) if failed_modes else "no usable output"
            print(f"CORVUS FAILED: {target.name} -> {detail}")
            batch_outcomes.append((f"corvus-{target.name}", "FAILED", detail))
        else:
            batch_outcomes.append((f"corvus-{target.name}", "OK", None))

    unique_failed = sorted(set(failed_ids))
    manifest.write_text(
        "\n".join(unique_failed) + ("\n" if unique_failed else ""),
        encoding="utf-8",
    )

    print(
        f"\nProcessed {len(targets)} dir(s); "
        f"{len(unique_failed)} with CORVUS failure(s)."
    )
    print(f"Wrote CORVUS failed-id manifest: {manifest}")

    if not args.no_batch_log:
        # The batch root is parent_dir in the non-recursive case, or parent_dir
        # itself when scanning recursively; batch-jobs.log lives at that root.
        batch_log = find_batch_log(parent_dir.resolve())
        if batch_log is not None:
            append_outcomes(batch_log, "CORVUS outcomes", batch_outcomes)
            print(f"Appended {len(batch_outcomes)} CORVUS outcome(s) to {batch_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
