#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from xas_pipeline import layout
from xas_pipeline.batch_log import append_outcomes, find_batch_log
from xas_pipeline.chem import feff as _chem_feff

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
    }
)


# The pipeline runs a single combined target: cfavg_target{ xas }. CORVUS builds
# Corvus3_cfavg_xas/Corvus1Zn_<absorber-index>_FEFF/{xanes,exafs} and writes the
# combined spectrum to Corvus.cfavg_xas.out at the run root.
CORVUS_MODES = ("xas",)
# The combined XAS output is the deliverable spectrum; xanes/exafs are FEFF subdirs.
XAS_COMPONENTS = ("xanes", "exafs")
# CORVUS's combined configurationally-averaged spectrum (6-col xmu-like table).
CFAVG_XAS_OUTPUT = "Corvus.cfavg_xas.out"
# Directories under the batch root that are never id/run directories.
SKIP_DIR_NAMES = layout.SKIP_DIR_NAMES


# FEFF table loaders + chi(k)->chi(R) FFT live in xas_pipeline.chem.feff;
# aliased here for the combined-xas processing below.
load_feff_table = _chem_feff.load_feff_table
xmu_reports_zero_paths = _chem_feff.xmu_reports_zero_paths
xftf_larch = _chem_feff.xftf_larch


def apply_plot_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(direction="in", width=2, length=8)


def run_for_xas(cfavg_path: Path, dest_dir: Path, name: str, args: argparse.Namespace) -> List[Path]:
    """Plot XANES/EXAFS and compute chi(R) from the combined 6-col XAS output.

    Corvus.cfavg_xas.out shares xmu.dat's column layout:
      1 photon energy (eV)  2 photoelectron energy (eV)  3 k (1/A)
      4 mu                  5 mu0 (atomic background)    6 chi = mu - mu0
    XANES is plotted as mu vs photon energy; EXAFS/FFT use (k, chi) over the
    physical k > 0 region (rows below the edge carry non-physical negative k).
    Artifacts are written into dest_dir named with the id.
    """
    omega, _energy, k, mu, _mu0, chi = load_feff_table(cfavg_path)
    saved_outputs: List[Path] = []

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omega, mu, lw=2)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("XANES")
    apply_plot_style(ax)
    fig.tight_layout()
    xanes_png = dest_dir / f"xanes-{name}.png"
    fig.savefig(xanes_png, dpi=300)
    saved_outputs.append(xanes_png)
    if not args.show:
        plt.close(fig)

    exafs_mask = np.isfinite(k) & (k > 0)
    ex_k = k[exafs_mask]
    ex_chi = chi[exafs_mask]

    if ex_k.size:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ex_k, ex_chi, lw=2)
        ax.set_xlabel(r"$k\ (1/\AA)$")
        ax.set_ylabel(r"$\chi(k)$")
        ax.set_title("EXAFS")
        apply_plot_style(ax)
        fig.tight_layout()
        exafs_png = dest_dir / f"exafs-{name}.png"
        fig.savefig(exafs_png, dpi=300)
        saved_outputs.append(exafs_png)
        if not args.show:
            plt.close(fig)

    if not args.skip_fft and ex_k.size:
        r, chir = xftf_larch(
            ex_k,
            ex_chi,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
            kweight=args.kweight,
            kstep=args.kstep,
            rmax_out=args.rmax,
            window=args.window,
        )

        out_dat = dest_dir / f"chi-R-{name}.dat"
        header = "r  chir_mag  chir_re  chir_im"
        np.savetxt(
            out_dat,
            np.column_stack([r, np.abs(chir), chir.real, chir.imag]),
            header=header,
        )
        saved_outputs.append(out_dat)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r, np.abs(chir), lw=2)
        ax.set_xlabel(r"$R\ (\AA)$")
        ax.set_ylabel(r"$|\chi(R)|$")
        ax.set_title("EXAFS FT")
        apply_plot_style(ax)
        fig.tight_layout()
        out_png = dest_dir / f"chi-R-{name}.png"
        fig.savefig(out_png, dpi=300)
        saved_outputs.append(out_png)
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()
    else:
        for out_path in saved_outputs:
            print(f"Saved: {out_path}")

    return saved_outputs


def mode_feff_dir(working_root: Path, mode: str) -> Path:
    """Resolve the CORVUS FEFF dir for a mode.

    CORVUS names it Corvus1Zn_<absorber-index>_FEFF (the index varies per
    structure, e.g. Corvus1Zn_0_FEFF here, Corvus1Zn_32_FEFF elsewhere), so we
    glob rather than hardcode. Returns the first match, or a deterministic
    non-existent path when none is present (so is_feff_dir() reports False).
    """
    mode_root = working_root / f"Corvus3_cfavg_{mode}"
    matches = sorted(mode_root.glob("Corvus1Zn_*_FEFF"))
    return matches[0] if matches else mode_root / "Corvus1Zn_FEFF"


def cfavg_xas_output(working_root: Path) -> Path:
    """Path to the combined 6-col XAS spectrum at a working root."""
    return working_root / CFAVG_XAS_OUTPUT


def is_feff_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    # The combined xas run nests per-component FEFF outputs in xanes/ and exafs/
    # subdirs; older flat layouts put xmu.dat/chi.dat directly in the FEFF dir.
    # Accept either so both are detectable.
    return (
        (path / "xanes" / "xmu.dat").is_file()
        or (path / "exafs" / "xmu.dat").is_file()
        or (path / "exafs" / "chi.dat").is_file()
        or (path / "xmu.dat").is_file()
        or (path / "chi.dat").is_file()
    )


def xas_is_valid(cfavg_path: Path, feff_dir: Path) -> tuple[bool, str]:
    """Return (valid, reason) for a combined xas run.

    The deliverable is Corvus.cfavg_xas.out; it must exist and hold a non-empty
    6-column table. As an extra EXAFS sanity check we flag the "0/0 paths used"
    case from the exafs subdir's xmu.dat (XANES legitimately reports 0/0 paths,
    so we only check the exafs component).
    """
    if not cfavg_path.is_file():
        return False, f"missing {CFAVG_XAS_OUTPUT} (CORVUS produced no combined XAS spectrum)"
    try:
        data = np.genfromtxt(cfavg_path, comments="#")
    except (OSError, ValueError) as exc:
        return False, f"could not read {cfavg_path.name}: {exc}"
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 6:
        return False, f"{cfavg_path.name} is empty or malformed (expected a 6-column table)"

    exafs_xmu = feff_dir / "exafs" / "xmu.dat"
    if exafs_xmu.is_file() and xmu_reports_zero_paths(exafs_xmu):
        return False, "exafs xmu.dat reports 0/0 paths used (FEFF found no EXAFS scattering paths)"
    return True, "ok"


def move_unprocessed_contents_to_working(system_dir: Path, working_dir: Path, output_dir: Path):
    # A run dir can also host other modes' run dirs (<id>-<mode>/) when a mode is
    # added to a batch that already ran. Those are separate runs with their own
    # results -- sweeping them into this run's working- dir would bury them.
    skip_names = {working_dir.name, output_dir.name}
    skip_names.update(child.name for child in layout.nested_mode_run_dirs(system_dir))
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
    return layout.has_working_output_pair(system_dir)


def working_roots(system_dir: Path) -> List[Path]:
    """Roots under which Corvus3_cfavg_<mode> dirs may live (flat or split layout)."""
    return layout.working_roots(system_dir)


def is_process_target(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False
    if has_working_output_pair(system_dir):
        return True
    for root in working_roots(system_dir):
        if cfavg_xas_output(root).is_file():
            return True
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

    # iter_id_dirs descends one level into per-structure group dirs, so a batch
    # laid out as <id>/<id>-<mode>/ yields each mode's run dir here just as a
    # pre-grouping batch yields its flat <id>/ dirs.
    return [child for child in layout.iter_id_dirs(parent_dir) if is_process_target(child)]


def process_system_dir(system_dir: Path, args: argparse.Namespace) -> tuple[bool, List[str]]:
    """Process the combined xas run for one id.

    The deliverable is Corvus.cfavg_xas.out (source of truth): plots, the chi(R)
    FFT, and the copied spectrum are all derived from it. Raw per-component FEFF
    tables are copied for provenance. Returns (ok, failures); ok is False (the id
    is treated as a CORVUS failure) when the combined spectrum is missing/invalid
    or processing raised.
    """
    name = system_dir.name
    output_dir = system_dir / f"output-{name}"
    working_dir = system_dir / f"working-{name}"

    already_processed = output_dir.is_dir() and working_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    working_dir.mkdir(exist_ok=True)

    if not already_processed:
        move_unprocessed_contents_to_working(system_dir, working_dir, output_dir)

    failures: List[str] = []
    cfavg_path = cfavg_xas_output(working_dir)
    feff_dir = mode_feff_dir(working_dir, "xas")

    valid, reason = xas_is_valid(cfavg_path, feff_dir)
    if not valid:
        failures.append(f"xas: {reason}")
        # Still copy the structure file so partial output dirs are useful.
        _copy_xyz(system_dir, working_dir, output_dir, name)
        return False, failures

    ok = True
    try:
        run_for_xas(cfavg_path, output_dir, name, args)
    except Exception as exc:  # noqa: BLE001 - treat as CORVUS failure, keep going
        failures.append(f"xas: processing error ({exc})")
        ok = False

    # The combined 6-col spectrum is the deliverable.
    copy_if_exists(cfavg_path, output_dir / f"xas-{name}.dat", "Corvus.cfavg_xas.out")

    # Raw per-component FEFF tables, copied for provenance.
    for component in XAS_COMPONENTS:
        comp_xmu = feff_dir / component / "xmu.dat"
        copy_if_exists(comp_xmu, output_dir / f"xmu-{component}-{name}.dat", f"xmu.dat ({component})")
    exafs_chi = feff_dir / "exafs" / "chi.dat"
    if exafs_chi.is_file():
        copy_if_exists(exafs_chi, output_dir / f"chi-exafs-{name}.dat", "chi.dat (exafs)")

    _copy_xyz(system_dir, working_dir, output_dir, name)

    return ok, failures


def _copy_xyz(system_dir: Path, working_dir: Path, output_dir: Path, name: str) -> None:
    xyz_src_candidates = [working_dir / f"{name}.xyz", system_dir / f"{name}.xyz"]
    xyz_src = next((path for path in xyz_src_candidates if path.is_file()), xyz_src_candidates[0])
    copy_if_exists(xyz_src, output_dir / f"{name}.xyz", "xyz")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Process the combined CORVUS xas output: for each id, read "
            "Corvus.cfavg_xas.out (the 6-col XAS spectrum), generate XANES/EXAFS "
            "plots and the chi(R) FFT, and copy xas-<id>.dat, chi-R-<id>.dat, the "
            "per-component xmu/chi provenance tables, and the xyz into output-<id>. "
            "Ids whose xas run failed are recorded in corvus-failed-ids.txt for the "
            "download stage."
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

if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
