"""Turn finished CORVUS runs into deliverables.

For each run dir, the combined spectrum ``Corvus.cfavg_xas.out`` (the source of
truth) is checked with :mod:`xas_pipeline.corvus_diagnosis`; a good one is plotted
(XANES, EXAFS), Fourier transformed to chi(R), and copied with the per-component
FEFF tables and the geometry into ``output-<id>/``. The run's other files move
into ``working-<id>/``. Failed run ids go to ``corvus-failed-ids.txt`` at the
batch root (read by the auto-rerun triage and the download stage), and every
outcome is appended to ``batch-jobs.log``.

``Corvus.cfavg_xas.out`` shares xmu.dat's columns: photon energy (eV),
photoelectron energy (eV), k (1/A), mu, mu0, chi = mu - mu0.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xas_pipeline import corvus_diagnosis, layout
from xas_pipeline.batch_log import append_outcomes, find_batch_log
from xas_pipeline.chem import xyz as chem_xyz
from xas_pipeline.chem.feff import load_feff_table, xftf_larch
from xas_pipeline.corvus_diagnosis import XAS_COMPONENTS, cfavg_output, mode_feff_dir
from xas_pipeline.logs import setup_cli_logging

log = logging.getLogger(__name__)

PLOT_STYLE = {"font.family": "serif", "font.size": 16}


@dataclass(frozen=True)
class FftWindow:
    """chi(k) -> chi(R) transform settings (the CLI defaults)."""

    kmin: float = 3.0
    kmax: float = 11.0
    dk: float = 3.0
    kweight: int = 2
    kstep: float = 0.05
    rmax: float = 6.0
    window: str = "kaiser"


DEFAULT_FFT = FftWindow()


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(direction="in", width=2, length=8)


def _save_plot(path: Path, x, y, xlabel: str, ylabel: str, title: str, show: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if not show:
        plt.close(fig)
    return path


def process_spectrum(
    cfavg_path: Path, dest_dir: Path, name: str, *, fft: FftWindow | None = DEFAULT_FFT, show: bool = False
) -> list[Path]:
    """Plot XANES (mu vs photon energy) and EXAFS (chi vs k > 0) and write chi(R).

    Rows below the edge carry non-physical negative k and are left out of the
    EXAFS plot and the transform. ``fft=None`` skips the transform. Returns the
    files written into ``dest_dir`` (all named with ``name``).
    """
    omega, _energy, k, mu, _mu0, chi = load_feff_table(cfavg_path)
    saved: list[Path] = []
    with plt.rc_context(PLOT_STYLE):
        saved.append(_save_plot(dest_dir / f"xanes-{name}.png", omega, mu, "Energy (eV)", r"$\mu$", "XANES", show))

        mask = np.isfinite(k) & (k > 0)
        ex_k, ex_chi = k[mask], chi[mask]
        if ex_k.size:
            saved.append(_save_plot(dest_dir / f"exafs-{name}.png", ex_k, ex_chi,
                                    r"$k\ (1/\AA)$", r"$\chi(k)$", "EXAFS", show))
        if fft is not None and ex_k.size:
            r, chir = xftf_larch(ex_k, ex_chi, kmin=fft.kmin, kmax=fft.kmax, dk=fft.dk, kweight=fft.kweight,
                                 kstep=fft.kstep, rmax_out=fft.rmax, window=fft.window)
            out_dat = dest_dir / f"chi-R-{name}.dat"
            np.savetxt(out_dat, np.column_stack([r, np.abs(chir), chir.real, chir.imag]),
                       header="r  chir_mag  chir_re  chir_im")
            saved.append(out_dat)
            saved.append(_save_plot(dest_dir / f"chi-R-{name}.png", r, np.abs(chir),
                                    r"$R\ (\AA)$", r"$|\chi(R)|$", "EXAFS FT", show))
    if show:
        plt.show()
    else:
        for path in saved:
            log.info("Saved: %s", path)
    return saved


def is_feff_dir(path: Path) -> bool:
    """A FEFF dir with results, combined-xas (``xanes/``, ``exafs/`` subdirs) or flat layout."""
    if not path.is_dir():
        return False
    return (
        (path / "xanes" / "xmu.dat").is_file()
        or (path / "exafs" / "xmu.dat").is_file()
        or (path / "exafs" / "chi.dat").is_file()
        or (path / "xmu.dat").is_file()
        or (path / "chi.dat").is_file()
    )


def xas_is_valid(cfavg_path: Path, feff_dir: Path) -> tuple[bool, str]:
    """(valid, reason) for a combined xas run, by the same rules as the auto-rerun policy.

    See :func:`xas_pipeline.corvus_diagnosis.diagnose_spectrum`; its non-fatal
    warnings are logged.
    """
    diag = corvus_diagnosis.diagnose_spectrum(cfavg_path, feff_dir)
    for warning in diag.warnings:
        log.warning("warning: %s", warning)
    return diag.ok, diag.reason


def move_unprocessed_contents_to_working(system_dir: Path, working_dir: Path, output_dir: Path) -> None:
    """Move a run dir's loose files into ``working-<id>/``.

    Other modes' run dirs nested here (``<id>-<mode>/``, from a mode added to an
    existing batch) are separate runs and stay put.
    """
    skip_names = {working_dir.name, output_dir.name}
    skip_names.update(child.name for child in layout.nested_mode_run_dirs(system_dir))
    for entry in list(system_dir.iterdir()):
        if entry.name not in skip_names:
            shutil.move(str(entry), str(working_dir / entry.name))


def copy_if_exists(src: Path, dst: Path, label: str) -> None:
    if src.is_file():
        shutil.copy2(src, dst)
        log.info("Copied %s: %s -> %s", label, src, dst)
    else:
        log.warning("warning: missing %s: %s", label, src)


def is_process_target(system_dir: Path) -> bool:
    if not system_dir.is_dir():
        return False
    if layout.has_working_output_pair(system_dir):
        return True
    return any(
        cfavg_output(root).is_file() or is_feff_dir(mode_feff_dir(root))
        for root in layout.working_roots(system_dir)
    )


def resolve_system_targets(parent_dir: Path, recursive: bool) -> list[Path]:
    """``parent_dir`` itself if it is a run dir, else (recursive) the run dirs under it."""
    if is_process_target(parent_dir):
        return [parent_dir]
    if not recursive:
        raise FileNotFoundError(
            f"No processing target found in {parent_dir}. "
            "Expected working/output directories or Corvus3_cfavg_<mode>/Corvus1Zn_FEFF dirs."
        )
    return [child for child in layout.iter_id_dirs(parent_dir) if is_process_target(child)]


def process_system_dir(
    system_dir: Path, *, fft: FftWindow | None = DEFAULT_FFT, show: bool = False
) -> tuple[bool, list[str]]:
    """Process one run dir; returns (ok, failure reasons).

    Not ok (a CORVUS failure) when the combined spectrum is missing or invalid, or
    processing it raised. The geometry is copied either way, so a partial
    ``output-<id>/`` is still useful.
    """
    name = system_dir.name
    output_dir = system_dir / f"output-{name}"
    working_dir = system_dir / f"working-{name}"

    already_processed = output_dir.is_dir() and working_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    working_dir.mkdir(exist_ok=True)
    if not already_processed:
        move_unprocessed_contents_to_working(system_dir, working_dir, output_dir)

    cfavg_path = cfavg_output(working_dir)
    feff_dir = mode_feff_dir(working_dir)
    valid, reason = xas_is_valid(cfavg_path, feff_dir)
    if not valid:
        copy_xyz(system_dir, working_dir, output_dir, name)
        return False, [f"xas: {reason}"]

    failures: list[str] = []
    try:
        process_spectrum(cfavg_path, output_dir, name, fft=fft, show=show)
    except Exception as exc:
        failures.append(f"xas: processing error ({exc})")

    copy_if_exists(cfavg_path, output_dir / f"xas-{name}.dat", "Corvus.cfavg_xas.out")
    # Raw per-component FEFF tables, for provenance.
    for component in XAS_COMPONENTS:
        copy_if_exists(feff_dir / component / "xmu.dat", output_dir / f"xmu-{component}-{name}.dat",
                       f"xmu.dat ({component})")
    exafs_chi = feff_dir / "exafs" / "chi.dat"
    if exafs_chi.is_file():
        copy_if_exists(exafs_chi, output_dir / f"chi-exafs-{name}.dat", "chi.dat (exafs)")
    copy_xyz(system_dir, working_dir, output_dir, name)
    return not failures, failures


def copy_xyz(system_dir: Path, working_dir: Path, output_dir: Path, name: str) -> None:
    """Ship the geometry the spectrum was computed at as ``output-<id>/<id>.xyz``.

    ``<run_id>.xyz`` (ORCA's optimized geometry, or the one prepare-orca writes
    for no-ORCA modes) when present; otherwise the input copy, resolved the same
    way the CORVUS and Hessian stages resolve it.
    """
    xyz_src = next((p for p in (working_dir / f"{name}.xyz", system_dir / f"{name}.xyz") if p.is_file()), None)
    if xyz_src is None:
        for search_dir in (working_dir, system_dir):
            try:
                xyz_src = chem_xyz.select_run_xyz(search_dir, name)
                break
            except FileNotFoundError:
                continue
    if xyz_src is None:
        log.warning("warning: missing xyz: no .xyz found for %s", name)
        return
    copy_if_exists(xyz_src, output_dir / f"{name}.xyz", "xyz")


def process_batch(
    parent_dir: Path,
    *,
    recursive: bool = True,
    fft: FftWindow | None = DEFAULT_FFT,
    show: bool = False,
    record_in_batch_log: bool = True,
) -> list[str]:
    """Process every run under ``parent_dir``; write the failed-id manifest; return the failed ids."""
    parent_dir = Path(parent_dir).resolve()
    if fft is not None and importlib.util.find_spec("larch") is None:
        # Without this every good run would fail in the transform and be
        # quarantined as a CORVUS failure.
        log.warning("WARNING: xraylarch is not installed (the 'exafs' extra); skipping chi(R) for this batch.")
        fft = None
    targets = resolve_system_targets(parent_dir, recursive=recursive)
    if not targets:
        log.warning(
            "warning: no processable directories found under %s. "
            "Expected working/output dirs or Corvus3_cfavg_<mode>/Corvus1Zn_FEFF dirs.", parent_dir,
        )
        layout.write_failed_corvus_ids(parent_dir, [])
        return []

    failed: set[str] = set()
    outcomes: list[tuple[str, str, str | None]] = []
    for target in targets:
        log.info("\nProcessing: %s", target)
        job_name = f"corvus-{target.name}"
        try:
            ok, reasons = process_system_dir(target, fft=fft, show=show)
        except Exception as exc:
            failed.add(target.name)
            log.error("error: failed processing %s: %s", target, exc)
            outcomes.append((job_name, "FAILED", f"error processing FEFF output: {exc}"))
            continue
        if ok:
            outcomes.append((job_name, "OK", None))
        else:
            failed.add(target.name)
            detail = "; ".join(reasons) or "no usable output"
            log.info("CORVUS FAILED: %s -> %s", target.name, detail)
            outcomes.append((job_name, "FAILED", detail))

    failed_ids = sorted(failed)
    manifest = layout.write_failed_corvus_ids(parent_dir, failed_ids)
    log.info("\nProcessed %d dir(s); %d with CORVUS failure(s).", len(targets), len(failed_ids))
    log.info("Wrote CORVUS failed-id manifest: %s", manifest)

    batch_log = find_batch_log(parent_dir) if record_in_batch_log else None
    if batch_log is not None:
        append_outcomes(batch_log, "CORVUS outcomes", outcomes)
        log.info("Appended %d CORVUS outcome(s) to %s", len(outcomes), batch_log)
    return failed_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process the combined CORVUS xas output of each run: XANES/EXAFS plots, the chi(R) FFT, "
        "and xas-<id>.dat, chi-R-<id>.dat, the per-component xmu/chi tables and the xyz in output-<id>. "
        "Failed runs are listed in corvus-failed-ids.txt for the download stage."
    )
    parser.add_argument("parent_dir", type=Path,
                        help="A run dir, or (with --recursive) a batch root to scan for run dirs")
    parser.add_argument("--recursive", action="store_true",
                        help="Process all run dirs under parent_dir when it is not itself one")
    parser.add_argument("--show", action="store_true", help="Display plots after saving")
    parser.add_argument("--skip-fft", action="store_true", help="Skip the chi(k) -> chi(R) Fourier transform")
    for f in fields(FftWindow):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default, help=f"(default: {f.default})")
    parser.add_argument("--no-batch-log", action="store_true",
                        help="Do not append CORVUS outcomes to batch-jobs.log")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    if not args.parent_dir.is_dir():
        log.error("error: directory not found: %s", args.parent_dir)
        return 2
    fft = None if args.skip_fft else FftWindow(**{f.name: getattr(args, f.name) for f in fields(FftWindow)})
    try:
        process_batch(args.parent_dir, recursive=args.recursive, fft=fft, show=args.show,
                      record_in_batch_log=not args.no_batch_log)
    except FileNotFoundError as exc:
        log.error("error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
