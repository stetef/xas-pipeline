"""xas-run-batch: prepare and submit a batch of structures for one mode.

Per structure: an ORCA job (unless the mode has none), then a CORVUS job
``afterok`` on it (the wrapper builds a spring Hessian first for spring modes,
then runs prepare-corvus and CORVUS). Per batch: one postprocess job ``afterany``
on every CORVUS job the batch has outstanding -- including earlier invocations'
modes -- replacing any earlier, now under-gated postprocess; and a watchdog
companion job that resubmits silently hung jobs.

Preparation is all-or-nothing: if any structure cannot be prepared, nothing is
submitted. Submission continues past individual failures so jobs already queued
still get their postprocess and record; the batch then reports failure.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, modes
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import Scheduler, add_scheduler_argument, check_executable, get_scheduler
from xas_pipeline.stages import orca_prep
from xas_pipeline.stages.interp_hessian import RUN_LIGAND_DIR_NAME, ligand_files_in

log = logging.getLogger(__name__)

NO_SUBMIT = "NO_SUBMIT"
NO_ORCA_STAGE = "NO_ORCA_STAGE"
# '{stem}' in a ligand directory: one database directory per structure.
LIGAND_DIR_STEM = "{stem}"


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    orca_job_id: str
    corvus_job_id: str | None


@dataclass
class BatchResult:
    batch_root: Path
    mode: str
    runs: list[RunRecord] = field(default_factory=list)
    postprocess_job_id: str | None = None
    watchdog_job_id: str | None = None
    #: Submission failures (job name -> error); the batch is incomplete if any.
    errors: dict[str, str] = field(default_factory=dict)


def ligand_files_for(ligand_dir: str, xyz_file: Path) -> list[Path]:
    """The ``.interp`` files a ligand directory names for one structure (raises if none)."""
    return ligand_files_in(Path(ligand_dir.replace(LIGAND_DIR_STEM, xyz_file.stem)).expanduser().resolve())


def install_run_ligands(run_dir: Path, ligand_files: list[Path]) -> None:
    """Copy a structure's ligand database into its run dir, replacing an earlier one.

    The Hessian stage prefers this copy over the packaged models, so every later
    wrapper regeneration (reruns, xas-submit-corvus) uses the same database.
    """
    target = run_dir / RUN_LIGAND_DIR_NAME
    target.mkdir(exist_ok=True)
    for stale in target.glob("*.interp"):
        stale.unlink()
    for path in ligand_files:
        shutil.copy2(path, target / path.name)


def submit_postprocess(
    sched: Scheduler, batch_root: Path, script: Path, corvus_job_ids: list[str], *, only_if_pending: bool = False
) -> str | None:
    """(Re)submit the batch postprocess, gated afterany on every outstanding CORVUS job.

    Earlier postprocess jobs still queued or running for this batch are gated on
    less than the batch now holds; the new one is submitted first and they are
    cancelled after, so a refused submission never leaves the batch with none. With
    *only_if_pending*, nothing happens unless one was pending (an operator who
    chose --no-postprocess keeps that choice).
    """
    log_path = batch_root / batch_log.LOG_NAME
    stale = batch_log.pending_postprocess_job_ids(log_path, sched)
    if only_if_pending and not stale:
        return None
    earlier = batch_log.outstanding_corvus_job_ids(log_path, sched, exclude=corvus_job_ids)
    if earlier:
        log.info("Postprocess also waits for %d CORVUS job(s) from earlier submissions.", len(earlier))
    job_name = f"postprocess-{batch_root.name}"
    job_id = batch_log.submit_and_log(
        sched, log_path, job_name, script, batch_root, afterany=[*corvus_job_ids, *earlier]
    )
    for old in stale:
        cancelled = sched.cancel(old)
        log.info("Replaced earlier postprocess job %s (%s)", old,
                 "cancelled" if cancelled else "cancel failed; it may run early")
        batch_log.append(log_path, job_name, "REPLACED", job_id=old)
    return job_id


def _live_watchdog(log_path: Path, sched: Scheduler, batch_name: str) -> str | None:
    live = sched.active_job_ids(batch_log.submitted_job_ids(log_path, f"watchdog-{batch_name}"))
    return live[-1] if live else None


def submit_batch(
    xyz_files: list[Path],
    batch_root: Path,
    mode: str,
    sched: Scheduler,
    *,
    ligand_dir: str | None = None,
    download_destination: Path | None = None,
    postprocess: bool = True,
    extract: bool = True,
    process_feff: bool = True,
    cleanup: bool = True,
    download: bool = True,
    watchdog: bool = True,
    watchdog_options: dict | None = None,
    submit: bool = True,
) -> BatchResult:
    """Prepare and (unless *submit* is False) submit one mode for *xyz_files*.

    Raises :class:`~xas_pipeline.stages.orca_prep.PrepError` before writing
    anything when the input cannot be prepared (missing ligand databases,
    point-charge files) and before submitting anything when a structure fails.
    """
    mode_spec = modes.get(mode)
    orca_prep.validate(xyz_files, mode)
    ligands: dict[Path, list[Path]] = {}
    if ligand_dir is not None:
        if not mode_spec.spring_hessian:
            raise orca_prep.PrepError(f"a ligand directory only applies to spring modes, not {mode}")
        missing = []
        for xyz in xyz_files:
            try:
                ligands[xyz] = ligand_files_for(ligand_dir, xyz)
            except (FileNotFoundError, NotADirectoryError) as exc:
                missing.append(f"  {xyz.name}: {exc}")
        if missing:
            raise orca_prep.PrepError("ligand directory has no .interp files for:\n" + "\n".join(missing))

    batch_root.mkdir(parents=True, exist_ok=True)
    log_path = batch_root / batch_log.LOG_NAME
    batch_log.init(log_path, sched)
    download_destination = download_destination or batch_root / "downloading-station"

    stages, failures = [], []
    for xyz in xyz_files:
        try:
            stages.append((xyz, orca_prep.prepare_stage(xyz, batch_root, mode, sched)))
        except orca_prep.PrepError as exc:
            failures.append(str(exc))
    if failures:
        batch_log.append(log_path, "prepare-orca", "FAILED")
        raise orca_prep.PrepError("preparation failed; nothing submitted:\n  " + "\n  ".join(failures))
    batch_log.append(log_path, "prepare-orca", "SUCCEEDED")

    result = BatchResult(batch_root=batch_root, mode=mode)
    for xyz, stage in stages:
        if xyz in ligands:
            install_run_ligands(stage.run_dir, ligands[xyz])
        wrapper = jobscripts.corvus_wrapper(sched, stage.run_dir, stage.run_id, mode=mode)
        orca_name, corvus_name = f"orca-{stage.run_id}", f"corvus-xas-{stage.run_id}"
        if not submit:
            batch_log.append(log_path, orca_name, "SKIPPED")
            batch_log.append(log_path, corvus_name, "SKIPPED")
            orca_id = NO_SUBMIT if stage.job_script else NO_ORCA_STAGE
            result.runs.append(RunRecord(stage.run_id, stage.run_dir, orca_id, NO_SUBMIT))
            continue
        try:
            if stage.job_script:
                orca_id = batch_log.submit_and_log(sched, log_path, orca_name, stage.job_script, stage.run_dir)
            else:
                # Logged, not omitted: an absent line reads as a lost submission.
                orca_id = NO_ORCA_STAGE
                batch_log.append(log_path, orca_name, "SKIPPED")
        except (RuntimeError, ValueError, OSError) as exc:
            result.errors[orca_name] = str(exc)
            continue
        try:
            corvus_id = batch_log.submit_and_log(
                sched, log_path, corvus_name, wrapper, stage.run_dir,
                afterok=[orca_id] if stage.job_script else [],
            )
        except (RuntimeError, ValueError, OSError) as exc:
            result.errors[corvus_name] = str(exc)
            corvus_id = None
        result.runs.append(RunRecord(stage.run_id, stage.run_dir, orca_id, corvus_id))

    script = jobscripts.postprocess(
        sched, batch_root, download_destination,
        extract=extract, process_feff=process_feff, cleanup=cleanup, download=download,
    )
    pp_name = f"postprocess-{batch_root.name}"
    if not postprocess:
        batch_log.append(log_path, pp_name, "SKIPPED")
    elif not submit:
        result.postprocess_job_id = NO_SUBMIT
        batch_log.append(log_path, pp_name, "SKIPPED")
    else:
        corvus_ids = [r.corvus_job_id for r in result.runs if r.corvus_job_id]
        try:
            result.postprocess_job_id = submit_postprocess(sched, batch_root, script, corvus_ids)
        except (RuntimeError, ValueError, OSError) as exc:
            result.errors[pp_name] = str(exc)

    if watchdog and submit:
        live = _live_watchdog(log_path, sched, batch_root.name)
        if live:
            log.info("Watchdog job %s is already watching this batch; not starting another.", live)
            result.watchdog_job_id = live
        else:
            wd_script = jobscripts.watchdog(sched, batch_root, **(watchdog_options or {}))
            try:
                result.watchdog_job_id = batch_log.submit_and_log(
                    sched, log_path, f"watchdog-{batch_root.name}", wd_script, batch_root
                )
            except (RuntimeError, ValueError, OSError) as exc:
                # Best-effort: the batch runs without hang recovery.
                log.warning("Watchdog failed to submit (%s); hangs will not be auto-recovered.", exc)
    return result


def render_state(result: BatchResult, *, sched: Scheduler, input_path: Path, download_destination: Path) -> str:
    """The plain-text submission record (pipeline-state-<batch>.log)."""
    lines = [
        "CSD Zn-complex pipeline - batch submission state",
        "=" * 48,
        f"created_utc:          {batch_log.utc_now()}",
        f"input_path:           {input_path}",
        f"output_root:          {result.batch_root}",
        f"scheduler:            {sched.name}",
        f"optimization_mode:    {result.mode}",
        "corvus_mode:          xas",
        f"download_destination: {download_destination}",
        f"postprocess_job_id:   {result.postprocess_job_id}",
        f"watchdog_job_id:      {result.watchdog_job_id}",
        "",
        f"Runs ({len(result.runs)}):",
        "",
    ]
    for run in result.runs:
        lines += [
            f"  {run.run_id}",
            f"    run_dir:        {run.run_dir}",
            f"    orca_job_id:    {run.orca_job_id}",
            f"    corvus_job_id:  {run.corvus_job_id}",
            "",
        ]
    for name, error in result.errors.items():
        lines += [f"SUBMISSION FAILED: {name}", *(f"    {line}" for line in error.splitlines()), ""]
    lines += [
        "Statuses here are SUBMISSION only. Per-run outcomes: batch-jobs.log,",
        "orca-convergence-report.log and corvus-failed-ids.txt.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and submit ORCA -> CORVUS jobs for a batch, plus its postprocess and watchdog."
    )
    parser.add_argument("path", type=Path, help="An XYZ file or a directory of them")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Batch root (default: parent of the XYZ directory)")
    add_scheduler_argument(parser)
    modes.add_mode_arguments(parser)
    parser.add_argument(
        "--ligand-dir", default=None, metavar="DIR",
        help="Spring modes: interpolate from every .interp file in DIR instead of the packaged "
             "models. '{stem}' becomes each structure's XYZ stem (one directory per structure). "
             f"The files are copied into each run dir's {RUN_LIGAND_DIR_NAME}/, so reruns use them too.",
    )
    parser.add_argument("--download-destination", type=Path, default=None,
                        help="Where the postprocess stages deliverables (default: <batch>/downloading-station)")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    skip = parser.add_argument_group("postprocess steps")
    skip.add_argument("--skip-extract", action="store_true", help="Skip the ORCA convergence/timing check")
    skip.add_argument("--skip-process-feff", action="store_true",
                      help="Skip FEFF post-processing (and the CORVUS auto-rerun triage)")
    skip.add_argument("--skip-cleanup", action="store_true", help="Skip pruning FEFF scratch and superseded spectra")
    skip.add_argument("--skip-prepare-download", action="store_true", help="Skip staging deliverables")
    skip.add_argument("--no-postprocess", action="store_true",
                      help="Submit no postprocess job (run xas-postprocess yourself later)")
    wd = parser.add_argument_group("watchdog")
    wd.add_argument("--no-watchdog", dest="watchdog", action="store_false",
                    help="Do not submit the hang-watchdog companion job")
    wd.add_argument("--watchdog-interval", type=int, default=jobscripts.WATCHDOG_INTERVAL,
                    help="Poll interval in seconds (default: %(default)s)")
    wd.add_argument("--watchdog-idle-threshold", type=int, default=jobscripts.WATCHDOG_IDLE_THRESHOLD,
                    help="Seconds of silent log + flat CPU before a job counts as hung (default: %(default)s)")
    wd.add_argument("--watchdog-max-reruns", type=int, default=jobscripts.WATCHDOG_MAX_RERUNS,
                    help="Hang resubmits per structure before escalating (default: %(default)s)")
    wd.add_argument("--watchdog-walltime", default=jobscripts.WATCHDOG_WALLTIME,
                    help="Walltime of the watchdog job (default: %(default)s)")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Where to write the submission record (default: <batch>/pipeline-state-<batch>.log)")
    parser.add_argument("--no-submit", "--dry-run", dest="no_submit", action="store_true",
                        help="Write everything, submit nothing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    sched = get_scheduler(args.scheduler)
    if not args.no_submit:
        check_executable(sched.submit_command)
    try:
        xyz_files, input_dir = orca_prep.discover_xyz_files(args.path)
        batch_root = (args.out_dir.expanduser() if args.out_dir else input_dir.parent).resolve()
        download_destination = (
            args.download_destination.expanduser().resolve() if args.download_destination
            else batch_root / "downloading-station"
        )
        result = submit_batch(
            xyz_files, batch_root, args.mode, sched,
            ligand_dir=args.ligand_dir,
            download_destination=download_destination,
            postprocess=not args.no_postprocess,
            extract=not args.skip_extract,
            process_feff=not args.skip_process_feff,
            cleanup=not args.skip_cleanup,
            download=not args.skip_prepare_download,
            watchdog=args.watchdog,
            watchdog_options={
                "interval": args.watchdog_interval,
                "idle_threshold": args.watchdog_idle_threshold,
                "max_reruns": args.watchdog_max_reruns,
                "walltime": args.watchdog_walltime,
            },
            submit=not args.no_submit,
        )
    except orca_prep.PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    state_file = (args.state_file.expanduser().resolve() if args.state_file
                  else batch_root / f"pipeline-state-{batch_root.name}.log")
    state_file.write_text(
        render_state(result, sched=sched, input_path=args.path.expanduser().resolve(),
                     download_destination=download_destination),
        encoding="utf-8",
    )
    print(f"Prepared runs: {len(result.runs)}")
    for run in result.runs:
        print(f"  {run.run_id}: ORCA={run.orca_job_id}, CORVUS=[{run.corvus_job_id}]")
    if args.no_postprocess:
        print(f"Postprocess job: not submitted. When the batch is done: xas-postprocess {batch_root} --submit")
    else:
        print(f"Postprocess job: {result.postprocess_job_id}")
    if result.watchdog_job_id:
        print(f"Watchdog job: {result.watchdog_job_id}")
    print(f"State file: {state_file}")
    print(f"Batch job log: {batch_root / batch_log.LOG_NAME}")
    for name, error in result.errors.items():
        print(f"ERROR: {name} was not submitted: {error.splitlines()[0]}", file=sys.stderr)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
