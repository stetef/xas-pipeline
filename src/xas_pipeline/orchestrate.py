#!/usr/bin/env python3
"""Submit an end-to-end ORCA -> CORVUS -> postprocess workflow for a batch.

Workflow overview (per structure ID):
1) Prepare ORCA inputs/scripts with prepare-orca.py (always called with --dry-run)
2) Submit ORCA job
3) Submit dependent CORVUS job (afterok on ORCA) that:
    - runs prepare-corvus.py inside the ORCA run directory with mode-specific templates
   - fails fast if <ID>.hess is missing (prepare-corvus enforces this)
    - executes generated mode-specific corvus-job-<mode>.script inline in the same allocated job

Batch-level postprocess:
4) Submit one dependent postprocess job (afterany on *all* CORVUS jobs, so it runs
   once every job has finished regardless of success/failure) that runs:
   - script-check-orca-convergence-and-extract-times.py (moves failed ORCA runs to failed-orca/)
   - script-process-feff-output.py --recursive
   - script-prepare-files-for-download.py (moves failed CORVUS runs to failed-corvus/)
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from xas_pipeline import scheduler as _sched
from xas_pipeline import layout, templates, resources
from xas_pipeline.stages.orca_prep import INTERP_HESSIAN_MODES

# Scheduler slurm/pbs differences live in xas_pipeline.scheduler now. These names
# are kept as thin bindings so the (transitional) importlib consumers
# (rerun-corvus, submit-corvus-only) and the CLI keep working unchanged.
SCHEDULER_SUBMIT_COMMAND = _sched.SUBMIT_COMMAND
SCHEDULER_CANCEL_COMMAND = _sched.CANCEL_COMMAND
SCHEDULER_TEMPLATE_DIR = _sched.TEMPLATE_DIR

# How each corvus-wrapper template invokes Python, so commands injected into it
# (the [INTERP_HESS_CMD] step) resolve the same interpreter the wrapper's own
# prepare-corvus call does. The slurm wrapper resolves $PYTHON_BIN explicitly;
# the PBS one relies on `-V` inheriting the submit-time venv on PATH.
WRAPPER_PYTHON = {"slurm": '"$PYTHON_BIN"', "pbs": "python"}


@dataclass
class JobRecord:
    run_id: str
    run_dir: str
    orca_script: str
    orca_job_id: str
    orca_submitted_utc: str
    corvus_wrapper_scripts: list[str]
    corvus_job_ids: list[str]
    corvus_submitted_utc: str


@dataclass
class BatchState:
    created_utc: str
    input_path: str
    output_root: str
    scheduler: str
    download_destination: str
    h_only: bool
    optimization_mode: str
    corvus_mode: str
    postprocess_job_id: str | None
    runs: list[JobRecord]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_check_executable = _sched.check_executable


def _dependency_debug_command(job_id: str, scheduler: str) -> str:
    return _sched.get_scheduler(scheduler).debug_command(job_id)


def _append_batch_job_log(
    log_path: Path,
    job_name: str,
    status: str,
    job_id: str | None = None,
) -> None:
    # The per-job dep_debug command is identical apart from the job id, so it lives
    # once in the header (see _initialize_batch_log) rather than being repeated on
    # every line; here we record only job_name, status, and job_id. `status` is the
    # SUBMISSION result (SUBMITTED / SUBMIT_FAILED / SKIPPED), not the computational
    # outcome — the postprocess stage appends authoritative OK/FAILED outcomes.
    if job_id is None:
        line = f"{job_name}\t{status}\n"
    else:
        line = f"{job_name}\t{status}\tjob_id={job_id}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _initialize_batch_log(log_path: Path, scheduler: str) -> None:
    if not log_path.exists():
        example_dep_debug = _dependency_debug_command("<JOB_ID>", scheduler)
        status_note = (
            "#\n"
            "# The 'status' column is the SUBMISSION result (SUBMITTED = the scheduler\n"
            "# accepted the job), NOT the computational outcome. Authoritative per-run\n"
            "# outcomes are appended below by the postprocess stage under\n"
            "# '# --- ORCA outcomes ... ---' / '# --- CORVUS outcomes ... ---', and are\n"
            "# detailed in orca-convergence-report.log and corvus-failed-ids.txt.\n"
        )
        if scheduler == "pbs":
            header = (
                "# Helpful PBS debug commands (replace <JOB_ID>)\n"
                f"# dependency + exit-status check (dep_debug): {example_dep_debug}\n"
                "# full tracejob history:\n"
                "#   tracejob -n 200 <JOB_ID>\n"
                "# full scheduler history (if enabled):\n"
                "#   qstat -x -f <JOB_ID>\n"
                + status_note
            )
        else:
            header = (
                "# Helpful Slurm debug commands (replace <JOB_ID>)\n"
                f"# dependency + state check (dep_debug): {example_dep_debug}\n"
                "# queue status:\n"
                "#   squeue -j <JOB_ID>\n"
                "# memory high-water mark (to confirm an OOM kill):\n"
                "#   sacct -j <JOB_ID> --format=JobID,State,ExitCode,MaxRSS,ReqMem -n\n"
                + status_note
            )
        log_path.write_text(
            header + "\njob_name\tstatus\tjob_id\n",
            encoding="utf-8",
        )
        return

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"# --- invocation {_utc_now_iso()} ---\n")


def _indent_block(text: str, prefix: str = "    ") -> str:
    """Indent a (possibly multi-line) block for the plain-text state log."""
    text = (text or "").rstrip("\n")
    if not text:
        return f"{prefix}(none)"
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _render_state_text(
    state: BatchState,
    prepare_orca_stdout: str,
    prepare_orca_stderr: str,
) -> str:
    """Render the batch submission state as a human-readable plain-text log.

    This is a submission record, not an outcome record: job ids reflect what was
    submitted. Authoritative per-run pass/fail lives in batch-jobs.log (and
    orca-convergence-report.log / corvus-failed-ids.txt).
    """
    lines = [
        "CSD Zn-complex pipeline - batch submission state",
        "=" * 48,
        f"created_utc:          {state.created_utc}",
        f"input_path:           {state.input_path}",
        f"output_root:          {state.output_root}",
        f"scheduler:            {state.scheduler}",
        f"optimization_mode:    {state.optimization_mode}",
        f"corvus_mode:          {state.corvus_mode}",
        f"h_only:               {state.h_only}",
        f"download_destination: {state.download_destination}",
        f"postprocess_job_id:   {state.postprocess_job_id}",
        "",
        f"Runs ({len(state.runs)}):",
        "",
    ]
    for rec in state.runs:
        corvus_ids = ", ".join(rec.corvus_job_ids) if rec.corvus_job_ids else "(none)"
        lines.extend(
            [
                f"  {rec.run_id}",
                f"    run_dir:        {rec.run_dir}",
                f"    orca_script:    {rec.orca_script}",
                f"    orca_job_id:    {rec.orca_job_id}   (submitted {rec.orca_submitted_utc})",
                f"    corvus_job_ids: {corvus_ids}   (submitted {rec.corvus_submitted_utc})",
                "",
            ]
        )

    lines.extend(
        [
            "prepare-orca stdout:",
            _indent_block(prepare_orca_stdout),
            "",
            "prepare-orca stderr:",
            _indent_block(prepare_orca_stderr),
            "",
            "Note: statuses here reflect SUBMISSION only. See batch-jobs.log for the",
            "authoritative per-run outcomes, and orca-convergence-report.log /",
            "corvus-failed-ids.txt for the detailed reasons.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


_parse_submitted_job_id = _sched.parse_job_id


def _discover_xyz_files(path_arg: Path) -> tuple[list[Path], Path]:
    if not path_arg.exists():
        raise FileNotFoundError(f"Path not found: {path_arg}")

    if path_arg.is_file():
        if path_arg.suffix.lower() != ".xyz":
            raise ValueError(f"Expected an XYZ file, got: {path_arg}")
        return [path_arg.resolve()], path_arg.resolve().parent

    xyz_files = sorted(path_arg.glob("*.xyz"))
    if not xyz_files:
        raise ValueError(f"No .xyz files found in directory: {path_arg}")
    return [p.resolve() for p in xyz_files], path_arg.resolve()


def _run_id_from_xyz(xyz_file: Path, optimization_mode: str) -> str:
    return layout.run_id_for(xyz_file.stem, optimization_mode)


def _write_corvus_wrapper_script(
    script_path: Path,
    run_dir: Path,
    run_id: str,
    scheduler: str,
    corvus_mode: str = "xas",
    optimization_mode: str | None = None,
) -> None:
    template_path = resources.template_root() / SCHEDULER_TEMPLATE_DIR[scheduler] / "corvus-wrapper.script"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template: {template_path}")

    # Callers that know the mode (the orchestrator) pass it; the rerun/submit CLIs
    # work from an existing run dir, so recover it from the run id. A run dir
    # predating mode suffixes yields None -> treated as "ORCA wrote the Hessian",
    # which is right for every batch that existed before --interp.
    if optimization_mode is None:
        optimization_mode = layout.mode_from_run_id(run_id) or "unknown"

    # For modes whose ORCA input omits "! AnFreq" (currently --interp), ORCA
    # writes no .hess, so the wrapper interpolates one from the ligand spring
    # models first. Every other mode gets `true`, i.e. the ORCA Hessian is used
    # exactly as before.
    if optimization_mode in INTERP_HESSIAN_MODES:
        interp_hess_cmd = (
            f"{WRAPPER_PYTHON[scheduler]} -m xas_pipeline.stages.interp_hessian "
            '"$RUN_DIR" --run-id "$RUN_ID"'
        )
    else:
        interp_hess_cmd = "true"

    # The wrapper runs `python -m xas_pipeline.stages.corvus_prep`; PIPELINE_ROOT
    # anchors .venv/.env discovery on the compute node (formerly derived from the
    # injected prepare-corvus.py path).
    templates.render(
        template_path,
        script_path,
        {
            "RUN_DIR": run_dir,
            "RUN_ID": run_id,
            "PIPELINE_ROOT": resources.project_root(),
            "SCHEDULER": scheduler,
            "CORVUS_MODE": corvus_mode,
            "PIPELINE_ENV": resources.project_root() / ".env",
            "INTERP_HESS_CMD": interp_hess_cmd,
            "OPTIMIZATION_MODE": optimization_mode,
        },
        executable=True,
        ensure_trailing_newline=True,
    )


def _write_postprocess_script(
    script_path: Path,
    scheduler: str,
    output_root: Path,
    download_destination: Path,
    skip_extract: bool,
    skip_process_feff: bool,
    skip_prepare_download: bool,
    prepare_download_refresh: bool = False,
    skip_cleanup: bool = False,
) -> None:
    template_path = resources.template_root() / SCHEDULER_TEMPLATE_DIR[scheduler] / "postprocess-job.script"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template: {template_path}")

    # Post-processing stages run as `python -m xas_pipeline.stages.<stage>`.
    extract_cmd = (
        f"python -m xas_pipeline.stages.orca_check \"{output_root}\" --output-dir \"{output_root}\""
        if not skip_extract
        else "true"
    )
    process_feff_cmd = (
        f"python -m xas_pipeline.stages.feff_process \"{output_root}\" --recursive"
        if not skip_process_feff
        else "true"
    )
    # Runs after feff_process (deliverables captured in output-<id>) but before
    # download, so the clean-replace refresh mirrors the pruned output dirs and
    # never re-copies stale scratch/component files into the station.
    cleanup_cmd = (
        f"python -m xas_pipeline.stages.cleanup \"{output_root}\" --execute"
        if not skip_cleanup
        else "true"
    )
    refresh_flag = " --refresh" if prepare_download_refresh else ""
    prepare_download_cmd = (
        f"python -m xas_pipeline.stages.download \"{output_root}\" -d \"{download_destination}\"{refresh_flag}"
        if not skip_prepare_download
        else "true"
    )

    templates.render(
        template_path,
        script_path,
        {
            "BATCH_NAME": output_root.name,
            "OUTPUT_ROOT": output_root,
            "EXTRACT_CMD": extract_cmd,
            "PROCESS_FEFF_CMD": process_feff_cmd,
            "CLEANUP_CMD": cleanup_cmd,
            "PREPARE_DOWNLOAD_CMD": prepare_download_cmd,
        },
        executable=True,
        ensure_trailing_newline=True,
    )


def _submit_job(
    script_path: Path,
    cwd: Path,
    scheduler: str,
    depend_afterok: Iterable[str] | None = None,
    depend_afterany: Iterable[str] | None = None,
) -> str:
    sched = _sched.get_scheduler(scheduler)
    submit_command = sched.submit_command
    # afterok: dependent runs only if all parents succeed.
    # afterany: dependent runs once all parents finish, regardless of exit status.
    dep_flag: list[str] = []
    if depend_afterok:
        dep_flag = sched.dependency_flag("afterok", [str(j) for j in depend_afterok])
    elif depend_afterany:
        dep_flag = sched.dependency_flag("afterany", [str(j) for j in depend_afterany])
    cmd = [submit_command, *dep_flag, script_path.name]

    result = _run_command(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{submit_command} failed for {script_path} (cwd={cwd})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return _parse_submitted_job_id(result.stdout)


def _cancel_job(job_id: str, scheduler: str) -> bool:
    """Best-effort cancel of a scheduled job (scancel/qdel). Never raises.

    Returns True if the cancel command ran and exited 0. Cancelling an already
    terminated/unknown job is treated as success-ish (nothing to clean up) but
    reported via the return code the scheduler gives.
    """
    cancel_command = _sched.CANCEL_COMMAND[scheduler]
    try:
        result = subprocess.run(
            [cancel_command, str(job_id)],
            capture_output=True, text=True, check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


_default_scheduler = _sched.default_scheduler_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit ORCA->CORVUS pipeline with scheduler dependencies, then submit a final "
            "batch postprocess job after all CORVUS jobs succeed."
        )
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(SCHEDULER_SUBMIT_COMMAND),
        default=_default_scheduler(),
        help="Scheduler backend used for templates and submission command.",
    )
    parser.add_argument("path", type=Path, help="XYZ directory or single XYZ file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        help=(
            "Batch output directory where per-ID run dirs are created "
            "(default: parent of input XYZ directory)"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--H",
        action="store_true",
        help="Use H-only ORCA template (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--single",
        action="store_true",
        help="Use single-point ORCA template (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--free",
        action="store_true",
        help="Use no-constraints ORCA template (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--backbone",
        action="store_true",
        help="Use backbone point-charge ORCA template (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--xtb-free",
        action="store_true",
        help="Use XTB-free ORCA template mode (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--xtb-constrained",
        action="store_true",
        help="Use XTB-constrained ORCA template mode (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Use quick ORCA template, no CA fixing (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--quick-ca-fixed",
        action="store_true",
        help="Use quick CA-fixed ORCA template (propagates to prepare-orca)",
    )
    mode_group.add_argument(
        "--interp",
        action="store_true",
        help=(
            "Use the interp ORCA template: a single point for the energy with no "
            "AnFreq. The Hessian is interpolated from the packaged ligand spring "
            "models before CORVUS instead of being computed by ORCA."
        ),
    )
    parser.add_argument(
        "--download-destination",
        type=Path,
        default=None,
        help=(
            "Destination for script-prepare-files-for-download.py "
            "(default: <output_root>/downloading-station)"
        ),
    )
    parser.add_argument(
        "--corvus-mode",
        choices=["xas"],
        default="xas",
        help=(
            "Corvus target to run. Only the combined 'xas' target is supported: a single "
            "CORVUS run reads xanes.in and exafs.in and produces Corvus.cfavg_xas.out."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip script-check-orca-convergence-and-extract-times.py in final postprocess job",
    )
    parser.add_argument(
        "--skip-process-feff",
        action="store_true",
        help="Skip script-process-feff-output.py in final postprocess job",
    )
    parser.add_argument(
        "--skip-prepare-download",
        action="store_true",
        help="Skip script-prepare-files-for-download.py in final postprocess job",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip the automatic xas-cleanup pass (FEFF scratch + superseded "
             "xanes/exafs) in the final postprocess job",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional explicit path for the plain-text pipeline state log",
    )
    parser.add_argument(
        "--no-submit",
        "--dry-run",
        dest="no_submit",
        action="store_true",
        help="Generate scripts and state file only; do not submit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    optimization_mode = "ca-fixed"
    if args.H:
        optimization_mode = "h-only"
    elif args.single:
        optimization_mode = "single-point"
    elif args.free:
        optimization_mode = "no-constraints"
    elif args.backbone:
        optimization_mode = "backbone"
    elif args.xtb_free:
        optimization_mode = "xtb-free"
    elif args.xtb_constrained:
        optimization_mode = "xtb-constrained"
    elif args.quick:
        optimization_mode = "quick"
    elif args.quick_ca_fixed:
        optimization_mode = "quick-ca-fixed"
    elif args.interp:
        optimization_mode = "interp"

    if not args.skip_process_feff:
        # Keep explicit: script-process-feff-output imports numpy/matplotlib/larch at runtime.
        # This pre-check catches missing python early on head/login nodes.
        _check_executable("python")

    submit_command = SCHEDULER_SUBMIT_COMMAND[args.scheduler]
    if not args.no_submit:
        _check_executable(submit_command)

    xyz_files, input_base_dir = _discover_xyz_files(args.path.expanduser())

    if args.out_dir is None:
        output_root = input_base_dir.parent.resolve()
        print(f"No --out-dir provided; defaulting to: {output_root}")
    else:
        output_root = args.out_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_log = output_root / "batch-jobs.log"
    _initialize_batch_log(batch_log, args.scheduler)

    # Default the download destination into the batch output root so the
    # downloading-station lands next to the run dirs (the postprocess job's cwd),
    # not wherever this submit script was launched from.
    if args.download_destination is None:
        download_destination = output_root / "downloading-station"
    else:
        download_destination = args.download_destination.expanduser().resolve()

    prepare_cmd = [
        "python",
        "-m",
        "xas_pipeline.stages.orca_prep",
        str(args.path.expanduser()),
        "--out-dir",
        str(output_root),
        "--dry-run",
        "--scheduler",
        args.scheduler,
    ]
    if args.H:
        prepare_cmd.append("--H")
    elif args.single:
        prepare_cmd.append("--single")
    elif args.free:
        prepare_cmd.append("--free")
    elif args.backbone:
        prepare_cmd.append("--backbone")
    elif args.xtb_free:
        prepare_cmd.append("--xtb-free")
    elif args.xtb_constrained:
        prepare_cmd.append("--xtb-constrained")
    elif args.quick:
        prepare_cmd.append("--quick")
    elif args.quick_ca_fixed:
        prepare_cmd.append("--quick-ca-fixed")
    elif args.interp:
        prepare_cmd.append("--interp")

    prep_result = _run_command(prepare_cmd)
    if prep_result.returncode != 0:
        _append_batch_job_log(batch_log, "prepare-orca", "FAILED")
        raise RuntimeError(
            "prepare-orca.py failed:\n"
            f"stdout:\n{prep_result.stdout}\n"
            f"stderr:\n{prep_result.stderr}"
        )
    _append_batch_job_log(batch_log, "prepare-orca", "SUCCEEDED")

    records: list[JobRecord] = []
    for xyz in xyz_files:
        run_id = _run_id_from_xyz(xyz, optimization_mode=optimization_mode)
        run_dir = layout.run_dir_for(output_root, xyz.stem, optimization_mode)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Expected run directory not found: {run_dir}")

        orca_script = run_dir / f"generated-{run_id}-orca.script"
        if not orca_script.is_file():
            matches = sorted(run_dir.glob("generated-*-orca.script"))
            if len(matches) == 1:
                orca_script = matches[0]
            else:
                raise FileNotFoundError(
                    f"Could not locate generated ORCA job script in {run_dir}"
                )

        corvus_submitted_utc = _utc_now_iso()
        corvus_modes_to_submit = [args.corvus_mode]
        corvus_wrappers = []
        corvus_job_ids: list[str] = []

        for cmode in corvus_modes_to_submit:
            corvus_wrapper = run_dir / f"generated-{run_id}-corvus-{cmode}-wrapper.script"
            _write_corvus_wrapper_script(
                corvus_wrapper,
                run_dir,
                run_id,
                args.scheduler,
                corvus_mode=cmode,
                optimization_mode=optimization_mode,
            )
            corvus_wrappers.append(corvus_wrapper)

        if args.no_submit:
            orca_job_id = "NO_SUBMIT"
            orca_submitted_utc = _utc_now_iso()
            corvus_submitted_utc = _utc_now_iso()
            corvus_job_ids = ["NO_SUBMIT"] * len(corvus_modes_to_submit)
            _append_batch_job_log(batch_log, f"orca-{run_id}", "SKIPPED")
            for cmode in corvus_modes_to_submit:
                _append_batch_job_log(batch_log, f"corvus-{cmode}-{run_id}", "SKIPPED")
        else:
            orca_submitted_utc = _utc_now_iso()
            try:
                orca_job_id = _submit_job(orca_script, cwd=run_dir, scheduler=args.scheduler)
                _append_batch_job_log(
                    batch_log,
                    f"orca-{run_id}",
                    "SUBMITTED",
                    job_id=orca_job_id,
                )
            except Exception:
                _append_batch_job_log(batch_log, f"orca-{run_id}", "SUBMIT_FAILED")
                raise
            corvus_submitted_utc = _utc_now_iso()
            for cmode, corvus_wrapper in zip(corvus_modes_to_submit, corvus_wrappers):
                try:
                    cjid = _submit_job(
                        corvus_wrapper,
                        cwd=run_dir,
                        scheduler=args.scheduler,
                        depend_afterok=[orca_job_id],
                    )
                    corvus_job_ids.append(cjid)
                    _append_batch_job_log(
                        batch_log,
                        f"corvus-{cmode}-{run_id}",
                        "SUBMITTED",
                        job_id=cjid,
                    )
                except Exception:
                    _append_batch_job_log(batch_log, f"corvus-{cmode}-{run_id}", "SUBMIT_FAILED")
                    raise

        records.append(
            JobRecord(
                run_id=run_id,
                run_dir=str(run_dir),
                orca_script=str(orca_script),
                orca_job_id=orca_job_id,
                orca_submitted_utc=orca_submitted_utc,
                corvus_wrapper_scripts=[str(w) for w in corvus_wrappers],
                corvus_job_ids=corvus_job_ids,
                corvus_submitted_utc=corvus_submitted_utc,
            )
        )

    postprocess_script = output_root / f"generated-postprocess-{output_root.name}.script"
    _write_postprocess_script(
        postprocess_script,
        args.scheduler,
        output_root,
        download_destination,
        skip_extract=args.skip_extract,
        skip_process_feff=args.skip_process_feff,
        skip_prepare_download=args.skip_prepare_download,
        skip_cleanup=args.skip_cleanup,
    )

    postprocess_job_id: str | None
    if args.no_submit:
        postprocess_job_id = "NO_SUBMIT"
        _append_batch_job_log(
            batch_log,
            f"postprocess-{output_root.name}",
            "SKIPPED",
        )
    else:
        corvus_ids = [jid for rec in records for jid in rec.corvus_job_ids]
        try:
            # afterany (not afterok): the postprocess scripts now handle failed ORCA
            # and CORVUS runs themselves (failed-orca/ and failed-corvus/), so the job
            # must run once every CORVUS job has finished regardless of exit status.
            postprocess_job_id = _submit_job(
                postprocess_script,
                cwd=output_root,
                scheduler=args.scheduler,
                depend_afterany=corvus_ids,
            )
            _append_batch_job_log(
                batch_log,
                f"postprocess-{output_root.name}",
                "SUBMITTED",
                job_id=postprocess_job_id,
            )
        except Exception:
            _append_batch_job_log(
                batch_log,
                f"postprocess-{output_root.name}",
                "SUBMIT_FAILED",
            )
            raise

    state_file = (
        args.state_file.expanduser().resolve()
        if args.state_file is not None
        else output_root / f"pipeline-state-{output_root.name}.log"
    )

    state = BatchState(
        created_utc=_utc_now_iso(),
        input_path=str(args.path.expanduser().resolve()),
        output_root=str(output_root),
        scheduler=args.scheduler,
        download_destination=str(download_destination),
        h_only=optimization_mode == "h-only",
        optimization_mode=optimization_mode,
        corvus_mode=args.corvus_mode,
        postprocess_job_id=postprocess_job_id,
        runs=records,
    )
    state_file.write_text(
        _render_state_text(state, prep_result.stdout, prep_result.stderr),
        encoding="utf-8",
    )

    print(f"Prepared runs: {len(records)}")
    for rec in records:
        corvus_str = ", ".join(rec.corvus_job_ids)
        print(
            f"  {rec.run_id}: ORCA={rec.orca_job_id}, CORVUS=[{corvus_str}]"
        )
    print(f"Postprocess job: {postprocess_job_id}")
    print(f"State file: {state_file}")
    print(f"Batch job log: {batch_log}")
    return 0

if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
