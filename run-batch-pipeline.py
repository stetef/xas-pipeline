#!/usr/bin/env python3
"""Submit an end-to-end ORCA -> CORVUS -> postprocess workflow for a batch.

Workflow overview (per structure ID):
1) Prepare ORCA inputs/scripts with prepare-orca.py (always called with --dry-run)
2) Submit ORCA job
3) Submit dependent CORVUS job (afterok on ORCA) that:
   - runs prepare-corvus.py inside the ORCA run directory
   - fails fast if <ID>.hess is missing (prepare-corvus enforces this)
    - executes generated corvus-job.script inline in the same allocated job

Batch-level postprocess:
4) Submit one dependent postprocess job (afterok on *all* CORVUS jobs) that runs:
   - script-extract-orca-compute-times.py
   - script-process-feff-output.py --recursive
   - script-prepare-files-for-download.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


JOB_ID_RE = re.compile(r"(?P<id>\d+)(?:\.[^\s]+)?")
SCHEDULER_SUBMIT_COMMAND = {
    "pbs": "qsub",
    "slurm": "sbatch",
}
SCHEDULER_TEMPLATE_DIR = {
    "pbs": "pbs-scripts",
    "slurm": "slurm-scripts",
}


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


def _check_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def _dependency_debug_command(job_id: str, scheduler: str) -> str:
    if scheduler == "pbs":
        return (
            f"tracejob -n 100 {job_id} 2>&1 | "
            "grep -Ei 'deleted as result of dependency|Dependency on job|Exit_status|Obit'"
        )
    return (
        f"scontrol show job {job_id} && "
        f"sacct -j {job_id} --format=JobID,State,ExitCode -n"
    )


def _append_batch_job_log(
    log_path: Path,
    scheduler: str,
    job_name: str,
    status: str,
    job_id: str | None = None,
) -> None:
    if job_id is None:
        line = f"{job_name}\t{status}\n"
    else:
        dep_cmd = _dependency_debug_command(job_id, scheduler)
        line = f"{job_name}\t{status}\tjob_id={job_id}\tdep_debug=\"{dep_cmd}\"\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _initialize_batch_log(log_path: Path, scheduler: str) -> None:
    if not log_path.exists():
        if scheduler == "pbs":
            header = (
                "# Helpful PBS debug commands (replace <JOB_ID>)\n"
                "# dependency-deletion check:\n"
                "#   tracejob -n 100 <JOB_ID> 2>&1 | grep -Ei 'deleted as result of dependency|Dependency on job|Exit_status|Obit'\n"
                "# full tracejob history:\n"
                "#   tracejob -n 200 <JOB_ID>\n"
                "# full scheduler history (if enabled):\n"
                "#   qstat -x -f <JOB_ID>\n"
            )
        else:
            header = (
                "# Helpful Slurm debug commands (replace <JOB_ID>)\n"
                "# dependency and state check:\n"
                "#   scontrol show job <JOB_ID>\n"
                "# accounting summary:\n"
                "#   sacct -j <JOB_ID> --format=JobID,State,ExitCode -n\n"
                "# queue status:\n"
                "#   squeue -j <JOB_ID>\n"
            )
        log_path.write_text(
            header + "\njob_name\tstatus\tjob_id\tdep_debug\n",
            encoding="utf-8",
        )
        return

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"# --- invocation {_utc_now_iso()} ---\n")


def _run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _parse_submitted_job_id(stdout_text: str) -> str:
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept both "12345.server" (PBS) and "Submitted batch job 12345" (Slurm).
        match = JOB_ID_RE.search(line)
        if match:
            return match.group("id")
    raise ValueError(f"Unable to parse scheduler job id from output: {stdout_text!r}")


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
    base = xyz_file.stem
    return f"{base}-H-only" if optimization_mode == "h-only" else base


def _write_corvus_wrapper_script(
    script_path: Path,
    run_dir: Path,
    run_id: str,
    prepare_corvus_py: Path,
    scheduler: str,
    corvus_mode: str = "both",
) -> None:
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / SCHEDULER_TEMPLATE_DIR[scheduler] / "corvus-wrapper.script"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template: {template_path}")

    script = template_path.read_text(encoding="utf-8")
    script = script.replace("[RUN_DIR]", str(run_dir))
    script = script.replace("[RUN_ID]", run_id)
    script = script.replace("[PREP_CORVUS]", str(prepare_corvus_py))
    script = script.replace("[SCHEDULER]", scheduler)
    script = script.replace("[CORVUS_MODE]", corvus_mode)

    script_path.write_text(script if script.endswith("\n") else script + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_postprocess_script(
    script_path: Path,
    script_dir: Path,
    scheduler: str,
    output_root: Path,
    download_destination: Path,
    skip_extract: bool,
    skip_process_feff: bool,
    skip_prepare_download: bool,
) -> None:
    extract_py = script_dir / "script-extract-orca-compute-times.py"
    process_feff_py = script_dir / "script-process-feff-output.py"
    prepare_download_py = script_dir / "script-prepare-files-for-download.py"

    template_path = script_dir / SCHEDULER_TEMPLATE_DIR[scheduler] / "postprocess-job.script"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template: {template_path}")

    extract_cmd = (
        f"python \"{extract_py}\" \"{output_root}\" --output-dir \"{output_root}\""
        if not skip_extract
        else "true"
    )
    process_feff_cmd = (
        f"python \"{process_feff_py}\" \"{output_root}\" --recursive"
        if not skip_process_feff
        else "true"
    )
    prepare_download_cmd = (
        f"python \"{prepare_download_py}\" \"{output_root}\" -d \"{download_destination}\""
        if not skip_prepare_download
        else "true"
    )

    script = template_path.read_text(encoding="utf-8")
    script = script.replace("[BATCH_NAME]", output_root.name)
    script = script.replace("[OUTPUT_ROOT]", str(output_root))
    script = script.replace("[EXTRACT_CMD]", extract_cmd)
    script = script.replace("[PROCESS_FEFF_CMD]", process_feff_cmd)
    script = script.replace("[PREPARE_DOWNLOAD_CMD]", prepare_download_cmd)

    script_path.write_text(script if script.endswith("\n") else script + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _submit_job(
    script_path: Path,
    cwd: Path,
    scheduler: str,
    depend_afterok: Iterable[str] | None = None,
) -> str:
    submit_command = SCHEDULER_SUBMIT_COMMAND[scheduler]
    cmd = [submit_command]
    if depend_afterok:
        dep_expr = "afterok:" + ":".join(str(jobid) for jobid in depend_afterok)
        if scheduler == "pbs":
            cmd.extend(["-W", f"depend={dep_expr}"])
        else:
            cmd.append(f"--dependency={dep_expr}")
    cmd.append(script_path.name)

    result = _run_command(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{submit_command} failed for {script_path} (cwd={cwd})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return _parse_submitted_job_id(result.stdout)


def _default_scheduler() -> str:
    scheduler = os.environ.get("PIPELINE_SCHEDULER", "pbs").strip().lower()
    if scheduler not in SCHEDULER_SUBMIT_COMMAND:
        supported = ", ".join(sorted(SCHEDULER_SUBMIT_COMMAND))
        raise SystemExit(
            f"Invalid PIPELINE_SCHEDULER={scheduler!r}. Supported values: {supported}"
        )
    return scheduler


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
    parser.add_argument(
        "--download-destination",
        type=Path,
        default=Path("downloading-station"),
        help="Destination for script-prepare-files-for-download.py",
    )
    parser.add_argument(
        "--corvus-mode",
        choices=["both", "exafs", "xanes"],
        default="both",
        help="Corvus template mode(s) to run: 'both' (default), 'exafs', or 'xanes'.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip script-extract-orca-compute-times.py in final postprocess job",
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
        "--state-file",
        type=Path,
        default=None,
        help="Optional explicit path for pipeline state JSON",
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

    if not args.skip_process_feff:
        # Keep explicit: script-process-feff-output imports numpy/matplotlib/larch at runtime.
        # This pre-check catches missing python early on head/login nodes.
        _check_executable("python")

    submit_command = SCHEDULER_SUBMIT_COMMAND[args.scheduler]
    if not args.no_submit:
        _check_executable(submit_command)

    script_dir = Path(__file__).resolve().parent
    prepare_orca_py = script_dir / "prepare-orca.py"
    prepare_corvus_py = script_dir / "prepare-corvus.py"

    if not prepare_orca_py.exists() or not prepare_corvus_py.exists():
        raise SystemExit("ERROR: Missing prepare-orca.py or prepare-corvus.py next to this script")

    xyz_files, input_base_dir = _discover_xyz_files(args.path.expanduser())

    if args.out_dir is None:
        output_root = input_base_dir.parent.resolve()
        print(f"No --out-dir provided; defaulting to: {output_root}")
    else:
        output_root = args.out_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_log = output_root / "batch-jobs.log"
    _initialize_batch_log(batch_log, args.scheduler)

    download_destination = args.download_destination.expanduser().resolve()

    prepare_cmd = [
        "python",
        str(prepare_orca_py),
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

    prep_result = _run_command(prepare_cmd)
    if prep_result.returncode != 0:
        _append_batch_job_log(batch_log, args.scheduler, "prepare-orca", "FAILED")
        raise RuntimeError(
            "prepare-orca.py failed:\n"
            f"stdout:\n{prep_result.stdout}\n"
            f"stderr:\n{prep_result.stderr}"
        )
    _append_batch_job_log(batch_log, args.scheduler, "prepare-orca", "SUCCEEDED")

    records: list[JobRecord] = []
    for xyz in xyz_files:
        run_id = _run_id_from_xyz(xyz, optimization_mode=optimization_mode)
        run_dir = output_root / run_id
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
        corvus_modes_to_submit = ["exafs", "xanes"] if args.corvus_mode == "both" else [args.corvus_mode]
        corvus_wrappers = []
        corvus_job_ids: list[str] = []

        for cmode in corvus_modes_to_submit:
            corvus_wrapper = run_dir / f"generated-{run_id}-corvus-{cmode}-wrapper.script"
            _write_corvus_wrapper_script(
                corvus_wrapper,
                run_dir,
                run_id,
                prepare_corvus_py,
                args.scheduler,
                corvus_mode=cmode,
            )
            corvus_wrappers.append(corvus_wrapper)

        if args.no_submit:
            orca_job_id = "NO_SUBMIT"
            orca_submitted_utc = _utc_now_iso()
            corvus_submitted_utc = _utc_now_iso()
            corvus_job_ids = ["NO_SUBMIT"] * len(corvus_modes_to_submit)
            _append_batch_job_log(batch_log, args.scheduler, f"orca-{run_id}", "SKIPPED")
            for cmode in corvus_modes_to_submit:
                _append_batch_job_log(batch_log, args.scheduler, f"corvus-{cmode}-{run_id}", "SKIPPED")
        else:
            orca_submitted_utc = _utc_now_iso()
            try:
                orca_job_id = _submit_job(orca_script, cwd=run_dir, scheduler=args.scheduler)
                _append_batch_job_log(
                    batch_log,
                    args.scheduler,
                    f"orca-{run_id}",
                    "SUCCEEDED",
                    job_id=orca_job_id,
                )
            except Exception:
                _append_batch_job_log(batch_log, args.scheduler, f"orca-{run_id}", "FAILED")
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
                        args.scheduler,
                        f"corvus-{cmode}-{run_id}",
                        "SUCCEEDED",
                        job_id=cjid,
                    )
                except Exception:
                    _append_batch_job_log(batch_log, args.scheduler, f"corvus-{cmode}-{run_id}", "FAILED")
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
        script_dir,
        args.scheduler,
        output_root,
        download_destination,
        skip_extract=args.skip_extract,
        skip_process_feff=args.skip_process_feff,
        skip_prepare_download=args.skip_prepare_download,
    )

    postprocess_job_id: str | None
    if args.no_submit:
        postprocess_job_id = "NO_SUBMIT"
        _append_batch_job_log(
            batch_log,
            args.scheduler,
            f"postprocess-{output_root.name}",
            "SKIPPED",
        )
    else:
        corvus_ids = [jid for rec in records for jid in rec.corvus_job_ids]
        try:
            postprocess_job_id = _submit_job(
                postprocess_script,
                cwd=output_root,
                scheduler=args.scheduler,
                depend_afterok=corvus_ids,
            )
            _append_batch_job_log(
                batch_log,
                args.scheduler,
                f"postprocess-{output_root.name}",
                "SUCCEEDED",
                job_id=postprocess_job_id,
            )
        except Exception:
            _append_batch_job_log(
                batch_log,
                args.scheduler,
                f"postprocess-{output_root.name}",
                "FAILED",
            )
            raise

    state_file = (
        args.state_file.expanduser().resolve()
        if args.state_file is not None
        else output_root / f"pipeline-state-{output_root.name}.json"
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
        json.dumps(
            {
                **asdict(state),
                "prepare_orca_stdout": prep_result.stdout,
                "prepare_orca_stderr": prep_result.stderr,
            },
            indent=2,
        )
        + "\n",
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


if __name__ == "__main__":
    raise SystemExit(main())
