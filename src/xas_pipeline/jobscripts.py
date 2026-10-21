"""Render the generated job scripts: ORCA job, CORVUS wrapper, batch postprocess, watchdog.

Each comes from ``data/<scheduler>-scripts/<name>.script`` with its
``[TOKEN]`` placeholders filled; all are written executable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import config, layout, modes, resources, templates
from .scheduler import Scheduler

# Every job script sets PYTHON_BIN (to [PYTHON], the interpreter rendering it,
# unless $PIPELINE_PYTHON overrides); injected commands run under it.
PY = '"$PYTHON_BIN"'


def _render(sched: Scheduler, template_name: str, dest: Path, tokens: dict) -> Path:
    template = resources.template_root() / sched.template_subdir / template_name
    tokens = {
        "PIPELINE_ROOT": resources.pipeline_root(),
        "PIPELINE_ENV": resources.pipeline_env(),
        "PYTHON": sys.executable,
        **tokens,
    }
    return templates.render(template, dest, tokens, executable=True, ensure_trailing_newline=True)


def corvus_wrapper_path(run_dir: Path, run_id: str, corvus_mode: str = "xas") -> Path:
    return Path(run_dir) / f"generated-{run_id}-corvus-{corvus_mode}-wrapper.script"


def orca_job(sched: Scheduler, run_dir: Path, run_id: str, *, nprocs: int, mem_gb: int) -> Path:
    """``generated-<run_id>-orca.script``: runs ``<run_id>.in`` in node scratch."""
    return _render(sched, "orca-job.script", Path(run_dir) / f"generated-{run_id}-orca.script", {
        "NPROCS": nprocs or 1,
        "BASENAME": run_id,
        "INPUT_FILE": f"{run_id}.in",
        # Slurm wants "64G", PBS "64gb".
        "MEM": f"{mem_gb}gb" if sched.name == "pbs" else f"{mem_gb}G",
        "SCRATCH_EXCLUDE": "|".join(config.SCRATCH_EXCLUDE_GLOBS),
    })


def corvus_job(sched: Scheduler, run_dir: Path, run_id: str, corvus_mode: str = "xas") -> Path:
    """``corvus-job-<mode>.script``: runs CORVUS on ``corvus-<run_id>-<mode>.in``.

    Executed inline by the CORVUS wrapper, so it is not made executable.
    """
    template = resources.template_root() / sched.template_subdir / "corvus-job.script"
    return templates.render(template, Path(run_dir) / f"corvus-job-{corvus_mode}.script", {
        "DIRECTORY": f"{run_dir}/",
        "ID": run_id,
        "CORVUS_MODE": corvus_mode,
        "CORVUS_INPUT_BASENAME": f"corvus-{run_id}-{corvus_mode}.in",
        "CORVUS_OUTPUT_BASENAME": f"corvus-{run_id}-{corvus_mode}.out",
        "PIPELINE_ENV": resources.pipeline_env(),
    })


def corvus_wrapper(
    sched: Scheduler,
    run_dir: Path,
    run_id: str,
    *,
    mode: str | None = None,
    corvus_mode: str = "xas",
    dest: Path | None = None,
) -> Path:
    """The CORVUS job: (spring Hessian if the mode needs it), prepare-corvus, CORVUS.

    *mode* defaults to the one encoded in *run_id*; a run dir predating mode
    suffixes is treated as having an ORCA Hessian, which every such batch did.
    """
    mode = mode or layout.mode_from_run_id(run_id) or "unknown"
    spring = mode in modes.SPRING_HESSIAN_MODES
    interp_hess_cmd = f'{PY} -m xas_pipeline.stages.interp_hessian "$RUN_DIR" --run-id "$RUN_ID"' if spring else "true"
    return _render(sched, "corvus-wrapper.script", dest or corvus_wrapper_path(run_dir, run_id, corvus_mode), {
        "RUN_DIR": run_dir,
        "RUN_ID": run_id,
        "SCHEDULER": sched.name,
        "CORVUS_MODE": corvus_mode,
        "INTERP_HESS_CMD": interp_hess_cmd,
        "OPTIMIZATION_MODE": mode,
        "SPRING_HESSIAN": int(spring),
    })


def postprocess(
    sched: Scheduler,
    batch_root: Path,
    download_destination: Path,
    *,
    extract: bool = True,
    process_feff: bool = True,
    cleanup: bool = True,
    download: bool = True,
    refresh_download: bool = False,
    dest: Path | None = None,
) -> Path:
    """The batch postprocess job: ORCA check -> FEFF processing -> cleanup ->
    CORVUS auto-rerun triage -> download staging -> stage times. Disabled steps
    become ``true``."""
    py = PY
    root = f'"{batch_root}"'
    # The auto-rerun triage runs after cleanup (nothing prunes a recompute
    # mid-flight) and before download (rerun ids leave corvus-failed-ids.txt
    # before quarantine reads it). It passes this job's download destination and
    # cleanup choice on to the follow-up postprocess it queues.
    steps = {
        "EXTRACT_CMD": extract and f"{py} -m xas_pipeline.stages.orca_check {root} --output-dir {root}",
        "PROCESS_FEFF_CMD": process_feff and f"{py} -m xas_pipeline.stages.feff_process {root} --recursive",
        "CLEANUP_CMD": cleanup and f"{py} -m xas_pipeline.stages.cleanup {root} --execute",
        "AUTO_RERUN_CORVUS_CMD": process_feff and (
            f"{py} -m xas_pipeline.cli.auto_rerun_corvus {root} --scheduler {sched.name} "
            f'--download-destination "{download_destination}"' + ("" if cleanup else " --skip-cleanup")
        ),
        "PREPARE_DOWNLOAD_CMD": download and (
            f'{py} -m xas_pipeline.stages.download {root} -d "{download_destination}"'
            + (" --refresh" if refresh_download else "")
        ),
        "STAGE_TIMES_CMD": f"{py} -m xas_pipeline.stages.stage_times {root}",
    }
    batch_root = Path(batch_root)
    return _render(
        sched,
        "postprocess-job.script",
        dest or batch_root / f"generated-postprocess-{batch_root.name}.script",
        {
            "BATCH_NAME": batch_root.name,
            "OUTPUT_ROOT": batch_root,
            **{token: cmd or "true" for token, cmd in steps.items()},
        },
    )


# Watchdog defaults, shared by xas-run-batch and xas-watchdog.
WATCHDOG_INTERVAL = 600
WATCHDOG_IDLE_THRESHOLD = 3600
WATCHDOG_MAX_RERUNS = 2
WATCHDOG_WALLTIME = "168:00:00"


def watchdog(
    sched: Scheduler,
    batch_root: Path,
    *,
    interval: int = WATCHDOG_INTERVAL,
    idle_threshold: int = WATCHDOG_IDLE_THRESHOLD,
    max_reruns: int = WATCHDOG_MAX_RERUNS,
    walltime: str = WATCHDOG_WALLTIME,
) -> Path:
    """The hang-watchdog companion job for a batch."""
    batch_root = Path(batch_root)
    return _render(sched, "watchdog-job.script", batch_root / f"generated-watchdog-{batch_root.name}.script", {
        "BATCH_NAME": batch_root.name,
        "OUTPUT_ROOT": batch_root,
        "SCHEDULER": sched.name,
        "WATCHDOG_INTERVAL": interval,
        "WATCHDOG_IDLE_THRESHOLD": idle_threshold,
        "WATCHDOG_MAX_RERUNS": max_reruns,
        "WATCHDOG_WALLTIME": walltime,
    })
