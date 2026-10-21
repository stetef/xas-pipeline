"""xas-rerun-corvus: recompute CORVUS spectra on a finished batch, ORCA untouched.

Typically after editing the CORVUS inputs (corvus-template-xas.in, xanes.in,
exafs.in). For each selected run with a ``<id>.hess`` (flat, or split into
``working-<id>/`` + ``output-<id>/``):

1. archives the prior CORVUS outputs (:func:`xas_pipeline.recovery.archive_corvus_outputs`)
   so the recompute starts clean;
2. regenerates the CORVUS wrapper and submits it (no ORCA dependency);

then, unless ``--no-postprocess``, one batch postprocess (no ORCA check; download
with ``--refresh``) gated on those jobs. Its cleanup pass deletes the archived
snapshots once the new spectra exist; ``--skip-cleanup`` keeps them (and every
FEFF intermediate).

:func:`rerun_ids` is the same sequence as a function; the automatic triage of
dead XANES legs (:mod:`xas_pipeline.cli.auto_rerun_corvus`) calls it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, layout, recovery
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import add_scheduler_argument, check_executable, get_scheduler

log = logging.getLogger(__name__)

NO_SUBMIT = "NO_SUBMIT"


@dataclass
class RerunRecord:
    run_id: str
    run_dir: Path
    corvus_job_id: str
    archived: list[str]


@dataclass
class RerunOutcome:
    tag: str
    records: list[RerunRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: run id -> submission error (archived but not resubmitted).
    failed: dict[str, str] = field(default_factory=dict)
    postprocess_job_id: str | None = None
    state_file: Path | None = None


def resolve_run_dir(id_dir: Path) -> tuple[Path | None, str]:
    """``(dir holding <id>.hess, "split"|"flat")``, or ``(None, reason)``."""
    run_id = id_dir.name
    working = id_dir / f"working-{run_id}"
    if (working / f"{run_id}.hess").is_file():
        return working, "split"
    if (id_dir / f"{run_id}.hess").is_file():
        return id_dir, "flat"
    return None, "no <id>.hess (in the run dir or working-<id>/)"


def default_tag() -> str:
    """``rerun-YYYYMMDDTHHMMSSZ``: filename-safe, and marked for cleanup."""
    stamp = batch_log.utc_now().split("+", 1)[0].replace(":", "").replace("-", "")
    return f"rerun-{stamp}Z"


def rerun_ids(
    batch_root: Path,
    *,
    only_ids: set[str] | None = None,
    match_groups: bool = True,
    tag: str | None = None,
    scheduler: str,
    download_destination: Path | None = None,
    postprocess: bool = True,
    postprocess_after: str = "afterok",
    skip_cleanup: bool = False,
    submit: bool = True,
) -> RerunOutcome:
    """Archive and resubmit CORVUS for the selected runs (see the module docstring).

    *only_ids* matches run dirs by their own name or their group's (None: all);
    *match_groups* False requires their own name -- a pre-suffix ``<id>/`` can
    also group later mode runs, whose healthy spectra an automated caller must
    not drag in. *postprocess_after*: ``afterok`` holds the postprocess unless
    every recompute succeeds (the default, for hand-run reruns of healthy runs);
    ``afterany`` lets it triage failed recomputes. Without *submit*, nothing is
    archived, submitted or logged.
    """
    batch_root = Path(batch_root).expanduser().resolve()
    sched = get_scheduler(scheduler)
    if submit:
        check_executable(sched.submit_command)
    tag = tag or default_tag()
    download_destination = Path(download_destination or batch_root / "downloading-station").expanduser().resolve()
    log_path = batch_root / batch_log.LOG_NAME
    if submit:
        batch_log.init(log_path, sched)
    outcome = RerunOutcome(tag=tag)

    for id_dir in layout.iter_id_dirs(batch_root, only_ids=only_ids):
        run_id = id_dir.name
        if not match_groups and only_ids is not None and run_id not in only_ids:
            continue
        run_dir, how = resolve_run_dir(id_dir)
        if run_dir is None:
            log.info("SKIP %s: %s", run_id, how)
            outcome.skipped.append(f"{run_id}: {how}")
            continue
        output_dir = id_dir / f"output-{run_id}"
        log.info("=== %s (%s) ===", run_id, how)
        archived = recovery.archive_corvus_outputs(
            run_dir, output_dir if output_dir.is_dir() else None, run_id, tag, dry_run=not submit
        )
        wrapper = jobscripts.corvus_wrapper(sched, run_dir, run_id)
        if not submit:
            outcome.records.append(RerunRecord(run_id, run_dir, NO_SUBMIT, archived))
            continue
        try:
            job_id = batch_log.submit_and_log(sched, log_path, f"rerun-corvus-xas-{run_id}", wrapper, run_dir)
        except (RuntimeError, ValueError, OSError) as exc:
            outcome.failed[run_id] = str(exc)
            log.error("  submit failed for %s: %s", run_id, exc)
            continue
        outcome.records.append(RerunRecord(run_id, run_dir, job_id, archived))

    submitted = [r.corvus_job_id for r in outcome.records if r.corvus_job_id != NO_SUBMIT]
    if postprocess and outcome.records:
        script = jobscripts.postprocess(
            sched, batch_root, download_destination,
            extract=False, cleanup=not skip_cleanup, refresh_download=True,
            dest=batch_root / f"generated-rerun-postprocess-{batch_root.name}-xas.script",
        )
        if not submit:
            outcome.postprocess_job_id = NO_SUBMIT
        elif submitted:
            try:
                outcome.postprocess_job_id = batch_log.submit_and_log(
                    sched, log_path, f"rerun-postprocess-{batch_root.name}", script, batch_root,
                    **{postprocess_after: submitted},
                )
            except (RuntimeError, ValueError, OSError) as exc:
                outcome.failed[f"rerun-postprocess-{batch_root.name}"] = str(exc)

    if outcome.records:
        outcome.state_file = batch_root / f"rerun-state-{batch_root.name}-xas-{tag}.log"
        outcome.state_file.write_text(_render_state(outcome, batch_root, sched.name, download_destination,
                                                    skip_cleanup), encoding="utf-8")
    log.info("Re-ran %d run(s), skipped %d, failed %d; postprocess %s; tag %s",
             len(outcome.records), len(outcome.skipped), len(outcome.failed), outcome.postprocess_job_id, tag)
    return outcome


def _render_state(outcome: RerunOutcome, batch_root: Path, scheduler: str, destination: Path,
                  skip_cleanup: bool) -> str:
    lines = [
        "CSD Zn-complex pipeline - CORVUS rerun state",
        "=" * 44,
        f"created_utc:          {batch_log.utc_now()}",
        f"batch_root:           {batch_root}",
        "corvus_mode:          xas",
        f"tag:                  {outcome.tag}",
        f"scheduler:            {scheduler}",
        f"download_destination: {destination}",
        f"cleanup:              {'SKIPPED (all FEFF scratch retained)' if skip_cleanup else 'enabled'}",
        f"postprocess_job_id:   {outcome.postprocess_job_id}",
        "",
        f"Skipped ({len(outcome.skipped)}):",
        *(f"    {line}" for line in outcome.skipped or ["(none)"]),
        "",
        f"Runs ({len(outcome.records)}):",
        "",
    ]
    for rec in outcome.records:
        lines += [
            f"  {rec.run_id}",
            f"    run_dir:        {rec.run_dir}",
            f"    corvus_job_id:  {rec.corvus_job_id}",
            f"    archived:       {', '.join(rec.archived) or '(none)'}",
            "",
        ]
    for name, error in outcome.failed.items():
        lines.append(f"SUBMISSION FAILED: {name}: {error.splitlines()[0] if error else ''}")
    lines.append("Statuses here are SUBMISSION only; per-run outcomes are in batch-jobs.log.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute CORVUS spectra on a finished batch, ORCA untouched.")
    parser.add_argument("batch_root", type=Path, help="The batch root")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated run ids or structure names (default: every run with a .hess)")
    parser.add_argument("--tag", default=None, help="Archive suffix (default: rerun-<UTC timestamp>)")
    add_scheduler_argument(parser)
    parser.add_argument("--download-destination", type=Path, default=None,
                        help="Download destination (default: <batch_root>/downloading-station)")
    parser.add_argument("--no-postprocess", action="store_true", help="Submit the CORVUS jobs only")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Keep the archived snapshots and every FEFF intermediate")
    parser.add_argument("--no-submit", "--dry-run", dest="no_submit", action="store_true",
                        help="Write the wrappers; archive, submit and log nothing")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"ERROR: not a directory: {batch_root}")
    outcome = rerun_ids(
        batch_root,
        only_ids={s for s in args.ids.split(",") if s} if args.ids else None,
        tag=args.tag,
        scheduler=args.scheduler,
        download_destination=args.download_destination,
        postprocess=not args.no_postprocess,
        skip_cleanup=args.skip_cleanup,
        submit=not args.no_submit,
    )
    return 0 if outcome.records and not outcome.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
