"""Recompute, automatically, the CORVUS runs that hit a known sporadic failure.

Auto-remediable kinds (:data:`xas_pipeline.corvus_diagnosis.AUTO_REMEDIABLE`): a dead XANES leg (below), and the two
FEFF 10 artifacts the ``gg.bin`` check finds (a CR/LF header byte that breaks MKGTR, odd-XANES group A; an all-zero
FMS section from misfiled results, group B). Each recompute runs FEFF on a different MPI rank count
(:data:`FEFF_RANK_LADDER`): group B depends on the rank count, and some structures hit it on most runs at a given
count, so repeating the same count cannot clear them; the last rung, 1 rank, cannot hit it at all.

This is the CORVUS counterpart of ``xas-rerun-orca``: a postprocess stage rather
than a per-job hook, because a CORVUS failure is only visible once the spectra
have been read. It runs inside the batch postprocess job (between ``xas-cleanup``
and ``xas-download``) and can be run by hand on any batch root.

The failure it exists for
-------------------------
A CORVUS run can finish successfully and still produce a xanes ``xmu.dat`` whose
``chi`` column is identically zero -- ``mu == mu0`` on every row, no fine
structure at all. Nothing in the headers shows it (the "0/0 paths used" line is
meaningless for XANES, which is FMS rather than a path expansion), so until the
gate started looking at the numbers these runs passed as good spectra. The
failure is *sporadic*, not structural: the XANES leg is not bit-reproducible, and
the same geometry recomputed usually comes back clean. That is what makes a plain
recompute -- same inputs, fresh run -- a real remedy, and the only CORVUS failure
kind that is auto-remediable (see :mod:`xas_pipeline.corvus_diagnosis`).

What it does
------------
1. Reads ``corvus-failed-ids.txt`` -- the manifest ``xas-process-feff`` just
   wrote -- so only ids the gate already failed are considered. No manifest, or an
   empty one, means there is nothing to do.
2. Re-derives each id's verdict from disk (:func:`corvus_diagnosis.diagnose_run_dir`)
   rather than parsing failure text, and keeps only the auto-remediable kind.
3. Bounds the ladder with a per-run state file (``<id>-corvus-rerun-state.json``,
   :mod:`xas_pipeline.rerun_state`) exactly as the ORCA ladder does: at most
   ``MAX_ATTEMPTS`` automatic recomputes, then the run is escalated to a human
   (``resolution=needs_human`` + a ``NEEDS_HUMAN`` line in ``batch-jobs.log``).
4. Hands the surviving ids to :func:`xas_pipeline.cli.rerun_corvus.rerun_ids`,
   which archives the dead output, resubmits the corvus wrapper (FEFF on the rung's rank count), and queues one
   follow-up postprocess job (``afterany``) that re-derives the spectra -- and
   runs this triage again, continuing (or ending) the ladder whether the
   recompute came back dead or failed outright.
5. Drops the resubmitted ids from ``corvus-failed-ids.txt``, so the *next* stage
   in this same job (``xas-download``) leaves their run dirs alone instead of
   quarantining a directory a queued job is about to write into. Ids that were
   escalated, or that failed for any other reason, stay in the manifest and are
   quarantined into ``failed-corvus/`` exactly as before.

Turning it off: ``XAS_AUTO_RERUN=0`` in the job environment (the same switch as
the ORCA hook; the postprocess job sources the pipeline ``.env``). ``--no-submit``
previews the triage without archiving or submitting anything.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from xas_pipeline import batch_log, corvus_diagnosis, layout, recovery, rerun_state
from xas_pipeline.cli import rerun_corvus
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import add_scheduler_argument

log = logging.getLogger(__name__)

# FEFF MPI rank count for each automatic recompute (attempt 1, 2, 3); the first run
# uses the wrapper default, cores - 1 = 15 on the 16-core job, and every recompute
# keeps the 16 cores (tetef01 2026-11-20). FEFF 10's FMS never clears its rank map
# (fmstot.f90:110), so rank 0 can misfile results (odd-XANES group B); which
# structures it hits depends on the rank count (PUZKUQ_tilt s09 failed 4 of 5 runs
# at 15 ranks), so each recompute changes it. At 1 rank FMS gathers nothing and the
# bug cannot occur -- slow (serial FMS, ~15x) but certain, hence last.
FEFF_RANK_LADDER = (14, 13, 1)

# Automatic recomputes allowed per run dir: one per rung. The count is cumulative
# over the run dir's lifetime -- delete the state file to grant a run a fresh ladder.
MAX_ATTEMPTS = len(FEFF_RANK_LADDER)

# Recorded in the state file as the remedy: archive the dead output and recompute
# from the same inputs (on the attempt's rank count, see FEFF_RANK_LADDER).
REMEDY_LABEL = "archive-and-recompute"

# Environment switch shared with the ORCA auto-rerun hook.
AUTO_RERUN_ENV = "XAS_AUTO_RERUN"


@dataclass
class Candidate:
    """One failed id that the triage decided to recompute."""

    run_id: str
    run_dir: Path          # the id/run directory (state file lives here)
    attempt: int
    reason: str
    kind: str = corvus_diagnosis.CorvusFailureKind.XANES_ZERO_CHI.value

    @property
    def feff_ranks(self) -> int:
        return feff_ranks_for(self.attempt)


def feff_ranks_for(attempt: int) -> int:
    """FEFF's MPI rank count for automatic recompute *attempt* (1-based); past the
    ladder (a raised ``--max-attempts``) it stays on the last, safe rung."""
    return FEFF_RANK_LADDER[min(max(attempt, 1), len(FEFF_RANK_LADDER)) - 1]


def triage(
    batch_root: Path, mode: str = "xas", max_attempts: int = MAX_ATTEMPTS
) -> tuple[list[Candidate], list[str]]:
    """Decide which failed ids to recompute.

    Returns ``(candidates, keep_failed)``: the runs to resubmit, and the ids that
    must stay in the failed manifest (not auto-remediable, ladder exhausted, or
    already escalated). Escalations are recorded as a side effect, once -- a
    state file with a resolution is never re-escalated.
    """
    batch_root = Path(batch_root)
    failed_ids = set(layout.read_failed_corvus_ids(batch_root))
    if not failed_ids:
        return [], []

    candidates: list[Candidate] = []
    keep_failed: list[str] = []
    seen: set[str] = set()

    for run_dir in layout.iter_id_dirs(batch_root):
        run_id = run_dir.name
        if run_id not in failed_ids:
            continue
        seen.add(run_id)

        diag = corvus_diagnosis.diagnose_run_dir(run_dir, mode)
        log.info("[%s] diagnosis: %s -- %s", run_id, diag.kind.value, diag.reason)
        if diag.ok:
            # The gate failed this id for something outside the spectrum itself
            # (a processing error, a missing xyz); leave it to a human.
            keep_failed.append(run_id)
            continue
        if not diag.auto_remediable:
            log.info("[%s] '%s' is not auto-remediable; leaving it failed.", run_id, diag.kind.value)
            keep_failed.append(run_id)
            continue

        state_file = rerun_state.state_path(run_dir, run_id, kind="corvus")
        state = rerun_state.load_state(state_file, run_id)
        if state.is_terminal:
            log.info("[%s] already resolved '%s'; leaving it failed.", run_id, state.resolution)
            keep_failed.append(run_id)
            continue

        attempt = state.next_attempt
        if attempt > max_attempts:
            recovery.escalate(
                state_file, state, batch_log.find_batch_log(batch_root), f"corvus-{run_id}",
                f"auto-rerun ladder exhausted after {len(state.attempts)} recompute(s); "
                f"last failure '{diag.kind.value}': {diag.reason}",
                "auto-rerun escalation (corvus)",
            )
            keep_failed.append(run_id)
            continue

        candidates.append(
            Candidate(run_id=run_id, run_dir=run_dir, attempt=attempt, reason=diag.reason, kind=diag.kind.value)
        )

    # Ids in the manifest with no run dir under the batch root any more (already
    # quarantined by an earlier download stage, or moved by hand) stay listed:
    # this stage never resurrects a directory it cannot see.
    keep_failed.extend(sorted(failed_ids - seen))
    return candidates, keep_failed


def _record_attempts(candidates: list[Candidate], outcome: rerun_corvus.RerunOutcome) -> None:
    """Append one attempt per resubmitted run to its state file."""
    job_by_id = {rec.run_id: rec.corvus_job_id for rec in outcome.records}
    for cand in candidates:
        if cand.run_id not in job_by_id:
            continue
        state_file = rerun_state.state_path(cand.run_dir, cand.run_id, kind="corvus")
        state = rerun_state.load_state(state_file, cand.run_id)
        state.attempts.append(
            rerun_state.Attempt(
                attempt=cand.attempt,
                kind=cand.kind,
                remedy=REMEDY_LABEL,
                utc=batch_log.utc_now(),
                corvus_job_id=job_by_id[cand.run_id],
                note=f"archive tag {outcome.tag}; FEFF on {cand.feff_ranks} MPI rank(s); {cand.reason}",
            )
        )
        rerun_state.save_state(state_file, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Triage the CORVUS-failed ids of a batch and automatically recompute the "
            "ones with a sporadic failure: a dead XANES leg (chi identically zero) or a known FEFF "
            "artifact in FMS's gg.bin (CR/LF header byte, all-zero section)."
        )
    )
    parser.add_argument("batch_root", type=Path, help="Batch output root (parent of the per-id dirs)")
    parser.add_argument("--corvus-mode", choices=["xas"], default="xas", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-attempts", type=int, default=MAX_ATTEMPTS,
        help=f"Max automatic recomputes per run (default: {MAX_ATTEMPTS}).",
    )
    add_scheduler_argument(parser)
    parser.add_argument(
        "--download-destination", type=Path, default=None,
        help="Download destination for the follow-up postprocess job "
             "(default: <batch_root>/downloading-station).",
    )
    parser.add_argument(
        "--skip-cleanup", action="store_true",
        help="Pass --skip-cleanup on to the follow-up postprocess job, keeping every "
             "CORVUS/FEFF intermediate of the recomputed runs.",
    )
    parser.add_argument(
        "--no-submit", "--dry-run", dest="no_submit", action="store_true",
        help="Report the triage and write the wrappers, but archive, submit and "
             "record nothing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        print(f"ERROR: batch_root is not a directory: {batch_root}")
        return 2

    if os.environ.get(AUTO_RERUN_ENV, "1") == "0":
        print(f"{AUTO_RERUN_ENV}=0: skipping CORVUS auto-rerun triage.")
        return 0

    failed_ids = layout.read_failed_corvus_ids(batch_root)
    if not failed_ids:
        print(f"No ids in {layout.FAILED_CORVUS_MANIFEST}; nothing to triage.")
        return 0

    print(f"Triaging {len(failed_ids)} CORVUS-failed id(s) under {batch_root}")
    candidates, keep_failed = triage(batch_root, args.corvus_mode, args.max_attempts)

    if not candidates:
        print("\nNothing auto-remediable; leaving every failed id as it is.")
        return 0

    print(
        f"\nRecomputing {len(candidates)} run(s): "
        + ", ".join(
            f"{c.run_id} ({c.kind}, attempt {c.attempt}/{args.max_attempts}, FEFF on {c.feff_ranks} rank(s))"
            for c in candidates
        )
    )

    outcome = rerun_corvus.rerun_ids(
        batch_root,
        only_ids={c.run_id for c in candidates},
        # Exactly the runs triaged: a pre-suffix run dir that also groups later
        # mode runs must not drag its siblings' healthy spectra into the rerun.
        match_groups=False,
        scheduler=args.scheduler,
        download_destination=args.download_destination,
        # afterany: a recompute that fails outright must still be triaged (and
        # escalated), not leave the follow-up waiting forever.
        postprocess_after="afterany",
        skip_cleanup=args.skip_cleanup,
        submit=not args.no_submit,
        feff_ranks={c.run_id: c.feff_ranks for c in candidates},
    )

    resubmitted = {rec.run_id for rec in outcome.records}
    unrunnable = [c.run_id for c in candidates if c.run_id not in resubmitted]
    for run_id in unrunnable:
        # Skipped (no <id>.hess) or its submission failed: not something a retry
        # here fixes, so it stays failed and quarantinable.
        log.info("[%s] not resubmitted; leaving it failed.", run_id)

    if args.no_submit:
        print("\n--no-submit: manifest and attempt state left untouched.")
        return 0

    _record_attempts(candidates, outcome)

    manifest = layout.write_failed_corvus_ids(batch_root, keep_failed + unrunnable)
    print(
        f"\nResubmitted {len(resubmitted)} run(s); "
        f"{len(keep_failed) + len(unrunnable)} id(s) stay in {manifest.name} "
        "(and will be quarantined by the download stage)."
    )
    print(f"Follow-up postprocess job: {outcome.postprocess_job_id}")
    return 0


if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
