# Automatic recovery

Three mechanisms resubmit work without a human: the ORCA job's failure hook
(`xas-rerun-orca`), the watchdog (`xas-watchdog`), and the CORVUS dead-XANES
recompute (`xas-auto-rerun-corvus`). They share one rule: a rerun is attempted only
when the cure does not change the calculation, each ladder is bounded, and when it
runs out the run is handed to a human.

## Where decisions are recorded

Each ladder keeps its own JSON state file in the run dir, so the counters never
interfere and a re-triage of the same failure is idempotent:

| File | Ladder |
|---|---|
| `<run id>-rerun-state.json` | failed ORCA job (`xas-rerun-orca`) |
| `<run id>-hang-rerun-state.json` | hung ORCA job (watchdog) |
| `<run id>-corvus-hang-rerun-state.json` | hung CORVUS job (watchdog) |
| `<run id>-corvus-rerun-state.json` | dead XANES leg (`xas-auto-rerun-corvus`) |

Each holds one entry per attempt (failure kind, remedy, job ids, time) and, once
the ladder ends, `"resolution": "needs_human"`. The file is the authoritative
record; the counts are cumulative over the run dir's life. Delete a state file to
give a run a fresh ladder.

An escalation also appends a `NEEDS_HUMAN` line with the reason to
`batch-jobs.log`, next to every other outcome. A state file that cannot be read is
renamed to `<name>.corrupt` and the ladder is ended (`needs_human`), rather than
restarted.

`XAS_AUTO_RERUN=0` in the job environment turns off the ORCA failure hook and the
CORVUS recompute. The watchdog is turned off with `xas-run-batch --no-watchdog`.

## Failed ORCA jobs

When ORCA does not terminate normally, the ORCA job script runs
`xas-rerun-orca <run dir>` as `"$PYTHON_BIN" -m xas_pipeline.cli.rerun_orca`, so
nothing needs to be on the node's `PATH` (output in `<run id>-rerun.log`). This fires per
structure the moment a run fails, rather than when the batch postprocess runs. The
command:

1. diagnoses the failure from the ORCA log (`xas_pipeline.diagnosis`);
2. cancels the CORVUS job waiting `afterok` on this run. It can never start, and as
   a `DependencyNeverSatisfied` job it would hold the batch postprocess forever. The
   cancel is logged as `CANCELLED` in `batch-jobs.log`;
3. escalates when the failure is not auto-remediable or the ladder is spent;
4. otherwise applies the remedy to `<run id>.in`, always starting from the
   original input kept in `<run id>-rerun-history/`, so remedies never stack;
5. resubmits ORCA and a fresh CORVUS job, or, for a `--pre` stage, re-chains the
   next stage, and re-gates a pending batch postprocess onto the new jobs
   (cancelling the stale CORVUS job can otherwise release its `afterany`).

| Log signature | Diagnosis | Action |
|---|---|---|
| `The optimization did not converge but reached the maximum number of` | `opt_nonconvergence` | restart from the last geometry (1 attempt) |
| `No memory left for COSX` / `Increase the %MAXCORE` | `oom` | `%MaxCore` and the job's memory request ×1.6, then ×2.5 |
| `multiplicity (…) is odd/even and number of electrons (…) … -> impossible` | `charge_mult` | human: fix CHARGE/MULTIPLICITY or the carved cluster |
| `SCF has not converged` / `SCF NOT CONVERGED`, with `Small HOMO/LUMO gap` | `scf_near_degeneracy` | human |
| same, energy flat over the last iterations | `scf_stalled` | human |
| same, energy still moving | `scf_diverged` | human |
| `ORCA finished by error termination in` PROPINT, NUMFREQ, ANFREQ, FREQ or HESSIAN | `post_opt` | human |
| no normal termination, nothing more specific | `crash` | human |
| no ORCA log | `no_log` | human |

Details of the two automatic remedies:

- The out-of-memory remedy also restarts from the last geometry when at least two
  optimization cycles ran.
- Both start from the previous orbitals when a non-empty `.gbw` exists: the input
  gets `! MOREAD` and `%moinp "<run id>-moread.gbw"`, a copy of `<run id>.gbw`
  (the job is about to overwrite the original).
- Both are physics-neutral: more memory, or the same optimization continued.

Why SCF non-convergence is not retried automatically: curing it means changing the
SCF settings, which changes the calculation, and auto-rerun recovers from resource
problems only, so the input stays the one a human wrote. The diagnosis still names
the likely cure for whoever picks it up:

- near-degeneracy (small HOMO–LUMO gap, occupations oscillating): level shift plus
  damping (`SlowConv`). Not smearing: fractional occupations are incompatible with
  the response (CPHF) step of `! AnFreq`, and ORCA aborts at input check;
- stalled (energy flat, oscillating just above threshold): `SlowConv` and more
  iterations from the previous orbitals (`MOREAD`);
- diverged: `SlowConv` plus a level shift from a fresh guess.

A run converged past a near-zero gap is still worth a manual check (HOMO–LUMO gap,
spin state).

To retriage a failure by hand, run `xas-rerun-orca <run dir> --scheduler slurm`
(`--no-submit` to preview). It diagnoses whatever the current `<run id>-orca.log`
says, so if a later manual run overwrote the original failure log, fix the input
yourself rather than relying on the diagnosis. Resubmitting needs a
`batch-jobs.log` in the run dir or a parent directory.

The postprocess's ORCA check (`xas-orca-check`) remains the backstop: it moves
every run that did not terminate normally into `failed-orca/` and writes one line
per run to `orca-convergence-report.log`. It skips runs whose ORCA job is still
queued or running (a `<run id>-orca.timing` with no `exit_code=`).

## Hung jobs: the watchdog

An MPI job can wedge, with no progress and no exit, and hold its allocation until
the walltime. `xas-rerun-orca` only sees jobs that exit, so `xas-run-batch` also
submits one watchdog job per batch (`watchdog-<batch>`, 168 h walltime by default).
Every `--watchdog-interval` seconds (600) it:

1. lists the batch's RUNNING ORCA and CORVUS jobs from `batch-jobs.log` and the
   scheduler;
2. reads, per job, when its progress log (`<run id>-orca.log` or
   `corvus-<run id>-xas.out`) was last written and whether its CPU time advanced
   since the previous poll. On Slurm the CPU time is the sum of the `cpu=` totals
   (`sstat` `TRESUsageInTot`) of every step except `.extern`, so MPI ranks count
   whichever step they run in; on PBS it is `resources_used.cput`;
3. calls a job hung when the log has been silent for `--watchdog-idle-threshold`
   seconds (3600) and the CPU time is flat. Requiring both lets a quiet but busy
   step survive. A job is never flagged on the first poll it is seen. When the
   scheduler reports no CPU time for a job, the silent log alone decides;
4. cancels a hung job and resubmits it clean, under Slurm excluding the node it hung
   on: ORCA together with its follower (a new CORVUS job or the next `--pre`
   stage); CORVUS with its previous output archived. A pending batch postprocess is
   re-gated onto the new jobs. After `--watchdog-max-reruns` (2) resubmits of one
   structure, the next hang is cancelled and escalated instead.

If a hung job cannot be cancelled, the watchdog escalates instead of resubmitting,
so the job is never doubled. The watchdog exits when nothing in the batch is
outstanding. Only one watchdog watches a batch: a later `xas-run-batch` does not
start a second while one is alive, and of two live watchdog jobs the later one
exits. Run it by hand with `xas-watchdog <batch> --scheduler slurm` (`--once` for
a single poll, `--no-exclude-node` to allow the same node).

## Dead XANES legs: CORVUS recompute

A CORVUS run can exit 0 and still produce a dead XANES leg: the xanes `xmu.dat` has
χ identically zero on every row (μ = μ₀ throughout, no fine structure). Nothing in
the FEFF headers shows it. The `0/0 paths used` line means nothing for XANES, which
is a full-multiple-scattering calculation, not a path expansion.

`xas-process-feff` therefore reads the numbers (`xas_pipeline.corvus_diagnosis`,
`chem.feff.scan_chi_column`). An all-zero χ fails the run like any other CORVUS
failure: its id goes into `corvus-failed-ids.txt` and a `FAILED` line into
`batch-jobs.log`. A partly zero or NaN/inf column is printed as a warning, not
failed: it is suspicious, not dead.

| Verdict | Meaning | Action |
|---|---|---|
| `xanes_zero_chi` | xanes χ identically 0 | recompute |
| `missing_spectrum` | no `Corvus.cfavg_xas.out` | human |
| `malformed_spectrum` | `Corvus.cfavg_xas.out` unreadable or not a 6-column table | human |
| `no_exafs_paths` | exafs `xmu.dat` reports `0/0 paths used` | human |

Why a plain recompute is a real remedy: the XANES leg is not bit-reproducible (the
same geometry and inputs, run twice, give spectra that differ near the edge), and
the zero-χ failure is sporadic rather than structural, so a recompute of the same
inputs normally comes back clean. Every other verdict points at the inputs, the
Hessian or a killed job, and rerunning it unchanged would fail the same way.

`xas-auto-rerun-corvus` runs inside the postprocess job, after cleanup (so nothing
prunes a recompute in progress) and before download (so the quarantine reads the
updated manifest). It:

1. reads `corvus-failed-ids.txt`, written by `xas-process-feff` in the same job;
2. re-derives each id's verdict from disk and keeps only `xanes_zero_chi`;
3. for each, unless its ladder (2 recomputes, `--max-attempts`) is spent, archives
   the dead output (`Corvus3_cfavg_xas/`, `Corvus.cfavg_xas.out` and the CORVUS log
   get a `.rerun-<UTC time>` suffix; spectra already in `output-<id>/` move to
   `xas-archive.rerun-<UTC time>/`), resubmits the CORVUS job, and queues one
   follow-up postprocess `afterany` on the recomputes, which processes the new
   spectra and triages again;
4. removes the resubmitted ids from `corvus-failed-ids.txt`, so the download step
   in the same job leaves their run dirs in place instead of quarantining a
   directory a queued job is about to write into.

Ids that were escalated or failed for another reason stay in the manifest and are
moved to `failed-corvus/` by the download step. `--no-submit` previews the triage
without archiving, submitting or recording anything. The archived snapshots are
deleted by the next cleanup pass unless `--skip-cleanup` is given.

## Manual reruns

- `xas-rerun-corvus <batch> [--ids a,b]` archives and recomputes CORVUS for the
  selected runs (every run with a `.hess` by default), for example after editing
  `corvus-template-xas.in`, `xanes.in` or `exafs.in`. It then submits a postprocess
  without the ORCA check, gated `afterok` on the recomputes and refreshing the
  download copies. `--tag` names the archive suffix (default `rerun-<UTC time>`).
  It writes `rerun-state-<batch>-xas-<tag>.log` at the batch root.
- `xas-submit-corvus <batch>` submits CORVUS for every run dir that has its
  `<run id>.hess` or is a spring mode, without an ORCA dependency.
- `xas-postprocess <batch>` runs the six postprocess steps on the login node;
  `--submit` submits them as a job after the batch's outstanding CORVUS jobs and
  replaces any older pending postprocess. Run inline, the auto-rerun step still
  triages and can submit CORVUS recomputes; `XAS_AUTO_RERUN=0` disables it.
