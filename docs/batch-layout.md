# Batch layout and the postprocess

## Directories

```
batch/                                   the batch root (--out-dir)
  batch-jobs.log
  2j6a_cluster1/                         group dir: one per structure
    2j6a_cluster1-caopt-anfreq/          run dir: one per mode; its name is the run id
    2j6a_cluster1-hopt-spring/
  failed-orca/  failed-corvus/  downloading-station/
```

A run dir starts flat (everything in it). The first `xas-process-feff` pass splits
it into `working-<run id>/` (the calculation) and `output-<run id>/` (the
deliverables). Every stage handles both forms, and the bare `<id>/` run dirs of
older batches; see [modes-and-hessians.md](modes-and-hessians.md#naming-and-older-run-dirs).

`xyz_files/`, `optimized_xyz_files/`, `failed-orca/`, `failed-corvus/` and
`downloading-station/` under the batch root are never scanned for runs.

## Files in a run dir

| File | Written by | Content |
|---|---|---|
| `<id>.xyz` (input copy), `<run id>_clean.xyz`, `<run id>_comments.txt` | prepare-orca | input geometry, bare coordinates, atom tags |
| `<run id>.in`, `generated-<run id>-orca.script` | prepare-orca | ORCA input and job script (ORCA modes) |
| `<run id>-orca.log`, `<run id>-orca.timing` | ORCA job | ORCA output; start time, host, end time, exit code, normal termination |
| `<run id>.xyz`, `<run id>.hess`, `<run id>.gbw`, ... | ORCA job | optimized geometry (the input geometry in no-ORCA modes), Hessian, orbitals |
| `generated-<run id>-corvus-xas-wrapper.script` | xas-run-batch and every rerun command | the CORVUS job |
| `spring.model`, `interp-ligands/` | spring modes | interpolated constants; per-run ligand database |
| `corvus-begin-<run id>.xyz`, `<run id>.dym`, `corvus-<run id>.dym` | prepare-corvus | CORVUS geometry; dynamical matrix, and centred on the absorber |
| `corvus-<run id>-xas.in`, `xanes.in`, `exafs.in`, `corvus-job-xas.script` | prepare-corvus | CORVUS inputs and the script the wrapper runs |
| `Corvus.cfavg_xas.out`, `Corvus3_cfavg_xas/` | CORVUS | combined spectrum; FEFF work tree (`Corvus1Zn_<n>_FEFF/{xanes,exafs}/`) |
| `<run id>-corvus-xas.timing`, `corvus-<run id>-xas.out` | CORVUS job | timing; CORVUS log |
| `*-rerun-state.json`, `<run id>-rerun-history/`, `<run id>-rerun.log` | recovery | see [recovery.md](recovery.md) |

## Batch-level files

- `batch-jobs.log`: one tab-separated line per submission (`SUBMITTED`,
  `SUBMIT_FAILED`, `SKIPPED`, `REPLACED`, `CANCELLED`) with its job id, plus outcome
  blocks the postprocess and recovery steps append (`OK`, `FAILED`, `NEEDS_HUMAN`
  with a reason). The status column of a submission line is the submission result,
  not the computational outcome. Every command that submits appends to it, and it
  is how a later invocation finds the batch's outstanding jobs.
- `pipeline-state-<batch>.log`: what the last `xas-run-batch` submitted (job ids
  per run). `--state-file` writes it elsewhere.
- `<batch>-orca-compute-times.csv` and `orca-convergence-report.log`: from the ORCA
  check, written to the batch root by the postprocess job (to the current directory
  when `xas-orca-check` is run without `--output-dir`).
- `corvus-failed-ids.txt`: run ids whose spectrum failed the check; read by the
  CORVUS auto-rerun and the download step.
- `pipeline-stage-times.csv`: one row per run: ORCA and CORVUS `elapsed_seconds`
  and `exit_code` from the `.timing` files (blank when missing).
- `generated-postprocess-<batch>.script`, `generated-watchdog-<batch>.script`: the
  batch jobs.

## The postprocess job

One postprocess job per batch runs, in order:

1. `xas-orca-check <batch> --output-dir <batch>`: classify ORCA runs, move those
   that did not terminate normally (or did not converge) into `failed-orca/`.
   Runs whose ORCA job is still queued or running are skipped. Run dirs without an
   ORCA log or job script (the no-ORCA modes) are not considered.
2. `xas-process-feff <batch> --recursive`: for each run with CORVUS output, check
   the spectrum, plot XANES and EXAFS, compute χ(R) (k 3–11 Å⁻¹, dk 3, k-weight 2,
   Kaiser window, R up to 6 Å), split the run dir into `working-`/`output-`, and
   write `corvus-failed-ids.txt`. Without xraylarch, χ(R) is skipped with a
   warning and the rest proceeds.
3. `xas-cleanup <batch> --execute`: delete regenerable scratch (ORCA `.densities`,
   `.cpcm`, `.engrad`, `.bin`; FEFF `dmdw.out`, `*.bin`, `gg.dat`), `.rerun-`
   snapshots, and the per-component CORVUS trees of older runs once a combined
   result exists. Logs, `.gbw`, `.hess`, `.dym`, inputs, `xmu.dat` and `chi.dat`
   are always kept. Run dirs without an `output-*` dir are skipped.
4. `xas-auto-rerun-corvus`: recompute dead XANES legs (see
   [recovery.md](recovery.md)).
5. `xas-download <batch> -d <batch>/downloading-station`: move runs listed in
   `corvus-failed-ids.txt` into `failed-corvus/`, copy every other run's `output-*`
   dir to the destination at the same relative path. Existing copies are skipped
   unless `--refresh`.
6. `xas-stage-times <batch>`: write `pipeline-stage-times.csv`. Always runs; the
   `--skip-*` switches do not remove it.

In the job, a step that exits non-zero ends the job (`set -e`), except the
auto-rerun step, whose failure only prints a warning. Run inline with
`xas-postprocess <batch>`, a failing step does not stop the later ones.

### Gating

The postprocess job waits `afterany` on every CORVUS job the batch has
outstanding, read from `batch-jobs.log` and checked against the scheduler, so it
runs once for the whole batch rather than once per submission:

- Submitting a second mode into a batch makes the new postprocess also wait for the
  first mode's CORVUS jobs, and replaces (cancels) the earlier postprocess, queued
  or running, which was gated only on its own jobs. The new job is submitted
  before the old one is cancelled. Job ids the scheduler no longer knows are dropped, so adding a mode to
  an old batch does not wait on jobs that no longer exist.
- `--no-postprocess` submits none. Later, `xas-postprocess <batch> --submit` submits
  one gated the same way (or on `--after ID,ID`), and `xas-postprocess <batch>` runs
  the steps on the login node.
- The `--skip-*` switches of `xas-run-batch` remove steps from the job it writes;
  they do not affect `xas-postprocess`.
- `xas-rerun-orca` and the watchdog re-gate a pending postprocess onto the jobs
  they resubmit.

Every job script runs Python as `PYTHON_BIN="${PIPELINE_PYTHON:-<interpreter that
wrote the script>}"` and exits 127 if that interpreter is gone, so regenerate the
scripts (or set `PIPELINE_PYTHON`) after moving the virtualenv. A template
placeholder left unfilled stops the script from being written.
