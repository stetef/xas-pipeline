# xas-pipeline

Batch pipeline for computed Zn K-edge X-ray absorption spectra: ORCA geometry
optimization and Hessian, then CORVUS/FEFF for XANES and EXAFS, submitted to Slurm
(S3DF) or PBS and post-processed into plots and data tables.

For each XYZ structure, `xas-run-batch` writes a run directory and submits a chain of
jobs. ORCA optimizes the geometry and, in most modes, computes the analytic Hessian
(`! AnFreq`). In the spring modes the Hessian is interpolated from ligand spring
models instead, with or without an ORCA stage. The CORVUS job turns the Hessian into
a dynamical matrix, runs FEFF (Debye-Waller factors via DMDW) and writes one
combined XANES + EXAFS spectrum. A single postprocess job per batch then checks ORCA
convergence, plots the spectra, computes χ(R), cleans up scratch, recomputes dead
XANES legs and stages the results for download. A watchdog job resubmits jobs that
hang.

```
 XYZ ─► ORCA job ──afterok──► CORVUS job ──afterany (all CORVUS jobs)──► postprocess job
        opt (+AnFreq)          [spring Hessian]          orca-check → process-feff → cleanup
        (none in no-ORCA       prepare-corvus            → auto-rerun-corvus → download → stage-times
         spring modes)         run-corvus
                                                         watchdog job (whole batch, hang recovery)
```

## Install

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/). Clone the repository
somewhere on a shared file system: the generated job scripts use this checkout's
`.venv/` and `.env` on the compute nodes.

```bash
git clone <repo-url> automated-pipeline && cd automated-pipeline
uv sync --extra exafs         # .venv/ with an editable install, the xas-* commands and xraylarch
```

The `exafs` extra (xraylarch) does the χ(k) → χ(R) transform in the postprocess.
Without it the postprocess still runs, but skips χ(R) with a warning.

CORVUS is not on PyPI and is not in `uv.lock`. Install it into the same environment
from git, and use `uv sync --inexact` from then on (a plain `uv sync` removes
packages it does not manage):

```bash
uv pip install "corvus @ git+https://github.com/times-software/Corvus.git"
```

FEFF10 (with `dym2feffinp`) is built from source and configured in CORVUS's
`~/.Corvus/corvus.conf`. See [docs/feff10-build.md](docs/feff10-build.md) for both
steps. ORCA is used as installed on the cluster.

## Site configuration

Site paths live in `.env` at the repository root (git-ignored). Start from the
example and fill in your paths:

```bash
cp .env.example .env
```

| Variable | Used by | Purpose |
|---|---|---|
| `PIPELINE_ORCA_HOME` | ORCA job | ORCA install directory |
| `PIPELINE_OPENMPI_ROOT`, `PIPELINE_HPCX_ROOT`, `PIPELINE_GCC_ROOT` | Slurm job scripts | MPI and compiler runtime |
| `DYM2FEFFINP_BIN` | prepare-corvus | FEFF's `dym2feffinp` (else `/opt/feff10`, else `PATH`, else CORVUS's Python writer) |
| `CORVUS_RUNNER` | CORVUS job | `run-corvus` to use (else searched in the pipeline `.venv`, `~/.venv`, the `corvusenv` conda env and `PATH`) |
| `PIPELINE_SCRATCH_ROOT` | PBS job scripts | scratch root |
| `PIPELINE_SCHEDULER` | every command with `--scheduler` | `slurm` or `pbs` (default `pbs`) |
| `PIPELINE_PYTHON` | every job script | Python to run the pipeline with (default: the interpreter that wrote the script) |

Notes:

- `.env` is plain `KEY=VALUE`. Every generated job script sources it, on both
  schedulers; values already set in the environment win. Anything unset falls back
  to the default written in the job-script template.
- On the login node, every command with a `--scheduler` option reads `.env` first,
  so `PIPELINE_SCHEDULER=slurm` there makes Slurm the default; `--scheduler` or an
  exported variable overrides it. The site paths are read by the job scripts and by
  `xas-prepare-corvus`.
- The Slurm templates request `--partition=milano --account=ssrl:SMBXAS`. To use
  another account or partition, edit the templates in
  `src/xas_pipeline/data/slurm-scripts/`.
- The job scripts find `.env` and `.venv` through the pipeline root, which is the
  checkout for an editable install. Set `PIPELINE_ROOT` to override it.

## Input structures

One `.xyz` per structure; the file stem becomes the structure id. Line 2 must carry
the charge and multiplicity, and the atom lines may carry `#` tags that the
constrained modes read:

```
47
id=4j1v_cluster2_Zn CHARGE=0 MULTIPLICITY=1
ZN 0.0514950 0.0259379 0.0441127  # RES=ZN CHAIN=C RESSEQ=301 ATOM=ZN
S -0.5119776 -2.0802855 -0.6682717  # RES=CYS CHAIN=C RESSEQ=79 ATOM=SG COORD=TRUE
C -2.0891650 -2.4885360 1.6156870  # RES=CYS CHAIN=C RESSEQ=79 ATOM=CA COORD=FALSE
...
```

See [docs/charge-multiplicity.md](docs/charge-multiplicity.md) for the exact rules.

## Quickstart (S3DF)

```bash
export PIPELINE_SCHEDULER=slurm

# Write every input and job script, submit nothing; inspect the result:
xas-run-batch /path/to/xyz_dir --out-dir /path/to/batch --no-submit

# Submit the batch (default mode: caopt-anfreq):
xas-run-batch /path/to/xyz_dir --out-dir /path/to/batch

# Add a second mode to the same batch; it lands in its own run dirs:
xas-run-batch /path/to/xyz_dir --out-dir /path/to/batch --interp
```

`xas-run-batch` prints the job ids and writes
`<batch>/pipeline-state-<batch>.log` (what was submitted) and
`<batch>/batch-jobs.log` (every submission and outcome, appended by every later
command). When the postprocess job has run, the spectra are in
`<batch>/downloading-station/`.

Without `--out-dir`, the batch root is the parent of the XYZ directory.

## Modes

The mode decides how the geometry is prepared and where the Hessian comes from. Its
name is the suffix of every run directory (`<id>-<mode>`). The geometry/Hessian
modes are named `<geometry>-<hessian>`: geometry `caopt` (CA atoms fixed), `hopt`
(hydrogens only), `carved` (as carved, single point) or `asis` (as handed in);
Hessian `anfreq` (ORCA analytic) or `spring` (interpolated from ligand spring
models).

| Mode | Flag | ORCA template | Hessian for FEFF |
|---|---|---|---|
| `caopt-anfreq` | (default) | `orca-template-caopt-anfreq.in` | ORCA AnFreq |
| `quick` | `--quick` | `orca-template-quick.in` | none (no `.hess`) |
| `quick-ca-fixed` | `--quick-ca-fixed` | `orca-template-quick-ca-fixed.in` | none (no `.hess`) |
| `hopt-anfreq` | `--H` | `orca-template-hopt-anfreq.in` | ORCA AnFreq |
| `carved-anfreq` | `--single` | `orca-template-carved-anfreq.in` | ORCA AnFreq |
| `no-constraints` | `--free` | `orca-template-no-constraints.in` | ORCA AnFreq |
| `backbone` | `--backbone` | `orca-template-backbone-charges.in` | ORCA AnFreq |
| `xtb-free` | `--xtb-free` | `orca-template-xtb-free.in` | ORCA AnFreq |
| `xtb-constrained` | `--xtb-constrained` | `orca-template-xtb-constrained.in` | none (no `.hess`) |
| `carved-spring` | `--mode` only | `orca-template-carved-spring.in` | spring model |
| `hopt-spring` | `--interp` | `orca-template-hopt-spring.in` | spring model |
| `asis-spring` | `--interp-raw` | none (no ORCA job) | spring model |
| `caopt-spring` | `--caopt-spring` | none (no ORCA job) | spring model |

Every mode can also be selected by name with `--mode NAME`. Modes marked "none (no
`.hess`)" are geometry-only: their CORVUS job stops at prepare-corvus with a
missing-Hessian error. `backbone` needs a `<stem>.pc` point-charge file next to each
XYZ. The ORCA templates are in `src/xas_pipeline/data/orca-templates/`.

The table is generated from `src/xas_pipeline/modes.py`, the single registry every
command reads. Staged runs (`--pre`), the AnFreq toggle, the spring Hessian and
its ligand database are described in
[docs/modes-and-hessians.md](docs/modes-and-hessians.md).

## Commands

Each command has `--help`. Each is also `python -m <module>`, which is what the
generated job scripts call.

| Command | Module | What it does |
|---|---|---|
| `xas-run-batch` | `xas_pipeline.orchestrate` | Prepare and submit a whole batch for one mode: ORCA, CORVUS, postprocess, watchdog |
| `xas-prepare-orca` | `xas_pipeline.stages.orca_prep` | Write ORCA run dirs and submit the ORCA jobs only (logged in `batch-jobs.log`); `--pre` stages, `--anfreq`/`--no-anfreq` |
| `xas-prepare-corvus` | `xas_pipeline.stages.corvus_prep` | In one run dir: `.hess` → `.dym` → CORVUS input and job script |
| `xas-interp-hessian` | `xas_pipeline.stages.interp_hessian` | In one run dir: write `<id>.hess` from ligand spring models |
| `xas-build-interp` | `xas_pipeline.cli.build_interp` | Build one ligand's `.interp` model from two ORCA `.hess` files |
| `xas-orca-check` | `xas_pipeline.stages.orca_check` | Classify ORCA runs, move failures to `failed-orca/`, write the timing CSV |
| `xas-process-feff` | `xas_pipeline.stages.feff_process` | Check spectra, plot them, compute χ(R), fill `output-<id>/` |
| `xas-cleanup` | `xas_pipeline.stages.cleanup` | Delete regenerable scratch (dry run unless `--execute`) |
| `xas-auto-rerun-corvus` | `xas_pipeline.cli.auto_rerun_corvus` | Recompute runs whose XANES leg came back dead |
| `xas-download` | `xas_pipeline.stages.download` | Copy surviving `output-*` dirs to the download dir; move CORVUS failures to `failed-corvus/` |
| `xas-stage-times` | `xas_pipeline.stages.stage_times` | Write `pipeline-stage-times.csv`: ORCA and CORVUS wall time and exit code per run |
| `xas-postprocess` | `xas_pipeline.cli.postprocess` | Run the six steps above here, or `--submit` them as a job. Run here, the auto-rerun step can still submit CORVUS recomputes (`XAS_AUTO_RERUN=0` disables it) |
| `xas-submit-corvus` | `xas_pipeline.cli.submit_corvus` | Submit CORVUS only, for runs whose ORCA is done |
| `xas-rerun-corvus` | `xas_pipeline.cli.rerun_corvus` | Archive and recompute CORVUS for a finished batch (e.g. after editing a CORVUS template) |
| `xas-rerun-orca` | `xas_pipeline.cli.rerun_orca` | Diagnose one failed ORCA run and resubmit it if a remedy applies |
| `xas-watchdog` | `xas_pipeline.cli.watchdog` | Find hung ORCA/CORVUS jobs in a batch and resubmit them |
| `xas-count-imag-freq` | `xas_pipeline.stages.count_imag_freq` | Count CORVUS imaginary-frequency warnings per run |

Options worth knowing:

- `--no-submit` (alias `--dry-run`) writes everything and submits nothing in
  `xas-run-batch`, `xas-submit-corvus`, `xas-rerun-corvus`, `xas-auto-rerun-corvus`,
  `xas-rerun-orca` and `xas-postprocess --submit`. `xas-prepare-orca` uses
  `-n`/`--dry-run`; without it, it submits. `xas-run-batch` and `xas-submit-corvus`
  record each job they did not submit as a `SKIPPED` line in `batch-jobs.log`.
- `xas-run-batch` postprocess switches: `--skip-extract` (no ORCA check),
  `--skip-process-feff` (no spectra and no CORVUS auto-rerun), `--skip-cleanup`
  (keep FEFF scratch and rerun snapshots, useful when diagnosing a spectrum),
  `--skip-prepare-download`, `--no-postprocess`, `--download-destination DIR`.
- Watchdog switches: `--no-watchdog`, `--watchdog-interval` (600 s),
  `--watchdog-idle-threshold` (3600 s), `--watchdog-max-reruns` (2),
  `--watchdog-walltime` (168:00:00).
- `--ids a,b` on `xas-submit-corvus` and `xas-rerun-corvus` takes run ids
  (`<id>-<mode>`) or structure ids (all modes of that structure).

## Where results land

```
batch/
  batch-jobs.log                     every submission and outcome
  pipeline-state-batch.log           what the last xas-run-batch submitted
  generated-postprocess-batch.script, generated-watchdog-batch.script
  2j6a_cluster1/                     one group dir per structure
    2j6a_cluster1-caopt-anfreq/      one run dir per mode; its name is the run id
      working-2j6a_cluster1-caopt-anfreq/   calculation files (after postprocess)
      output-2j6a_cluster1-caopt-anfreq/    deliverables (after postprocess)
    2j6a_cluster1-hopt-spring/
  <batch>-orca-compute-times.csv     ORCA run time and Gibbs energy per converged run
  orca-convergence-report.log        one line per ORCA run: OK, FAILED (why), SKIPPED
  pipeline-stage-times.csv           ORCA and CORVUS wall time and exit code per run
  corvus-failed-ids.txt              run ids whose spectrum failed the check
  failed-orca/  failed-corvus/       quarantined run dirs
  downloading-station/               copies of every surviving output-<run id>/, grouped as above
```

`output-<run id>/` holds `xas-<run id>.dat` (the combined spectrum, columns: photon
energy, photoelectron energy, k, μ, μ₀, χ), `xmu-xanes-…dat`, `xmu-exafs-…dat`,
`chi-exafs-…dat`, `chi-R-…dat` (|χ(R)|, Re, Im), the PNG plots
(`xanes-`, `exafs-`, `chi-R-`) and `<run id>.xyz`, the geometry the spectrum was
computed at.

The postprocess job waits (`afterany`) for every CORVUS job the batch has
outstanding, including those of modes submitted earlier, and replaces any older
postprocess job that is still queued or running. Runs whose ORCA job is still
queued or running are skipped by the ORCA check, not quarantined. Details:
[docs/batch-layout.md](docs/batch-layout.md).

## Automatic recovery

All automatic reruns are bounded (two per structure by default), recorded per run
in `<run id>-*rerun-state.json`, and end with a `NEEDS_HUMAN` line in
`batch-jobs.log` when they run out. Set `XAS_AUTO_RERUN=0` in the job environment to
turn off the ORCA and CORVUS reruns.

- **Failed ORCA job.** The ORCA job script runs `xas-rerun-orca` when ORCA does
  not terminate normally. It cancels the CORVUS job waiting on it and, for an
  out-of-memory failure (more `%MaxCore` and job memory) or an optimization that
  ran out of cycles (restart from the last geometry), resubmits ORCA and a new
  CORVUS job and re-gates a pending postprocess onto it. Everything else, including SCF non-convergence and
  charge/multiplicity errors, is escalated to a human.
- **Hung job.** The watchdog job polls the batch's running jobs. A job whose log
  has been silent past the idle threshold while its CPU time stays flat is
  cancelled and resubmitted, on another node under Slurm.
- **Dead XANES leg.** A CORVUS run can finish with a XANES χ that is zero on every
  row. The postprocess check fails it, and `xas-auto-rerun-corvus` archives the
  output and recomputes the same inputs, because this failure is sporadic. Other
  CORVUS failures are left for a human.

Details, diagnosis tables and state files: [docs/recovery.md](docs/recovery.md).

## Use as a library

The commands are thin wrappers around importable functions. This runs from the
repository root against the test fixtures:

```python
import shutil, tempfile
from pathlib import Path

from xas_pipeline import corvus_diagnosis, modes
from xas_pipeline.chem import hessian, xyz
from xas_pipeline.orchestrate import submit_batch
from xas_pipeline.scheduler import get_scheduler
from xas_pipeline.stages.interp_hessian import build_interp_hessian
from xas_pipeline.stages.orca_prep import discover_xyz_files

fixtures = Path("tests/fixtures")
run_id = "2j6a_ZN_homo_d2.60_cluster1"

print(modes.get("hopt-spring").flag)                      # --interp
print(xyz.extract_charge_multiplicity(fixtures / "xyz_files" / f"{run_id}.xyz"))  # (-2, 1)

# ORCA's analytic Hessian, as the CORVUS stage reads it.
orca = hessian.read_hess(fixtures / "corvus_run" / run_id / f"{run_id}.hess")
print(orca.n_atoms, orca.matrix.shape)                     # 33 (99, 99)

# A spring-model Hessian for the same geometry, in a scratch copy of the run dir.
run_dir = Path(tempfile.mkdtemp()) / run_id
shutil.copytree(fixtures / "corvus_run" / run_id, run_dir)
spring = build_interp_hessian(run_dir, run_id)             # overwrites <run_id>.hess
print(spring.n_pairs, spring.n_imaginary)                  # 528 0

# The verdict the postprocess check gives a finished CORVUS run.
verdict = corvus_diagnosis.diagnose_run_dir(fixtures / "feff_run" / "5a3w_ZN_homo_d2.60_cluster2")
print(verdict.kind.value, verdict.auto_remediable)         # ok False

# A whole batch, prepared but not submitted (no scheduler needed).
xyz_files, _ = discover_xyz_files(fixtures / "xyz_files")
batch = submit_batch(xyz_files, Path(tempfile.mkdtemp()) / "batch", "hopt-spring",
                     get_scheduler("slurm"), submit=False)
print([run.run_id for run in batch.runs])
```

Where things live: `xas_pipeline.chem` holds the file formats and numerics (XYZ,
`.hess`, FEFF tables and χ(R), spring models) with no scheduler logic;
`xas_pipeline.stages` one module per pipeline step; `xas_pipeline.cli` the batch
commands; `modes`, `layout`, `scheduler`, `batch_log`, `jobscripts`, `diagnosis`,
`remedy`, `corvus_diagnosis`, `recovery` and `rerun_state` the shared machinery.
Job-script templates, ORCA and CORVUS input templates and the packaged ligand
models are package data under `src/xas_pipeline/data/`.

## Development

```bash
uv sync --group dev             # pytest, ruff and xraylarch
.venv/bin/python -m pytest      # whole suite; -m "not slow" skips end-to-end runs
.venv/bin/ruff check src tests
```

The tests do not run ORCA, CORVUS, FEFF or a scheduler. Golden tests snapshot the
generated inputs and job scripts; after an intended change, regenerate them with
`GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/cli/<file>` and review the diff.

Changes that might touch physics are checked with the oracle in
`/sdf/data/ssrl/smb/dft/users/tetef01/xas-pipeline-cleanup-2026-10/`. It runs every
entry point that writes inputs or numbers (all modes, both schedulers, staged
runs, spring Hessians, prepare-corvus, FEFF post-processing, `.interp` building)
from a source tree and normalises paths and timestamps:

```bash
O=/sdf/data/ssrl/smb/dft/users/tetef01/xas-pipeline-cleanup-2026-10
TMPDIR=/path/to/scratch $O/run_oracle.sh "$PWD" /path/to/scratch/oracle-new
$O/compare.sh $O/oracle-ecf530b /path/to/scratch/oracle-new
```

The first section of `compare.sh` output (output files) must be empty, or every
difference explained. Console text may differ.

## License

MIT; see [LICENSE](LICENSE).
