# ORCA and CORVUS pipeline automation

This directory contains template inputs and helper scripts to run ORCA geometry optimizations and then prepare CORVUS/FEFF inputs. The typical flow is:

1) `prepare-orca.py` → run ORCA geometry optimization
2) `prepare-corvus.py` → convert ORCA output to FEFF inputs and set up CORVUS (EXAFS and XANES)
3) `script-process-feff-output.py` → postprocess FEFF output (plots + chi(R) per CORVUS mode)

The batch driver `run-batch-pipeline.py` submits ORCA → CORVUS → a final postprocess
job. The postprocess job depends `afterany` on the CORVUS jobs (it runs once they all
finish, regardless of success), and the postprocess scripts handle failures themselves:
failed ORCA runs are moved to `failed-orca/`, failed CORVUS runs to `failed-corvus/`, and
only surviving jobs are copied to `downloading-station/`.

## Scripts

`prepare-orca.py`
- Copies ORCA input templates from `orca-templates/` and fills placeholders
- Cleans XYZ files and writes sidecar comments
- Generates ORCA input files per structure
- Generates `generated-<name>-orca.script` from `<scheduler>-scripts/orca-job.script`
- Use `--dry-run` to skip job submission
- Use `--scheduler {pbs,slurm}` to select scheduler templates and submit command
- Key args: `path`, `--out-dir`, mode flags, `--dry-run`
- Reads charge/multiplicity from XYZ header line 2 using:
	- `CHARGE_ROUNDED=<int>` or `ROUNDED_CHARGE=<int>`
	- `MULTIPLICITY=<int>`

`prepare-corvus.py`
- Requires ORCA `.hess` output
- Converts `.hess` → `.dym`, then runs `dym2feffinp` to build FEFF inputs
- Processes both EXAFS and XANES Corvus templates by default (generates separate mode-specific inputs and job scripts)
- Writes mode-specific files such as `corvus-<run_id>-<mode>.in` and `corvus-job-<mode>.script`
- Use `--scheduler {pbs,slurm}` to select job-script template
- Use `--corvus-mode {both,exafs,xanes}` to select which templates to use (default: both)
- Key args: `path`, `--corvus-mode`

`script-process-feff-output.py`
- Processes every CORVUS mode present per id (`Corvus3_cfavg_{xanes,exafs,xas}/Corvus1Zn_FEFF`)
- Plots XANES/EXAFS and converts chi(k) to R space via Larch
- Copies into `output-<id>`: `xmu-<mode>-<id>.dat`, `chi-R-<id>.dat`, the cfavg spectra
  `xanes-<id>.dat`/`exafs-<id>.dat`, and `<id>.xyz` (no `dw.dat`)
- An id with any failed/missing CORVUS mode is recorded in `corvus-failed-ids.txt`
- Key args: `parent_dir`, `--recursive`

`script-check-orca-convergence-and-extract-times.py`
- Checks each run dir for ORCA convergence/normal termination
- Moves failed ORCA runs into `failed-orca/`
- Writes TOTAL RUN TIME + Final Gibbs free energy for survivors to a CSV, plus a report

`script-prepare-files-for-download.py`
- Moves CORVUS-failed ids (from `corvus-failed-ids.txt`) into `failed-corvus/`
- Copies surviving ids' `output-*` dirs into the download destination
  (default: `./downloading-station` in the current working directory)

Additional helper scripts with `script-` prefix are included for reporting and packaging:
- `script-count-imag-freq.py`

## Template Inventory
ORCA input templates in `orca-templates/`:
- `orca-template-ca-fixed.in`
- `orca-template-h-only.in`
- `orca-template-single-point.in`
- `orca-template-no-constraints.in`
- `orca-template-backbone-charges.in`
- `orca-template-xtb-free.in`
- `orca-template-xtb-constrained.in`
- `orca-template-quick.in` (available, not selected by default CLI mode flags)
- `orca-template-ca-fixed-p450.in` (available, not selected by default CLI mode flags)

Scheduler job script templates:
- `pbs-scripts/orca-job.script`
- `pbs-scripts/corvus-job.script`
- `pbs-scripts/corvus-wrapper.script`
- `pbs-scripts/postprocess-job.script`
- `slurm-scripts/` (placeholder directory for future Slurm templates)

CORVUS input templates:
- `corvus-template-exafs.in`
- `corvus-template-xanes.in`

## Quick examples
```bash
python prepare-orca.py /path/to/xyz --out-dir /path/to/output
python prepare-corvus.py /path/to/orca/output               # both EXAFS + XANES (default)
python prepare-corvus.py /path/to/orca/output --corvus-mode exafs   # only EXAFS
python prepare-corvus.py /path/to/orca/output --corvus-mode xanes   # only XANES
python script-process-feff-output.py /path/to/batch --recursive
```