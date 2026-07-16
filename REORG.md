# automated-pipeline refactor — working document

Status/plan/decisions for the in-progress refactor of this repo into an
installable `xas_pipeline` package. Self-contained so a fresh session (or a
future me) can resume without re-deriving anything. Companion to the persistent
memory note `automated-pipeline-refactor-design` (loaded via MEMORY.md).

**Branch:** `refactor-package` (off `main`). `tests` was merged (ff) into `main`
as the stable checkpoint (test net + phases 1-5); phases 6+ continue on
`refactor-package`. Single developer/user; large changes are fine.
**Goal:** better-designed, testable package for: ORCA geometry optimization →
CORVUS/FEFF XANES/EXAFS spectra, Python-orchestrated, submitting to SLURM (or PBS).

---

## Current status

- **Suite: 68 tests, all green.** Run: `.venv/bin/python -m pytest -q`
- Package `xas_pipeline` (src/ layout, hatchling) editable-installed into `.venv`.
- Strategy: **characterization tests first, then reorg module-by-module staying green**,
  one commit per phase.

### Commits so far (newest last)
| Commit | Phase |
|---|---|
| `fe7f9e6` | Characterization test net (57 tests) before any refactor |
| `1da0f67` | Package skeleton + `pipeline_env.py`→`config.py`, `pipeline_batch_log.py`→`batch_log.py` |
| `68e5623` | `scheduler.py` — Scheduler ABC + Slurm/Pbs (killed 4× slurm/pbs duplication) |
| `565ebc1` | Characterization goldens for under-covered scripts (→68 tests) |
| `bd5aed7` | `layout.py` — SKIP_DIR_NAMES, iter_id_dirs, split/flat, quarantine_move; wired into 5 scripts |
| `7eb20b4` | (on main) Track uv.lock |
| `364a82c` | `templates.py` — fill/render placeholder engine; wired into 3 scripts; normalized fix #5 `[directory]`→`[DIRECTORY]` |
| `7b5161d` | `chem/{periodic,xyz,hessian,feff}.py` — pure parsers extracted; scripts keep old names as aliases |
| `269be28` | 8a: templates → `src/xas_pipeline/data/` (package data) + `resources.template_root()` |

---

## Target design (package)

```
src/xas_pipeline/
  config.py     DONE  .env loading (was pipeline_env.py)
  batch_log.py  DONE  batch-jobs.log outcomes (was pipeline_batch_log.py)
  scheduler.py  DONE  Scheduler ABC + SlurmScheduler/PbsScheduler
  layout.py     DONE  batch dir conventions (skip set, scan, split/flat, quarantine)
  templates.py  DONE  fill/render placeholder-fill engine
  chem/         DONE  periodic.py, xyz.py, hessian.py, feff.py (pure parsers/transforms)
  stages/       TODO  orca_prep, corvus_prep, orca_check, feff_process, download, cleanup
  orchestrate.py TODO run-batch core (JobRecord, dependency graph)
  cli/          TODO  thin argparse adapters -> console_scripts
```

### Transitional conventions (IMPORTANT for resuming)
- Top-level **hyphen scripts stay as the working entry points** and delegate into
  the package. They keep their old public names as thin bindings so the
  importlib-by-path consumers keep working. This is what keeps the CLI goldens
  (which invoke scripts by filename via subprocess) passing at every step.
- Each script that imports the package has a bootstrap line so it resolves whether
  or not installed (compute nodes):
  `sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))`
- `rerun-corvus.py` and `submit-corvus-only.py` still load `run-batch-pipeline.py`
  by importlib path; the hack is removed in the cli/ phase.

---

## Remaining phases (execute in order, suite green + commit after each)

6. ✅ DONE (`364a82c`) — templates.py fill/render engine; fix #5 casing normalized.
7. ✅ DONE (`7b5161d`) — chem/{periodic,xyz,hessian,feff}.py pure parsers. NOTE: the
   `.dym`/clean-xyz *writers* are I/O shells and stayed in prepare-corvus (move in 8).
8. **stages/ + orchestrate.py** — move stage logic into the package; top-level scripts
   become thin shims.
   - ✅ 8a DONE (`269be28`): templates relocated to `src/xas_pipeline/data/`, resolved via
     `resources.template_root()`. `.env` + sibling-script paths still repo-root (goldens intact).
   - ⬜ 8b: move each stage's logic into `stages/{orca_prep,corvus_prep,orca_check,feff_process,
     download,cleanup}.py`; top-level hyphen scripts become thin shims. **Constraint:** the shim
     must still expose the private names the importlib unit tests read (e.g. `_parse_submitted_job_id`,
     `orca_maxcore_mb`, `_atomic_number_from_token`) — do `globals().update(vars(module))` in the shim
     OR explicit re-exports, until phase 9 migrates tests to `xas_pipeline.*`. **Constraint:** `.env`
     path + sibling entry-point paths embedded in generated scripts must stay repo-root (inject from
     the shim via its own `__file__`) so goldens stay byte-identical until phase 10.
   - ⬜ 8c: run-batch core → `orchestrate.py`; run-batch-pipeline.py becomes a shim.
9. **cli/ → console_scripts** — RETIRE the hyphen scripts + importlib hack. Update
   `tests/conftest.py::load_script`/`run_script` and the subprocess golden invocations
   to the new entry points. Migrate `tests/unit/test_pure_core.py` off
   `rbp._parse_submitted_job_id` etc. to `xas_pipeline.*` imports.
10. **Apply the 8 fixes** as deliberate `GOLDEN_UPDATE=1` diffs (review each diff).

---

## The 8 known-issue decisions (agreed; apply in phase 10)

1. `submit-corvus-only.py` `--scheduler` default `slurm` → **FIX**: use `_default_scheduler()`
   (PIPELINE_SCHEDULER env → pbs) like the others.
2. batch-log vocab: `rerun-corvus` logs a *submission* as `SUCCEEDED` → **FIX**: use
   `SUBMITTED`/`SUBMIT_FAILED` everywhere; `submit-corvus-only` should log too (currently silent).
   NOTE: run-batch's `prepare-orca\tSUCCEEDED` is CORRECT (completed subprocess) — leave it.
   The affected submit-path lines aren't reachable in offline goldens → verify by inspection.
3. `failed-orca/` under `parent_dir` but `failed-corvus/` under `cwd` → **FIX**: both under batch root.
4. state file: run-batch text vs rerun-corvus JSON → **FIX**: plain-text everywhere.
5. placeholder casing `[DIRECTORY]` vs `[directory]` → normalize to `[UPPER_SNAKE]` (in phase 6).
6. dead code: `extract_h_bonded_atoms`, `_append_batch_job_log`'s unused `scheduler` param,
   count-imag-freq's unused `glob` import → delete.
7. `count-imag-freq.py` style outlier (raw sys.argv, os.path) → modernize (argparse/Path/main()->int).
8. **Copy-then-delete waste** (the one you flagged): `orca-job.script` does a blanket
   `cp !(*.in|*.inp|*.tmp*)` back from scratch, dragging in the exact files `cleanup` deletes.
   → **FIX (deny-list extend, shared list)**: one scratch-exclude pattern list in `config`, imported
   by BOTH the job-script generator (injected as a `[SCRATCH_EXCLUDE]` placeholder into the `cp`) and
   the cleanup stage. Exclude set = `*.densities *.densitiesinfo *.cpcm *.cpcm_corr *.engrad *.bin`
   (you confirmed ORCA `.bin` not needed). Also fix the PBS template not excluding `*.in`.
   Changes the `generated-*-orca.script` golden (deliberate).

---

## Test suite (tests/, run refactor-agnostically)

- `tests/conftest.py` — `load_script` (importlib-by-path for hyphen-named scripts) and
  `run_script`/`run_cli` (subprocess with venv on PATH). **These change in phase 9.**
- `tests/unit/` — `test_pipeline_env.py` (now imports `xas_pipeline.config`),
  `test_prepare_orca_helpers.py`, `test_pure_core.py` (scheduler job-id parse, ORCA sizing,
  corvus periodic table).
- `tests/cli/` — golden characterization, subprocess + `--dry-run`/`--no-submit`, path/timestamp-
  normalized snapshots, regenerable via `GOLDEN_UPDATE=1 .venv/bin/python -m pytest <file>`:
  run-batch (spine; also covers prepare-orca `ca-fixed`), prepare-corvus (up to `.dym`, stubs
  `dym2feffinp` via `DYM2FEFFINP_BIN`), process-feff (snapshots the larch chi(R) FFT; copy-fidelity
  + PNG existence for the rest), submit-corvus-only, rerun-corvus, orca-convergence-check, download,
  cleanup, count-imag-freq.
- Fixtures in `tests/fixtures/`: real inputs pulled from a completed run
  `calculations/clustering-validation/test-VIII/2j6a_ZN_homo_d2.60_cluster1`; goldens under
  `tests/fixtures/golden/` (force-tracked past the `*.log` gitignore via `!tests/fixtures/golden/**`).

### Coverage notes / gaps
- The `dym2feffinp` binary and real ORCA/FEFF/Corvus runs + real `sbatch`/`qsub` can't run here,
  so goldens stop at "correct files generated" / "correct submit command would issue".
- Deferred (optional): prepare-orca **mode-matrix** golden (`--H`/`--free`/`--backbone`/`--xtb-*`;
  `--backbone` needs a `.pc` fixture).

---

## How to resume in a fresh session
1. `cd automated-pipeline`; `.venv/bin/python -m pytest -q` → expect 68 passing.
   Branch: `refactor-package`.
2. Read this file + the `automated-pipeline-refactor-design` memory note.
3. Pick up at phase 8b (stage modules + shims); 8a is done. Keep the suite green; commit per step.
   Decision (2026-07-16): templates live as PACKAGE DATA under src/xas_pipeline/data/.
