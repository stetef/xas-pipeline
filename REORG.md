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

- **REFACTOR COMPLETE (phases 1-10).** Suite: **69 tests, all green.** Run: `.venv/bin/python -m pytest -q`
- Package `xas_pipeline` (src/ layout, hatchling) editable-installed into `.venv`; entry points are
  `xas-*` console_scripts (+ `python -m xas_pipeline...`). No top-level hyphen scripts remain.
- Strategy used: **characterization tests first, then reorg module-by-module staying green**,
  one commit per phase; the 8 behavior fixes applied last as reviewed GOLDEN_UPDATE diffs.

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
| `8164850` | 8b: cleanup stage → package (establishes thin-shim + `globals().update` re-export pattern) |
| `240a3c3` | 8b: download/orca_check/feff_process stages → package |
| `62fe20e` | 8b: orca_prep/corvus_prep stages → package (`.env` via `resources.project_root()`) |
| `0be764d`–`f953a28` | 9a-9d: `xas-*` console_scripts, `python -m`, retire shims, rerun/submit → `cli/` |
| `efd4200` | 10 fix #1: submit-corvus `--scheduler` default → `_default_scheduler()` |
| `b5d2cf0` | 10 fix #3: `failed-corvus/` anchored at batch root |
| `b1ae430` | 10 fix #4: plain-text rerun state (was JSON) |
| `207ff99` | 10 fix #2: batch-log `SUBMITTED`/`SUBMIT_FAILED` vocab; submit-corvus now logs |
| `249c965` | 10 fix #6: delete dead code (`extract_h_bonded_atoms`, unused `scheduler` param) |
| `f781ba7` | 10 fix #7: modernize count-imag-freq → `stages/count_imag_freq.py` |
| `e787ae5` | 10 fix #8: shared `SCRATCH_EXCLUDE_GLOBS`; stop copy-then-delete; PBS `*.in` |
| `50ee4c0` | 8c: run-batch core → `orchestrate.py`; run-batch-pipeline.py → shim |
| `0be764d` | 9a: drop importlib hack in rerun/submit; add `xas-*` console_scripts |
| `0faa7b1` | 9b: unit tests import `xas_pipeline.*` directly (no load_script) |
| `7737bbe` | 9c: generated scripts + orchestrator invoke `python -m` (goldens updated) |
| `f953a28` | 9d: retire 7 hyphen shims; rerun/submit → `cli/`; harness on `python -m` |

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
  resources.py  DONE  template_root() (package data) + project_root() (transitional repo root)
  stages/       DONE  orca_prep, corvus_prep, orca_check, feff_process, download, cleanup
  orchestrate.py DONE run-batch core (JobRecord, dependency graph)
  cli/          DONE  rerun_corvus.py, submit_corvus.py (console_scripts + `python -m`)
```

### Conventions after phase 9 (IMPORTANT for resuming)
- **No more top-level hyphen scripts** except `script-count-imag-freq.py` (retired in
  phase 10 / fix #7). Human entry points are `xas-*` console_scripts (see
  `[project.scripts]`); internal machinery + tests invoke `python -m xas_pipeline...`.
- The generated compute-node scripts invoke `python -m xas_pipeline.stages.<stage>`;
  the venv/`.env` anchor is injected as `[PIPELINE_ROOT]` = `resources.project_root()`.
- `resources.project_root()` (= `parents[2]` of the package dir) is KEPT, not retired:
  it still supplies the repo-root `.env` path + the `PIPELINE_ROOT` compute-node anchor.
  It is only valid in the run-from-checkout / editable layout (there is a `.venv` + `.env`
  at the repo root); a bare wheel install on a node without a checkout would need a
  different anchor (out of scope — the pipeline runs from a checkout).
- Tests: `tests/conftest.py::_SCRIPT_MODULES` maps historical filenames → modules so
  `run_script("prepare-corvus.py", ...)` routes to `python -m`. `load_script` is gone.

---

## Remaining phases (execute in order, suite green + commit after each)

6. ✅ DONE (`364a82c`) — templates.py fill/render engine; fix #5 casing normalized.
7. ✅ DONE (`7b5161d`) — chem/{periodic,xyz,hessian,feff}.py pure parsers. NOTE: the
   `.dym`/clean-xyz *writers* are I/O shells and stayed in prepare-corvus (move in 8).
8. ✅ **DONE** — stage logic moved into the package; all 7 top-level scripts are thin shims.
   - 8a (`269be28`): templates → package data, `resources.template_root()`.
   - 8b (`8164850`,`240a3c3`,`62fe20e`): 6 stage modules under `stages/`; shims re-export names via
     `globals().update({k:v for k,v in vars(module).items() if not k.startswith("__")})` so the
     importlib unit tests + rerun/submit consumers keep resolving privates.
   - 8c (`50ee4c0`): `orchestrate.py` (run-batch core); shim also importlib-loaded by rerun/submit.
   - Repo-root resolution (for embedded `.env` + sibling `*.py` paths) centralized in the transitional
     `resources.project_root()` (= `src/xas_pipeline` → `src` → repo), keeping generated output
     byte-identical. NOTE: `rerun-corvus.py`, `submit-corvus-only.py`, `script-count-imag-freq.py` are
     NOT shims yet — they fold into cli/ (phase 9) and fix #7 (phase 10).
9. ✅ **DONE** — console_scripts + `python -m`; hyphen scripts + importlib hack retired.
   - 9a (`0be764d`): rerun/submit `from xas_pipeline import orchestrate`; `xas-*` console_scripts.
   - 9b (`0faa7b1`): unit tests import `xas_pipeline.*` (chem.periodic public names).
   - 9c (`7737bbe`): generated wrapper/postprocess + orchestrator ORCA-prep → `python -m`
     (deliberate golden update: wrapper `PREP_CORVUS`→`PIPELINE_ROOT`, `python -m` lines).
   - 9d (`f953a28`): rerun/submit → `cli/`; 7 shims deleted; conftest `_SCRIPT_MODULES` map.
10. ✅ **DONE** — the 8 behavior fixes applied, one reviewed commit each (fix #5 was folded
    into phase 6). #1 `efd4200`, #2 `207ff99`, #3 `b5d2cf0`, #4 `b1ae430`, #6 `249c965`,
    #7 `f781ba7`, #8 `e787ae5`. #2's submit-path SUBMITTED/SUBMIT_FAILED lines are not
    offline-reachable — verified by inspection; the offline-reachable SKIPPED half is tested.

---

## The 8 known-issue decisions (ALL APPLIED — see phase 10 commit table)

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

## Status: refactor complete
All phases (1-10) done on `refactor-package`; suite 69 green. Next step is out of this doc's
scope: **merge `refactor-package` → `main`** (a normal merge/PR when ready) and delete the branch.

Decisions locked in (2026-07-16): templates are PACKAGE DATA under `src/xas_pipeline/data/`;
internal invocation is `python -m xas_pipeline...`; human commands are `xas-*` console_scripts;
`resources.project_root()` is a transitional repo-root anchor (valid only in the checkout/editable
layout). Post-merge follow-ups a future session could pick up: harden wheel packaging (verify
`src/xas_pipeline/data/**` ships in the wheel), and reconsider `project_root()` if the pipeline
ever runs from a bare install without a checkout.
