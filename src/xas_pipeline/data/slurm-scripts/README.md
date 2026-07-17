# Scheduler job/wrapper templates

Bash templates for one scheduler backend. There is a parallel `pbs-scripts/`
directory with the same file set:

- `orca-job.script` — submits the ORCA opt+AnFreq job
- `corvus-job.script` — runs CORVUS/FEFF for one mode
- `corvus-wrapper.script` — preps CORVUS then runs `corvus-job.script` inline
- `postprocess-job.script` — orca-check → process-feff → download

These are **package data** (shipped inside `xas_pipeline`, resolved via
`resources.template_root()`), not repo-root files. `templates.fill`/`render`
substitutes `[UPPER_SNAKE]` placeholders (e.g. `[RUN_ID]`, `[PIPELINE_ROOT]`,
`[SCHEDULER]`, `[SCRATCH_EXCLUDE]`) at generation time. The generated scripts
invoke the pipeline as `python -m xas_pipeline.stages.<stage>` and source the
site `.env` for scheduler-specific paths.

Edit these files to change what the compute-node jobs do; edit `.env` (not these
templates) for site paths.
