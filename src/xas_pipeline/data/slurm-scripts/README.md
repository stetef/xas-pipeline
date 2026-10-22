# Scheduler job templates

Bash templates for one scheduler; `pbs-scripts/` has the same set, with the same
bodies under different job directives.

- `orca-job.script`: runs ORCA in node scratch; on failure, hands the run to
  `xas_pipeline.cli.rerun_orca`
- `corvus-wrapper.script`: spring Hessian (spring modes), prepare-corvus, then
  `corvus-job.script` inline
- `corvus-job.script`: runs CORVUS/FEFF
- `postprocess-job.script`: orca-check, process-feff, cleanup, CORVUS auto-rerun
  triage, download staging, stage times
- `watchdog-job.script`: recovers silently hung ORCA/CORVUS jobs

They ship as package data (`resources.template_root()`). `templates.render` fills
the `[UPPER_SNAKE]` placeholders when a job is generated and refuses to write a
script with one left over. Every script runs Python as `$PYTHON_BIN`: the
interpreter that generated it (`[PYTHON]`), unless `$PIPELINE_PYTHON` overrides.
Site paths belong in `.env`, which the scripts source, not in these files.
