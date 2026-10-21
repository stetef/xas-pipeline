"""The generated postprocess script must not depend on the submitter's PATH.

Regression test for a real failure: a slurm batch submitted by invoking
``.venv/bin/xas-rerun-corvus`` at its absolute path (rather than from an
activated venv) produced a postprocess job that died instantly with
``python: command not found`` (exit 127). Slurm exports the submit environment,
so the template's bare ``python`` only resolved when the submitting shell
happened to have the venv on PATH. Both corvus jobs had already succeeded, so
the spectra were computed and then not processed.

The slurm template now resolves ``PYTHON_BIN`` itself, the way
``corvus-wrapper.script`` always has. PBS keeps inheriting the environment via
``-V``, which is its documented mechanism.
"""

from __future__ import annotations

import re
import sys

import pytest

from xas_pipeline import jobscripts
from xas_pipeline.scheduler import get_scheduler


def _render(tmp_path, scheduler: str, **kwargs) -> str:
    batch = tmp_path / "batch-out"
    batch.mkdir(exist_ok=True)
    script = jobscripts.postprocess(
        get_scheduler(scheduler),
        batch,
        batch / "downloading-station",
        dest=tmp_path / f"postprocess-{scheduler}.script",
        **kwargs,
    )
    return script.read_text(encoding="utf-8")


# Every line that actually invokes an interpreter: the five stage commands
# (orca-check, process-feff, cleanup, auto-rerun-corvus, download) plus the
# inline timing-summary heredoc.
_INVOCATION = re.compile(r"^\s*(\S+)\s+(?:-m xas_pipeline|- <<'PY')", re.MULTILINE)


def test_slurm_script_never_calls_a_bare_python(tmp_path):
    text = _render(tmp_path, "slurm")

    invocations = _INVOCATION.findall(text)
    assert invocations, "no interpreter invocations found -- test is not looking at the right lines"
    assert all(
        call == '"$PYTHON_BIN"' for call in invocations
    ), f"bare interpreter survives in the slurm postprocess script: {invocations}"


@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
def test_script_runs_the_interpreter_that_rendered_it(tmp_path, scheduler):
    """Neither a bare `python` nor the submitting shell's PATH: the rendering
    interpreter is baked in (overridable via $PIPELINE_PYTHON), on both schedulers."""
    text = _render(tmp_path, scheduler)

    assert f'PYTHON_BIN="${{PIPELINE_PYTHON:-{sys.executable}}}"' in text
    # Set before the first use, or $PYTHON_BIN expands to nothing under `set -u`.
    assert text.index("PYTHON_BIN=") < text.index('"$PYTHON_BIN" -m xas_pipeline')
    # exit 127 (the shell's "command not found") when the interpreter is gone.
    assert "exit 127" in text
    assert set(_INVOCATION.findall(text)) == {'"$PYTHON_BIN"'}


def test_stage_times_runs_last(tmp_path):
    text = _render(tmp_path, "slurm")
    assert text.index("xas_pipeline.stages.download") < text.index("xas_pipeline.stages.stage_times")


@pytest.mark.parametrize("skip", [True, False])
def test_cleanup_command_follows_the_same_interpreter_rule(tmp_path, skip):
    """--skip-cleanup renders `true`; otherwise the resolved interpreter, not `python`."""
    text = _render(tmp_path, "slurm", cleanup=not skip)

    if skip:
        assert "xas_pipeline.stages.cleanup" not in text
    else:
        assert '"$PYTHON_BIN" -m xas_pipeline.stages.cleanup' in text


def test_auto_rerun_triage_runs_between_cleanup_and_download(tmp_path):
    """Order is load-bearing: after cleanup, before the quarantine pass."""
    text = _render(tmp_path, "slurm")

    assert '"$PYTHON_BIN" -m xas_pipeline.cli.auto_rerun_corvus' in text
    assert (
        text.index("xas_pipeline.stages.cleanup")
        < text.index("xas_pipeline.cli.auto_rerun_corvus")
        < text.index("xas_pipeline.stages.download")
    )
    # Triage must never take the postprocess job down with it (`set -e`): a
    # failed triage just leaves the failed ids quarantined as before.
    triage_line = next(
        line for line in text.splitlines() if "auto_rerun_corvus" in line and "-m " in line
    )
    assert "||" in triage_line


def test_auto_rerun_triage_is_skipped_when_no_spectra_are_processed(tmp_path):
    """Nothing writes corvus-failed-ids.txt without process-feff, so triage is moot."""
    text = _render(tmp_path, "slurm", process_feff=False)
    assert "auto_rerun_corvus" not in text
