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

import pytest

from xas_pipeline import orchestrate


def _render(tmp_path, scheduler: str, **kwargs) -> str:
    script = tmp_path / f"postprocess-{scheduler}.script"
    orchestrate._write_postprocess_script(
        script,
        scheduler,
        tmp_path / "batch-out",
        tmp_path / "batch-out" / "downloading-station",
        skip_extract=False,
        skip_process_feff=False,
        skip_prepare_download=False,
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


def test_slurm_script_resolves_python_before_using_it(tmp_path):
    text = _render(tmp_path, "slurm")

    assert "PIPELINE_ROOT=" in text
    assert 'PYTHON_BIN="$PIPELINE_ROOT/.venv/bin/python"' in text
    # Resolution has to precede the first use, or $PYTHON_BIN expands to nothing
    # under `set -u` and the job dies just as it did before.
    assert text.index("PYTHON_BIN=") < text.index('"$PYTHON_BIN" -m xas_pipeline')
    # exit 127 is the shell's own "command not found" code; keep it as the signal
    # when no interpreter can be found at all.
    assert "exit 127" in text


def test_pbs_script_relies_on_the_inherited_environment(tmp_path):
    """PBS `-V` is the documented mechanism there; do not silently diverge."""
    text = _render(tmp_path, "pbs")

    assert "#PBS -V" in text
    assert _INVOCATION.findall(text) == ["python"] * 6


@pytest.mark.parametrize("skip", [True, False])
def test_cleanup_command_follows_the_same_interpreter_rule(tmp_path, skip):
    """--skip-cleanup renders `true`; otherwise the resolved interpreter, not `python`."""
    text = _render(tmp_path, "slurm", skip_cleanup=skip)

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
    script = tmp_path / "postprocess-noprocess.script"
    orchestrate._write_postprocess_script(
        script,
        "slurm",
        tmp_path / "batch-out",
        tmp_path / "batch-out" / "downloading-station",
        skip_extract=False,
        skip_process_feff=True,
        skip_prepare_download=False,
    )
    assert "auto_rerun_corvus" not in script.read_text(encoding="utf-8")
