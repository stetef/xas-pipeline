"""Template rendering: placeholders must all be filled; PBS and Slurm job bodies agree."""

from __future__ import annotations

import pytest

from xas_pipeline import resources, templates


def test_render_refuses_an_unfilled_placeholder(tmp_path):
    src = tmp_path / "t.script"
    src.write_text("run [A] then [B_2]; bash [[ -x x ]] stays\n", encoding="utf-8")
    dest = tmp_path / "out.script"
    with pytest.raises(ValueError, match=r"\[B_2\]"):
        templates.render(src, dest, {"A": "a"})
    assert not dest.exists()
    templates.render(src, dest, {"A": "a", "B_2": "b"})
    assert dest.read_text(encoding="utf-8") == "run a then b; bash [[ -x x ]] stays\n"


def _body(scheduler: str, name: str, start: str) -> str:
    text = (resources.template_root() / f"{scheduler}-scripts" / name).read_text(encoding="utf-8")
    return text[text.index(start):]


def test_corvus_job_bodies_match_across_schedulers():
    """PBS used to pre-create Corvus3_cfavg_<mode>/Corvus1Zn_FEFF, a name CORVUS does
    not use (the real one carries the absorber index), and had no runner search."""
    assert _body("pbs", "corvus-job.script", "input_file=") == _body("slurm", "corvus-job.script", "input_file=")
    assert "mkdir" not in _body("pbs", "corvus-job.script", "input_file=")


@pytest.mark.parametrize("name", ["postprocess-job.script", "watchdog-job.script"])
def test_batch_job_bodies_match_across_schedulers(name):
    assert _body("pbs", name, "set -euo pipefail") == _body("slurm", name, "set -euo pipefail")
