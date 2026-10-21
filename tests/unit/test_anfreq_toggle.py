"""Unit tests for the [ANFREQ] placeholder that replaced hard-coded "! AnFreq".

The refactor's whole claim is that it changed *nothing* by default: every mode
renders the same AnFreq state its template used to hard-code. These tests pin
that claim to the pre-refactor truth, recorded here as an explicit table, so a
future edit to a template or to a mode's ``anfreq`` default in
:mod:`xas_pipeline.modes` cannot quietly flip a mode's default on or off.
"""

from __future__ import annotations

import pytest

from xas_pipeline import modes, resources, templates
from xas_pipeline.stages import orca_prep

# What each template hard-coded before [ANFREQ] existed (git 6b970e2). Written out
# rather than read from the registry: a test that compares the table to itself
# would pass no matter what the table said.
ANFREQ_BEFORE_REFACTOR = {
    "caopt-anfreq": True,
    "quick": False,
    "quick-ca-fixed": False,
    "hopt-anfreq": True,
    "carved-anfreq": True,
    "no-constraints": True,
    "backbone": True,
    "xtb-free": True,
    "xtb-constrained": False,
    "carved-spring": False,
    "hopt-spring": False,
}


def _template_text(mode: str) -> str:
    return (resources.template_root() / "orca-templates" / modes.get(mode).template).read_text()


def test_every_orca_mode_has_a_recorded_default():
    """The table covers exactly the modes with an ORCA template."""
    assert set(ANFREQ_BEFORE_REFACTOR) == set(modes.ORCA_MODES)


def test_defaults_match_what_the_templates_used_to_hard_code():
    assert {name: modes.get(name).anfreq for name in modes.ORCA_MODES} == ANFREQ_BEFORE_REFACTOR


def test_no_orca_modes_do_not_claim_anfreq():
    """A mode with no ORCA stage has no AnFreq to run."""
    for name in modes.NO_ORCA_MODES:
        assert modes.get(name).anfreq is False, name


@pytest.mark.parametrize("mode", sorted(modes.ORCA_MODES))
def test_template_carries_both_placeholders(mode):
    """A template missing [ANFREQ] would silently ignore --anfreq/--no-anfreq, and
    one missing [GEOMETRY] would ignore a staged run's starting geometry."""
    text = _template_text(mode)
    assert "[ANFREQ]" in text, f"{mode}: template has no [ANFREQ] placeholder"
    assert "[GEOMETRY]" in text, f"{mode}: template has no [GEOMETRY] placeholder"
    # The keyword must not ALSO be hard-coded, or --no-anfreq could not remove it.
    assert "! AnFreq" not in text, f"{mode}: template still hard-codes ! AnFreq"


@pytest.mark.parametrize("mode", sorted(modes.ORCA_MODES))
def test_rendering_with_the_default_reproduces_the_old_keyword_state(mode):
    rendered = templates.fill(
        _template_text(mode), {"ANFREQ": orca_prep.anfreq_directive(modes.get(mode).anfreq)}
    )
    directives = [ln for ln in rendered.splitlines() if ln.strip() == "! AnFreq"]
    assert directives == (["! AnFreq"] if ANFREQ_BEFORE_REFACTOR[mode] else [])


@pytest.mark.parametrize("mode", sorted(modes.ORCA_MODES))
def test_the_toggle_can_move_every_mode_both_ways(mode):
    text = _template_text(mode)
    on = templates.fill(text, {"ANFREQ": orca_prep.anfreq_directive(True)})
    off = templates.fill(text, {"ANFREQ": orca_prep.anfreq_directive(False)})
    assert "\n! AnFreq\n" in on
    assert "! AnFreq" not in off


def test_plan_stages_single_stage_uses_the_mode_default():
    assert orca_prep.plan_stages("caopt-anfreq", [], None) == [("caopt-anfreq", True)]
    assert orca_prep.plan_stages("quick-ca-fixed", [], None) == [("quick-ca-fixed", False)]


def test_plan_stages_override_applies_to_the_final_stage_only():
    stages = orca_prep.plan_stages("caopt-anfreq", ["quick-ca-fixed"], False)
    assert stages == [("quick-ca-fixed", False), ("caopt-anfreq", False)]

    stages = orca_prep.plan_stages("quick-ca-fixed", ["quick"], True)
    assert stages == [("quick", False), ("quick-ca-fixed", True)]


def test_pre_stages_never_run_anfreq_even_when_their_mode_defaults_on():
    """An intermediate geometry is about to move again, so its Hessian is waste."""
    stages = orca_prep.plan_stages("caopt-anfreq", ["carved-anfreq", "hopt-anfreq"], None)
    assert [af for _, af in stages] == [False, False, True]
