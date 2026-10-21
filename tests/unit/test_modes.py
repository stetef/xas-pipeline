"""The mode registry (:mod:`xas_pipeline.modes`) and every table derived from it.

Before the registry each mode was listed in half a dozen places (templates,
AnFreq defaults, run-dir suffixes, no-ORCA set, spring-Hessian set, two argparse
front-ends) and they drifted -- e.g. a no-ORCA mode missing from the spring set
silently got no Hessian. These tests check that every mode has what every
consumer needs, and that the consumers really do read the registry.
"""

from __future__ import annotations

import argparse
import re

import pytest

from xas_pipeline import jobscripts, layout, modes, orchestrate, resources
from xas_pipeline.scheduler import SCHEDULERS
from xas_pipeline.stages import orca_prep

UNFILLED = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
ALL_MODES = sorted(modes.MODES)


def test_registry_keys_are_the_mode_names():
    assert all(name == m.name for name, m in modes.MODES.items())
    assert modes.DEFAULT_MODE in modes.MODES


def test_get_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode 'nope'"):
        modes.get("nope")


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_is_a_run_dir_suffix_that_round_trips(mode):
    assert mode in layout.RUN_DIR_MODES
    assert layout.mode_from_run_id(layout.run_id_for("cluster1", mode)) == mode


def test_the_derived_sets_partition_the_registry():
    assert set(modes.ORCA_MODES) | modes.NO_ORCA_MODES == set(modes.MODES)
    assert not set(modes.ORCA_MODES) & modes.NO_ORCA_MODES
    assert modes.NO_ORCA_MODES <= modes.SPRING_HESSIAN_MODES
    assert {"carved-spring", "hopt-spring", "asis-spring", "caopt-spring"} == modes.SPRING_HESSIAN_MODES


@pytest.mark.parametrize("mode", sorted(modes.ORCA_MODES))
def test_every_orca_mode_has_a_template_whose_placeholders_match_the_mode(mode):
    """[CA_ATOM] iff the mode freezes atoms, [INDICES] iff it has a QM region.

    A placeholder the mode does not know how to fill would be written verbatim
    into the ORCA input; a rule with no placeholder would be silently ignored.
    """
    spec = modes.get(mode)
    path = resources.template_root() / "orca-templates" / spec.template
    text = path.read_text()
    assert ("[CA_ATOM]" in text) == (spec.constraints is not None), mode
    assert ("[INDICES]" in text) == (spec.qm_region is not None), mode
    # Everything else is a simple token prepare_stage always fills.
    simple = set(UNFILLED.findall(text)) - {"[CA_ATOM]", "[INDICES]"}
    assert simple <= {"[CHARGE]", "[MULTIPLICITY]", "[PDB_ID]", "[ID_DIR]", "[MAXCORE]", "[ANFREQ]", "[GEOMETRY]"}


def test_only_point_charge_modes_ask_for_a_pc_file():
    assert {n for n, m in modes.MODES.items() if m.needs_point_charges} == {"backbone"}


@pytest.mark.parametrize("sched", sorted(SCHEDULERS))
@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_has_its_scheduler_scripts(tmp_path, mode, sched):
    """Each mode renders a CORVUS wrapper (and, with ORCA, a job script) on both schedulers."""
    scheduler = SCHEDULERS[sched]
    run_id = layout.run_id_for("cluster1", mode)
    scripts = [jobscripts.corvus_wrapper(scheduler, tmp_path, run_id, mode=mode)]
    if mode in modes.ORCA_MODES:
        scripts.append(jobscripts.orca_job(scheduler, tmp_path, run_id, nprocs=4, mem_gb=16))
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert not UNFILLED.search(text), (script.name, UNFILLED.findall(text))
    # Executable lines only: the wrapper's failure hint echoes the interp command.
    executable = [
        line.strip() for line in scripts[0].read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "echo "))
    ]
    assert f"OPTIMIZATION_MODE={mode}" in executable
    builds_hessian = any("stages.interp_hessian" in line for line in executable)
    assert builds_hessian == (mode in modes.SPRING_HESSIAN_MODES), mode


def test_flags_are_unique():
    flags = [m.flag for m in modes.MODES.values() if m.flag]
    assert len(flags) == len(set(flags))


@pytest.mark.parametrize(
    "build_parser", [orchestrate.build_parser, orca_prep.build_parser], ids=["run-batch", "prepare-orca"]
)
def test_both_front_ends_accept_every_mode_by_flag_and_by_name(build_parser):
    parser = build_parser()
    assert parser.parse_args(["x.xyz"]).mode == modes.DEFAULT_MODE
    for name, spec in modes.MODES.items():
        assert parser.parse_args(["x.xyz", "--mode", name]).mode == name
        if spec.flag:
            assert parser.parse_args(["x.xyz", spec.flag]).mode == name


def test_mode_flags_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    modes.add_mode_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--quick", "--interp"])


def test_historical_flags_keep_their_meaning():
    """Operators' muscle memory and scripts: the flag -> mode map must not move."""
    parser = argparse.ArgumentParser()
    modes.add_mode_arguments(parser)
    expected = {
        "--quick": "quick", "--quick-ca-fixed": "quick-ca-fixed", "--H": "hopt-anfreq",
        "--single": "carved-anfreq", "--free": "no-constraints", "--backbone": "backbone",
        "--xtb-free": "xtb-free", "--xtb-constrained": "xtb-constrained", "--interp": "hopt-spring",
        "--interp-raw": "asis-spring", "--caopt-spring": "caopt-spring",
    }
    assert {flag: parser.parse_args([flag]).mode for flag in expected} == expected
    assert {m.flag for m in modes.MODES.values() if m.flag} == set(expected)


def test_pre_accepts_only_modes_with_an_orca_stage():
    parser = orca_prep.build_parser()
    assert parser.parse_args(["x.xyz", "--pre", "quick-ca-fixed"]).pre == ["quick-ca-fixed"]
    for name in modes.NO_ORCA_MODES:
        with pytest.raises(SystemExit):
            parser.parse_args(["x.xyz", "--pre", name])
