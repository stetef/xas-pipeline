"""Unit tests for the per-structure grouped run-dir layout.

Run dirs are ``<batch>/<id>/<id>-<mode>/`` so several ORCA modes can be run from
one starting structure without overwriting each other. These tests pin the two
things the rest of the pipeline relies on: the naming round-trip
(mode -> run id -> mode) and the scan (:func:`layout.iter_id_dirs` descending
into group dirs while still handling pre-grouping flat batches).
"""

from __future__ import annotations

import pytest

from xas_pipeline import layout
from xas_pipeline.stages.orca_prep import INTERP_HESSIAN_MODES, TEMPLATE_FILE_BY_MODE


def _make_run_dir(parent, name, *, files=("marker.in",)):
    run_dir = parent / name
    run_dir.mkdir(parents=True)
    for filename in files:
        (run_dir / filename).write_text("", encoding="utf-8")
    return run_dir


def test_known_modes_match_the_template_registry():
    """layout owns the naming vocabulary; orca_prep owns mode -> template."""
    assert set(layout.KNOWN_MODES) == set(TEMPLATE_FILE_BY_MODE)


@pytest.mark.parametrize("mode", layout.KNOWN_MODES)
def test_run_id_round_trips_through_mode_from_run_id(mode):
    run_id = layout.run_id_for("2j6a_ZN_cluster1", mode)
    assert layout.mode_from_run_id(run_id) == mode


def test_longest_mode_suffix_wins():
    """'<id>-quick-ca-fixed' must not be read as the 'ca-fixed' it also ends with."""
    run_id = layout.run_id_for("cluster", "quick-ca-fixed")
    assert run_id == "cluster-quick-ca-fixed"
    assert layout.mode_from_run_id(run_id) == "quick-ca-fixed"


def test_h_only_keeps_its_historical_casing():
    # Batches on disk predate the systematic suffix and use "-H-only".
    assert layout.run_id_for("cluster", "h-only") == "cluster-H-only"
    assert layout.mode_from_run_id("cluster-H-only") == "h-only"


def test_mode_from_run_id_is_none_for_a_pre_grouping_run_dir():
    assert layout.mode_from_run_id("2j6a_ZN_cluster1") is None


def test_opt_interp_is_not_read_as_plain_interp():
    """Regression: '-opt-interp' ends with '-interp' and used to resolve to it.

    That silently relabelled every opt-interp run as interp -- 302 run dirs in the
    clustering-validation tree -- so per-mode tallies and any mode-keyed dispatch
    saw the wrong mode with no error anywhere.
    """
    assert layout.mode_from_run_id("2j6a_ZN_cluster1-opt-interp") == "opt-interp"
    assert layout.mode_from_run_id("2j6a_ZN_cluster1-interp") == "interp"


@pytest.mark.parametrize("mode", ["interp-hopt", "interp-raw"])
def test_interp_variants_are_not_read_as_plain_interp(mode):
    """The interp family shares a stem, and only one of them is `interp`.

    Deliberately named ``interp-<variant>`` rather than ``<variant>-interp``: the
    latter would also end with ``-interp``, so a suffix left out of RUN_DIR_MODES
    would quietly resolve to ``interp`` -- the opt-interp bug again. With the stem
    first, an unregistered suffix returns None and surfaces as a loud error
    instead of a wrong label.
    """
    run_id = layout.run_id_for("2j6a_ZN_cluster1", mode)
    assert run_id == f"2j6a_ZN_cluster1-{mode}"
    assert layout.mode_from_run_id(run_id) == mode


@pytest.mark.parametrize("mode", layout.SUFFIX_ONLY_MODES)
def test_suffix_only_modes_round_trip_too(mode):
    """They have no template, but their run dirs still have to parse back."""
    assert layout.mode_from_run_id(layout.run_id_for("cluster", mode)) == mode


def test_no_orca_modes_is_exactly_the_untemplated_set():
    """NO_ORCA_MODES is derived from SUFFIX_ONLY_MODES, not maintained beside it.

    A mode is suffix-only precisely because it has no ORCA template, and a mode
    with no template cannot run ORCA -- so the two sets coincide by construction.
    Keeping them as independent literals is what would let them drift.
    """
    assert layout.NO_ORCA_MODES == frozenset(layout.SUFFIX_ONLY_MODES)
    for mode in layout.NO_ORCA_MODES:
        assert mode not in TEMPLATE_FILE_BY_MODE


def test_suffix_only_modes_are_recognized_but_have_no_orca_template():
    """opt-interp skips ORCA entirely, so it is a suffix without a template.

    This is why KNOWN_MODES (modes with templates) and RUN_DIR_MODES (the suffix
    vocabulary) are separate; collapsing them breaks one of the two invariants.
    """
    for mode in layout.SUFFIX_ONLY_MODES:
        assert mode in layout.RUN_DIR_MODES
        assert mode not in layout.KNOWN_MODES
        assert mode not in TEMPLATE_FILE_BY_MODE


def test_run_dir_modes_covers_every_templated_mode():
    assert set(layout.KNOWN_MODES) <= set(layout.RUN_DIR_MODES)


@pytest.mark.parametrize("mode", ["interp", "interp-hopt", "interp-raw", "opt-interp"])
def test_interp_hessian_modes_covers_every_interpolated_route(mode):
    """None of these routes gets a Hessian from ORCA, so the wrapper must build one.

    opt-interp reached this set only via the mode_from_run_id bug; teaching the
    parser the real suffix without listing it here would have silently stopped the
    corvus wrapper interpolating the Hessian for every opt-interp run.
    """
    assert mode in INTERP_HESSIAN_MODES


def test_every_no_orca_mode_interpolates_its_hessian():
    """A mode with no ORCA stage has no other way to get one.

    Left off INTERP_HESSIAN_MODES, such a mode would reach prepare-corvus with no
    .hess and no step that could have written it -- so tie the two sets together
    rather than relying on both lists being edited at once.
    """
    assert layout.NO_ORCA_MODES <= INTERP_HESSIAN_MODES


def test_nested_mode_run_dirs_finds_a_suffix_only_run(tmp_path):
    """An opt-interp run dir must still be discovered as a run of its group."""
    group = tmp_path / "cluster1"
    _make_run_dir(group, "cluster1-opt-interp")
    found = [p.name for p in layout.nested_mode_run_dirs(group)]
    assert found == ["cluster1-opt-interp"]


def test_run_dir_for_nests_the_mode_under_the_structure(tmp_path):
    run_dir = layout.run_dir_for(tmp_path, "cluster1", "interp")
    assert run_dir == tmp_path / "cluster1" / "cluster1-interp"


def test_iter_id_dirs_descends_into_group_dirs(tmp_path):
    group = tmp_path / "cluster1"
    for mode in ("ca-fixed", "no-constraints", "interp"):
        _make_run_dir(group, f"cluster1-{mode}")

    found = [p.name for p in layout.iter_id_dirs(tmp_path)]
    assert found == ["cluster1-ca-fixed", "cluster1-interp", "cluster1-no-constraints"]


def test_iter_id_dirs_still_yields_flat_pre_grouping_run_dirs(tmp_path):
    _make_run_dir(tmp_path, "old_cluster")
    group = tmp_path / "cluster1"
    _make_run_dir(group, "cluster1-interp")

    found = [p.name for p in layout.iter_id_dirs(tmp_path)]
    assert found == ["cluster1-interp", "old_cluster"]


def test_iter_id_dirs_skips_helper_dirs(tmp_path):
    _make_run_dir(tmp_path, "cluster1")
    for skipped in layout.SKIP_DIR_NAMES:
        _make_run_dir(tmp_path, skipped)

    assert [p.name for p in layout.iter_id_dirs(tmp_path)] == ["cluster1"]


def test_only_ids_accepts_either_a_group_name_or_a_run_id(tmp_path):
    group = tmp_path / "cluster1"
    _make_run_dir(group, "cluster1-ca-fixed")
    _make_run_dir(group, "cluster1-interp")
    other = tmp_path / "cluster2"
    _make_run_dir(other, "cluster2-interp")

    by_group = [p.name for p in layout.iter_id_dirs(tmp_path, only_ids={"cluster1"})]
    assert by_group == ["cluster1-ca-fixed", "cluster1-interp"]

    by_run = [p.name for p in layout.iter_id_dirs(tmp_path, only_ids={"cluster1-interp"})]
    assert by_run == ["cluster1-interp"]


def test_split_run_dir_is_not_mistaken_for_a_group_dir(tmp_path):
    """A cleaned split run dir holds only working-/output- subdirs and no files."""
    run_dir = tmp_path / "cluster1"
    (run_dir / "working-cluster1").mkdir(parents=True)
    (run_dir / "output-cluster1").mkdir(parents=True)

    assert not layout.is_group_dir(run_dir)
    assert [p.name for p in layout.iter_id_dirs(tmp_path)] == ["cluster1"]


def test_run_dir_with_files_is_not_a_group_dir(tmp_path):
    run_dir = _make_run_dir(tmp_path, "cluster1")
    (run_dir / "cluster1-sub").mkdir()  # e.g. a Corvus working subdir
    assert not layout.is_group_dir(run_dir)


def test_group_dir_requires_a_prefixed_child(tmp_path):
    """Unrelated nesting is left alone rather than being scanned one level down."""
    parent = tmp_path / "cluster1"
    (parent / "something-else").mkdir(parents=True)
    assert not layout.is_group_dir(parent)
