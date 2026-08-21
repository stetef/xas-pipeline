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
from xas_pipeline.stages.orca_prep import SPRING_HESSIAN_MODES, TEMPLATE_FILE_BY_MODE


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


def test_legacy_suffixes_resolve_to_the_mode_they_meant():
    """Run dirs written before the geometry x Hessian rename must still parse.

    They resolve to the *canonical* mode, so an un-migrated dir is reported as
    what it is rather than as unknown. Deliberately one-way: run_id_for emits only
    canonical names, so the round-trip does not close, and that asymmetry is the
    signal a dir predates the rename.
    """
    for legacy, canonical in layout.LEGACY_SUFFIX_TO_MODE.items():
        assert layout.mode_from_run_id(f"cluster-{legacy}") == canonical
        assert canonical in layout.RUN_DIR_MODES, canonical

    # "-H-only" is the pre-suffix casing; batches on disk carry it.
    assert layout.mode_from_run_id("cluster-H-only") == "hopt-anfreq"
    # ...and we now emit the canonical spelling instead.
    assert layout.run_id_for("cluster", "hopt-anfreq") == "cluster-hopt-anfreq"


def test_no_canonical_suffix_is_a_suffix_of_another():
    """The structural fix behind the 302-dir mislabel, now enforced.

    "-opt-interp" ended with "-interp", so a missing registration resolved to the
    wrong mode silently. Under the two-axis names that collision cannot recur --
    assert it, so a future mode named e.g. "fast-hopt-spring" fails here rather
    than quietly shadowing hopt-spring.
    """
    for mode in layout.RUN_DIR_MODES:
        others = [m for m in layout.RUN_DIR_MODES if m != mode]
        clashes = [m for m in others if mode.endswith(f"-{m}")]
        assert not clashes, f"{mode} ends with another mode: {clashes}"


def test_mode_from_run_id_is_none_for_a_pre_grouping_run_dir():
    assert layout.mode_from_run_id("2j6a_ZN_cluster1") is None


def test_legacy_opt_interp_is_not_read_as_legacy_interp():
    """Regression: '-opt-interp' ends with '-interp' and used to resolve to it.

    That silently relabelled every opt-interp run as interp -- 302 run dirs in the
    clustering-validation tree -- so per-mode tallies and any mode-keyed dispatch
    saw the wrong mode with no error anywhere. Both spellings are legacy now, and
    longest-suffix-wins still has to separate them.
    """
    assert layout.mode_from_run_id("2j6a_ZN_cluster1-opt-interp") == "caopt-spring"
    assert layout.mode_from_run_id("2j6a_ZN_cluster1-interp") == "carved-spring"
    assert layout.mode_from_run_id("2j6a_ZN_cluster1-interp-hopt") == "hopt-spring"


SPRING_FAMILY = ["caopt-anfreq", "caopt-spring", "hopt-spring", "carved-spring", "asis-spring"]


@pytest.mark.parametrize("mode", SPRING_FAMILY)
def test_comparison_family_round_trips(mode):
    """The geometry x Hessian names, which the A/B/C comparison is keyed on."""
    run_id = layout.run_id_for("2j6a_ZN_cluster1", mode)
    assert run_id == f"2j6a_ZN_cluster1-{mode}"
    assert layout.mode_from_run_id(run_id) == mode


def test_comparison_family_names_both_axes():
    """Each name is <geometry>-<hessian>, so the axes are readable off the name."""
    geometries = {"caopt", "hopt", "carved", "asis"}
    hessians = {"anfreq", "spring"}
    for mode in SPRING_FAMILY:
        geom, _, hess = mode.rpartition("-")
        assert geom in geometries, mode
        assert hess in hessians, mode


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


@pytest.mark.parametrize("mode", ["carved-spring", "hopt-spring", "caopt-spring", "asis-spring"])
def test_spring_hessian_modes_covers_every_interpolated_route(mode):
    """None of these routes gets a Hessian from ORCA, so the wrapper must build one.

    caopt-spring (formerly opt-interp) reached this set only via the
    mode_from_run_id bug; teaching the parser the real suffix without listing it
    here would have silently stopped the corvus wrapper interpolating the Hessian
    for every one of those runs.
    """
    assert mode in SPRING_HESSIAN_MODES


def test_every_no_orca_mode_interpolates_its_hessian():
    """A mode with no ORCA stage has no other way to get one.

    Left off INTERP_HESSIAN_MODES, such a mode would reach prepare-corvus with no
    .hess and no step that could have written it -- so tie the two sets together
    rather than relying on both lists being edited at once.
    """
    assert layout.NO_ORCA_MODES <= SPRING_HESSIAN_MODES


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
