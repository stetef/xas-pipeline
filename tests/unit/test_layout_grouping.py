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
from xas_pipeline.stages.orca_prep import TEMPLATE_FILE_BY_MODE


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
