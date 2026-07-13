"""Seed unit test proving the importlib loader works on a HYPHENATED script.

prepare-orca.py cannot be `import`ed normally; load_script() bridges that. If
this passes, every pure helper in every hyphen-named script is testable today,
before any rename. _format_index_ranges is pure (no I/O), so its output is a
stable behavioral contract to lock down.
"""

from conftest import load_script

prepare_orca = load_script("prepare-orca.py")


def test_format_index_ranges_empty():
    assert prepare_orca._format_index_ranges([]) == ""


def test_format_index_ranges_single():
    assert prepare_orca._format_index_ranges([5]) == "{5}"


def test_format_index_ranges_contiguous():
    assert prepare_orca._format_index_ranges([1, 2, 3]) == "{1:3}"


def test_format_index_ranges_mixed_runs():
    assert prepare_orca._format_index_ranges([1, 2, 3, 7, 9, 10, 11]) == "{1:3} {7} {9:11}"
