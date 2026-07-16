"""Unit test for the pure ORCA index-range formatter.

_format_index_ranges is pure (no I/O), so its output is a stable behavioral
contract. Now imported directly from the package (phase 9); before the reorg it
was reached via conftest.load_script against the hyphenated prepare-orca.py.
"""

from xas_pipeline.stages import orca_prep as prepare_orca


def test_format_index_ranges_empty():
    assert prepare_orca._format_index_ranges([]) == ""


def test_format_index_ranges_single():
    assert prepare_orca._format_index_ranges([5]) == "{5}"


def test_format_index_ranges_contiguous():
    assert prepare_orca._format_index_ranges([1, 2, 3]) == "{1:3}"


def test_format_index_ranges_mixed_runs():
    assert prepare_orca._format_index_ranges([1, 2, 3, 7, 9, 10, 11]) == "{1:3} {7} {9:11}"
