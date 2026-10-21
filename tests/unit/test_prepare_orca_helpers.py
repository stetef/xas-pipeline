"""Unit tests for the xTB QMATOMS index-range formatter.

The old private ``_format_index_ranges(indices)`` is now folded into
:func:`orca_prep.qm_region_ranges`, which selects the COORD=TRUE atoms and
formats them in one go. The output format is the ORCA input contract, so the
same four cases are pinned through the public function (rule ``"all"`` selects
every COORD=TRUE atom, so the indices in == the indices formatted).
"""

from xas_pipeline.stages.orca_prep import AtomTag, qm_region_ranges


def _ranges(indices):
    tags = [AtomTag(i, "X", None, "TRUE") for i in indices]
    return qm_region_ranges(tags, "all")


def test_format_index_ranges_empty():
    assert _ranges([]) == ""


def test_format_index_ranges_single():
    assert _ranges([5]) == "{5}"


def test_format_index_ranges_contiguous():
    assert _ranges([1, 2, 3]) == "{1:3}"


def test_format_index_ranges_mixed_runs():
    assert _ranges([1, 2, 3, 7, 9, 10, 11]) == "{1:3} {7} {9:11}"


def test_format_index_ranges_sorts_and_deduplicates():
    """As before: the old caller passed sorted(set(...)) to the formatter."""
    assert _ranges([11, 9, 10, 9, 1]) == "{1} {9:11}"
