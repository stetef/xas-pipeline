"""Unit tests for filling an ORCA template: which atoms are frozen / in the QM region.

The selection rules are the physics-relevant part of ``render_orca_input``, so
each is pinned against the rule the pre-refactor code applied (git 4096d8c:
``extract_ca_atoms``, ``get_coord_true_nco_atoms``, ``get_xtb_constrained_atoms``,
``get_xtb_qm_indices``), written out here rather than derived from the code.

Behaviour change (deliberate, WIP 77e7365): a template placeholder the tags
cannot fill -- [CA_ATOM] with no matching atom, [INDICES] with no QM atom -- used
to be left in the input with a printed warning and exit 0, so ORCA died on a
literal ``[CA_ATOM]``. It is now a :class:`PrepError` and no input is written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xas_pipeline import modes, resources
from xas_pipeline.scheduler import get_scheduler
from xas_pipeline.stages import orca_prep
from xas_pipeline.stages.orca_prep import AtomTag, PrepError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CYS_XYZ = FIXTURES / "xyz_files" / "2j6a_ZN_homo_d2.60_cluster1.xyz"  # ATOM= tags, no COORD=
HIS_XYZ = FIXTURES / "xyz_his" / "4j1v_cluster2_Zn.xyz"  # ATOM= and COORD= tags
UNFILLED = re.compile(r"\[[A-Z][A-Z0-9_]*\]")

# A synthetic site covering every branch of the rules. The tags are what the
# carving step writes; nothing here is a geometry.
TAGS = [
    AtomTag(0, "ZN", None, None),
    AtomTag(1, "SG", None, "TRUE"),     # side-chain donor
    AtomTag(2, "CB", None, "FALSE"),
    AtomTag(3, "CA", None, "FALSE"),    # CA, not coordinating
    AtomTag(4, "O", None, "TRUE"),      # backbone carbonyl O coordinating
    AtomTag(5, "H", "N", "TRUE"),       # H on a backbone N, in the coordinating group
    AtomTag(6, "NE2", None, "TRUE"),    # side-chain donor
    AtomTag(7, "H", "CB", "FALSE"),
    AtomTag(8, "CA", None, "TRUE"),     # coordinating-group CA
    AtomTag(9, "OW", None, None),       # untagged COORD
]


def _template(mode: str) -> str:
    return (resources.template_root() / "orca-templates" / modes.get(mode).template).read_text()


def _render(mode: str, tags=TAGS) -> str:
    return orca_prep.render_orca_input(
        _template(mode), mode=modes.get(mode), tags=tags,
        tokens={"CHARGE": 0, "MULTIPLICITY": 1, "PDB_ID": "x", "ID_DIR": "/d", "MAXCORE": 1000,
                "ANFREQ": orca_prep.anfreq_directive(False), "GEOMETRY": "/d/x_clean.xyz"},
    )


def _frozen(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"^  \{C (\d+) C\}  # Freeze", text, re.MULTILINE)]


# ── Selection rules (old semantics written out) ──────────────────────────────


def test_ca_rule_freezes_every_ca_whatever_its_coord_tag():
    assert orca_prep.constrained_atoms(TAGS, "ca") == [3, 8]


def test_backbone_rule_freezes_coordinating_backbone_groups_only():
    """COORD=TRUE and (ATOM or BONDEDATOM in N/C/O/CA)."""
    assert orca_prep.constrained_atoms(TAGS, "backbone") == [4, 5, 8]


def test_xtb_rule_freezes_non_coordinating_atoms_and_coordinating_backbone():
    """COORD=FALSE, or COORD=TRUE on a backbone group; untagged atoms stay free."""
    assert orca_prep.constrained_atoms(TAGS, "xtb") == [2, 3, 4, 5, 7, 8]


def test_qm_region_rules():
    assert orca_prep.qm_region_ranges(TAGS, "all") == "{1} {4:6} {8}"
    assert orca_prep.qm_region_ranges(TAGS, "non-backbone") == "{1} {6}"


def test_tags_are_read_from_the_comments_sidecar(tmp_path):
    comments = tmp_path / "c.txt"
    comments.write_text(
        "id=x CHARGE=0 MULTIPLICITY=1\n"
        "Atom 0: # RES=ZN ATOM=ZN\n"
        "Atom 1: # RES=CYS ATOM=sg coord=true\n"
        "Atom 2: # RES=CYS ATOM=H BONDEDATOM=CA COORD=FALSE\n"
        "Atom 3: (no comment)\n",
        encoding="utf-8",
    )
    assert orca_prep.read_atom_tags(comments) == [
        AtomTag(0, "ZN", None, None),
        AtomTag(1, "SG", None, "TRUE"),
        AtomTag(2, "H", "CA", "FALSE"),
        AtomTag(3, None, None, None),
    ]


# ── Rendering ────────────────────────────────────────────────────────────────


def test_backbone_template_freezes_the_backbone_rule_atoms():
    """Positive backbone case. No fixture carries a COORD=TRUE backbone atom, so
    the tags are synthetic; the rendered block is what the old code wrote."""
    text = _render("backbone")
    assert _frozen(text) == [4, 5, 8]
    assert "  {C 4 C}  # Freeze all coordinates (X, Y, Z) of atom 4\n" in text
    assert not UNFILLED.search(text), UNFILLED.findall(text)


@pytest.mark.parametrize("mode", ["caopt-anfreq", "quick-ca-fixed"])
def test_ca_templates_freeze_the_ca_atoms(mode):
    text = _render(mode)
    assert _frozen(text) == [3, 8]
    assert not UNFILLED.search(text)


def test_xtb_constrained_fills_both_placeholders():
    text = _render("xtb-constrained")
    assert _frozen(text) == [2, 3, 4, 5, 7, 8]
    assert "%QMMM QMATOMS {1} {4:6} {8} END END" in text
    assert not UNFILLED.search(text)


def test_xtb_free_qm_region_excludes_backbone_groups():
    text = _render("xtb-free")
    assert "%QMMM QMATOMS {1} {6} END END" in text
    assert _frozen(text) == []
    assert not UNFILLED.search(text)


@pytest.mark.parametrize("mode", sorted(modes.ORCA_MODES))
def test_every_orca_mode_renders_with_no_placeholder_left(mode):
    assert not UNFILLED.search(_render(mode)), mode


@pytest.mark.parametrize(
    ("mode", "tags", "token"),
    [
        ("caopt-anfreq", [t for t in TAGS if t.atom != "CA"], "[CA_ATOM]"),
        ("quick-ca-fixed", [t for t in TAGS if t.atom != "CA"], "[CA_ATOM]"),
        ("backbone", [t for t in TAGS if t.index not in (4, 5, 8)], "[CA_ATOM]"),
        ("xtb-free", [t for t in TAGS if t.index not in (1, 6)], "[INDICES]"),
        ("xtb-constrained", [t for t in TAGS if t.coord != "TRUE"], "[INDICES]"),
    ],
)
def test_an_unfillable_placeholder_is_an_error(mode, tags, token):
    """Regression (Phase 1): used to be written verbatim with exit 0."""
    with pytest.raises(PrepError) as excinfo:
        _render(mode, tags)
    assert token in str(excinfo.value)
    assert mode in str(excinfo.value)


def test_modes_without_constraints_need_no_tags():
    for mode in ("quick", "hopt-anfreq", "carved-anfreq", "no-constraints", "hopt-spring", "carved-spring"):
        assert not UNFILLED.search(_render(mode, [])), mode


# ── prepare_stage: a failed render writes no input ──────────────────────────


@pytest.mark.parametrize(("xyz", "mode", "token"), [
    (CYS_XYZ, "xtb-free", "[INDICES]"),        # no COORD= tags at all
    (CYS_XYZ, "xtb-constrained", "[INDICES]"),
    (HIS_XYZ, "backbone", "[CA_ATOM]"),         # COORD=TRUE atoms are all side chain
])
def test_prepare_stage_raises_and_writes_no_input_or_job_script(tmp_path, xyz, mode, token):
    src = tmp_path / "in"
    src.mkdir()
    staged = src / xyz.name
    staged.write_bytes(xyz.read_bytes())
    staged.with_suffix(".pc").write_text("0\n", encoding="utf-8")
    out = tmp_path / "out"

    with pytest.raises(PrepError) as excinfo:
        orca_prep.prepare_stage(staged, out, mode, get_scheduler("slurm"))
    assert token in str(excinfo.value)
    assert xyz.name in str(excinfo.value)

    written = {p.name for p in out.rglob("*") if p.is_file()}
    assert not [n for n in written if n.endswith(".in")], written
    assert not [n for n in written if n.endswith(".script")], written


def test_prepare_stage_with_ca_atoms_writes_the_input(tmp_path):
    """The counterpart: the Cys fixture has ATOM=CA tags, so caopt-anfreq fills."""
    stage = orca_prep.prepare_stage(CYS_XYZ, tmp_path, "caopt-anfreq", get_scheduler("slurm"))
    text = (stage.run_dir / f"{stage.run_id}.in").read_text()
    assert len(_frozen(text)) == 4
    assert not UNFILLED.search(text)
    assert stage.job_script is not None and stage.job_script.is_file()
