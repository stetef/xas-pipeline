"""Characterization unit tests for the pure helpers now living in the package.

These pin the input->output behavior of logic that was previously
duplicated/embedded across the flat scripts. They touch no filesystem or
scheduler and run in milliseconds. Post-reorg (phase 9) they import from
``xas_pipeline.*`` directly rather than via conftest.load_script.
"""

from __future__ import annotations

import pytest

from xas_pipeline import orchestrate as rbp
from xas_pipeline.stages import orca_prep as orca
from xas_pipeline.chem import periodic


# --- scheduler: job-id parsing (-> scheduler.Scheduler.parse_job_id) ----------

class TestParseSubmittedJobId:
    def test_slurm_line(self):
        assert rbp._parse_submitted_job_id("Submitted batch job 123456") == "123456"

    def test_pbs_line_with_server_suffix(self):
        assert rbp._parse_submitted_job_id("78910.master-node") == "78910"

    def test_skips_leading_nonnumeric_lines(self):
        out = "Warning: something\nSubmitted batch job 42\n"
        assert rbp._parse_submitted_job_id(out) == "42"

    def test_empty_output_raises(self):
        with pytest.raises(ValueError):
            rbp._parse_submitted_job_id("")


# --- prepare-orca: resource sizing (-> chem/orca sizing helpers) --------------

class TestOrcaMaxcore:
    @pytest.mark.parametrize(
        "natoms, expected",
        [(10, 1000), (50, 1000), (51, 1800), (90, 1800), (140, 2800), (200, 4200), (201, 5600)],
    )
    def test_tiers(self, natoms, expected):
        assert orca.orca_maxcore_mb(natoms) == expected


class TestOrcaMemGb:
    def test_small_hits_floor(self):
        assert orca.orca_mem_gb(1, 1000) == 16

    def test_scales_with_nprocs_and_maxcore(self):
        # 16 * 4200 / 0.70 = 96000 MB -> ceil(93.75) = 94 GB
        assert orca.orca_mem_gb(16, 4200) == 94

    def test_zero_nprocs_coerced_to_one(self):
        assert orca.orca_mem_gb(0, 1000) == 16


class TestExtractNprocs:
    def test_bang_line_pal(self):
        assert orca.extract_nprocs_from_text("! B3LYP def2-SVP PAL8") == 8

    def test_pal_block_inline(self):
        assert orca.extract_nprocs_from_text("%pal nprocs 16 end") == 16

    def test_pal_block_multiline(self):
        assert orca.extract_nprocs_from_text("%pal\n  nprocs 12\nend") == 12

    def test_default_when_absent(self):
        assert orca.extract_nprocs_from_text("! B3LYP def2-SVP\n* xyz 0 1\n") == 1


# --- prepare-corvus: periodic table (-> chem/periodic) ------------------------

class TestAtomicNumber:
    def test_digit_passthrough(self):
        assert periodic.atomic_number_from_token("6") == 6

    def test_symbol_case_insensitive(self):
        assert periodic.atomic_number_from_token("c") == 6
        assert periodic.atomic_number_from_token("Zn") == 30

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            periodic.atomic_number_from_token("")

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError):
            periodic.atomic_number_from_token("Xx")


class TestCanonicalSymbol:
    def test_round_trip_through_number(self):
        # Robust to the table's casing choice: symbol -> Z must be stable.
        sym = periodic.canonical_symbol_from_token("zn")
        assert periodic.atomic_number_from_token(sym) == 30


class TestAtomicMass:
    def test_hydrogen_and_zinc_are_physical(self):
        assert 1.0 < periodic.atomic_mass_amu(1) < 1.1
        assert 60.0 < periodic.atomic_mass_amu(30) < 70.0

    def test_unknown_z_raises(self):
        with pytest.raises(ValueError):
            periodic.atomic_mass_amu(999)
