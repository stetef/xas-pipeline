#!/usr/bin/env python3
"""
Process XYZ files for ORCA calculations with specified ligand composition.
"""

import argparse
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from xas_pipeline import scheduler as _sched
from xas_pipeline import layout, templates, resources, config
from xas_pipeline.chem import xyz as _chem_xyz


# ---------------------------------------------------------------------------
# Size-adaptive memory (%MaxCore + scheduler --mem)
# ---------------------------------------------------------------------------
# The generated inputs previously set no %MaxCore, so ORCA used a small default
# and the RIJCOSX exchange build in the analytic-frequency (AnFreq) step ran out
# of per-process memory on the larger clusters ("No memory left for COSX RHS").
# Rather than over-request a flat large allocation for every job (slow to
# schedule), we size %MaxCore by atom count and request only as much scheduler
# memory as that implies.
#
# %MaxCore is PER PROCESS (MB); total ORCA memory ~= nprocs * MaxCore. These
# tiers keep small systems modest (faster in the queue) and give big systems
# enough headroom. Tune here in one place if jobs still OOM or over-request.
# (natoms_upper_inclusive, maxcore_mb_per_process)
ORCA_MAXCORE_TIERS = [
    (50, 1000),
    (90, 1800),
    (140, 2800),
    (200, 4200),
]
ORCA_MAXCORE_ABOVE = 5600  # for natoms greater than the largest tier bound

# ORCA can transiently exceed %MaxCore and also needs memory outside the
# per-core pool (program image, integral/disk buffers). Request scheduler memory
# so that nprocs*MaxCore is ~70% of the allocation, leaving ~30% headroom, and
# never below a sane floor.
ORCA_MEM_MAXCORE_FRACTION = 0.70
ORCA_MEM_FLOOR_GB = 16


# XYZ parsers moved to xas_pipeline.chem.xyz; aliased here for internal callers
# (and the importlib-based characterization tests). Retired in phase 9.
count_atoms_xyz = _chem_xyz.count_atoms_xyz


def orca_maxcore_mb(natoms):
    """Return the per-process %MaxCore (MB) for a given atom count."""
    for upper, maxcore in ORCA_MAXCORE_TIERS:
        if natoms <= upper:
            return maxcore
    return ORCA_MAXCORE_ABOVE


def orca_mem_gb(nprocs, maxcore_mb):
    """Return the scheduler --mem (whole GB) that backs nprocs*MaxCore + headroom."""
    nprocs = max(1, int(nprocs or 1))
    total_mb = nprocs * maxcore_mb / ORCA_MEM_MAXCORE_FRACTION
    return max(ORCA_MEM_FLOOR_GB, math.ceil(total_mb / 1024))


TEMPLATE_FILE_BY_MODE = {
    "caopt-anfreq": "orca-templates/orca-template-caopt-anfreq.in",
    "quick": "orca-templates/orca-template-quick.in",
    "quick-ca-fixed": "orca-templates/orca-template-quick-ca-fixed.in",
    "hopt-anfreq": "orca-templates/orca-template-hopt-anfreq.in",
    "carved-anfreq": "orca-templates/orca-template-carved-anfreq.in",
    "no-constraints": "orca-templates/orca-template-no-constraints.in",
    "backbone": "orca-templates/orca-template-backbone-charges.in",
    "xtb-free": "orca-templates/orca-template-xtb-free.in",
    "xtb-constrained": "orca-templates/orca-template-xtb-constrained.in",
    "carved-spring": "orca-templates/orca-template-carved-spring.in",
    "hopt-spring": "orca-templates/orca-template-hopt-spring.in",
}


# ---------------------------------------------------------------------------
# AnFreq toggle
# ---------------------------------------------------------------------------
# "! AnFreq" used to be hard-coded in each template, which made "the same
# optimization, minus the analytic frequencies" reachable only by adding a
# near-duplicate template. Every registered template now carries an [ANFREQ]
# placeholder instead, and the keyword is filled in here.
#
# MODE_ANFREQ_DEFAULT reproduces, per mode, exactly what its template said before
# the placeholder was introduced -- so the default behaviour of every mode is
# unchanged and a test pins that. --anfreq / --no-anfreq override it.
#
# It is deliberately an explicit table and not derived from the mode name: the
# "-anfreq"/"-spring" suffixes only cover the geometry x Hessian family, and the
# other six modes disagree among themselves (no-constraints/backbone/xtb-free run
# AnFreq; quick/quick-ca-fixed/xtb-constrained do not).
MODE_ANFREQ_DEFAULT = {
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

# What [ANFREQ] becomes. The "on" text is byte-for-byte the block the templates
# carried before, so a generated input for an AnFreq mode is unchanged.
ANFREQ_ON_TEXT = "# Do frequency calculation (produce .hess file)\n! AnFreq"
ANFREQ_OFF_TEXT = "# No frequency calculation: this run produces no .hess file"


def anfreq_default_for(mode):
    """Whether *mode* runs analytic frequencies unless told otherwise."""
    return MODE_ANFREQ_DEFAULT.get(mode, False)


def anfreq_directive(enabled):
    """The text filled into a template's [ANFREQ] placeholder."""
    return ANFREQ_ON_TEXT if enabled else ANFREQ_OFF_TEXT


# Modes whose Hessian comes from xas_pipeline.stages.interp_hessian (interpolated
# from ligand spring models) rather than from ORCA, which the corvus wrapper runs
# before prepare-corvus. Consulted by the orchestrator when it generates that
# wrapper, and by cli.submit_corvus when deciding what is submittable without a
# .hess already on disk.
#
#   carved-spring  - ORCA runs but its input deliberately omits "! AnFreq", so ORCA
#                    computes the energy only and writes no .hess. Superseded by
#                    hopt-spring and no longer reachable from any flag; kept so the
#                    run dirs on disk still parse back to what produced them.
#   hopt-spring    - as carved-spring, but ORCA optimizes the hydrogens instead of
#                    running a single point. Still no "! AnFreq".
#   caopt-spring   - no ORCA stage at all; the geometry arrives CA-fixed-optimized.
#   asis-spring    - no ORCA stage at all; the geometry is used exactly as handed in.
#
# Every no-ORCA mode must appear here: with no ORCA stage there is no other way to
# get a Hessian, and a mode left off this set reaches prepare-corvus with no .hess
# and no step that could have written one. A test ties the two sets together.
SPRING_HESSIAN_MODES = frozenset(
    {"carved-spring", "hopt-spring", "caopt-spring", "asis-spring"}
)

# Former name, kept so nothing importing it breaks mid-rename.
INTERP_HESSIAN_MODES = SPRING_HESSIAN_MODES

SCHEDULER_SUBMIT_COMMAND = _sched.SUBMIT_COMMAND
_default_scheduler = _sched.default_scheduler_name


extract_charge_multiplicity = _chem_xyz.extract_charge_multiplicity


def extract_ca_atoms(comments_file, atom_type=None, coord=None):
    """Extract atom numbers from comments file filtered by ATOM and COORD tags."""
    atom_numbers = []

    if not os.path.exists(comments_file):
        print(f"Warning: Comments file not found: {comments_file}")
        return atom_numbers

    atom_type_filter = atom_type.upper() if atom_type else None
    coord_filter = None if coord is None else str(bool(coord)).upper()

    with open(comments_file, 'r') as f:
        for line in f:
            # Extract atom number from "Atom 18: " format
            parts = line.split()
            if len(parts) < 2 or parts[0] != 'Atom':
                continue

            atom_match = re.search(r"\bATOM=([^\s#]+)", line, re.IGNORECASE)
            coord_match = re.search(r"\bCOORD=(TRUE|FALSE)", line, re.IGNORECASE)

            if atom_type_filter:
                if not atom_match or atom_match.group(1).upper() != atom_type_filter:
                    continue

            if coord_filter is not None:
                if not coord_match or coord_match.group(1).upper() != coord_filter:
                    continue

            atom_num = parts[1].rstrip(':')
            atom_numbers.append(atom_num)

    return atom_numbers


def _format_index_ranges(indices):
    """Format sorted indices as space-separated contiguous start:end ranges."""
    if not indices:
        return ""

    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        if start == prev:
            ranges.append(f"{{{start}}}")
        else:
            ranges.append(f"{{{start}:{prev}}}")
        start = prev = idx
    if start == prev:
        ranges.append(f"{{{start}}}")
    else:
        ranges.append(f"{{{start}:{prev}}}")
    return " ".join(ranges)


def _extract_comment_atom_metadata(line):
    """Parse atom index and key metadata tags from one comments-file line."""
    parts = line.split()
    if len(parts) < 2 or parts[0] != 'Atom':
        return None

    try:
        atom_idx = int(parts[1].rstrip(':'))
    except ValueError:
        return None

    coord_match = re.search(r"\bCOORD=(TRUE|FALSE)", line, re.IGNORECASE)
    atom_match = re.search(r"\bATOM=([^\s#]+)", line, re.IGNORECASE)
    bonded_match = re.search(r"\bBONDEDATOM=([^\s#]+)", line, re.IGNORECASE)

    coord_value = coord_match.group(1).upper() if coord_match else None
    atom_value = atom_match.group(1).upper() if atom_match else None
    bonded_value = bonded_match.group(1).upper() if bonded_match else None

    return atom_idx, coord_value, atom_value, bonded_value


def get_xtb_qm_indices(comments_file, include_nco_groups):
    """Return XTB QMATOMS [INDICES] based on COORD/ATOM/BONDEDATOM rules."""
    if not os.path.exists(comments_file):
        print(f"Warning: Comments file not found: {comments_file}")
        return None

    selected_indices = []
    nco = {"N", "C", "O", "CA"}

    with open(comments_file, 'r') as f:
        for line in f:
            metadata = _extract_comment_atom_metadata(line)
            if metadata is None:
                continue

            atom_idx, coord_value, atom_value, bonded_value = metadata
            if coord_value != "TRUE":
                continue

            in_nco_group = atom_value in nco or bonded_value in nco
            if include_nco_groups or not in_nco_group:
                selected_indices.append(atom_idx)

    if not selected_indices:
        return None

    return _format_index_ranges(sorted(set(selected_indices)))


def get_xtb_constrained_atoms(comments_file):
    """Return atoms to freeze for xtb-constrained mode."""
    if not os.path.exists(comments_file):
        print(f"Warning: Comments file not found: {comments_file}")
        return []

    selected = []
    seen = set()
    nco = {"N", "C", "O", "CA"}

    with open(comments_file, 'r') as f:
        for line in f:
            metadata = _extract_comment_atom_metadata(line)
            if metadata is None:
                continue

            atom_idx, coord_value, atom_value, bonded_value = metadata
            in_nco_group = atom_value in nco or bonded_value in nco

            should_constrain = (
                coord_value == "FALSE"
                or (coord_value == "TRUE" and in_nco_group)
            )

            if should_constrain and atom_idx not in seen:
                seen.add(atom_idx)
                selected.append(str(atom_idx))

    return selected


def get_coord_true_nco_atoms(comments_file):
    """Return atoms where COORD=TRUE and ATOM/BONDEDATOM is N/C/O/CA."""
    if not os.path.exists(comments_file):
        print(f"Warning: Comments file not found: {comments_file}")
        return []

    selected = []
    seen = set()
    nco = {"N", "C", "O", "CA"}

    with open(comments_file, 'r') as f:
        for line in f:
            metadata = _extract_comment_atom_metadata(line)
            if metadata is None:
                continue

            atom_idx, coord_value, atom_value, bonded_value = metadata
            in_nco_group = atom_value in nco or bonded_value in nco

            if coord_value == "TRUE" and in_nco_group and atom_idx not in seen:
                seen.add(atom_idx)
                selected.append(str(atom_idx))

    return selected


def clean_xyz_and_comments(input_path, clean_path=None, comments_path=None):
    """Clean XYZ and extract trailing comments to a sidecar file."""
    input_path = Path(input_path)
    # Use full filename stem so similarly prefixed XYZ files stay distinct.
    base = input_path.stem
    clean_path = (
        Path(clean_path)
        if clean_path is not None
        else input_path.with_name(f"{base}_clean.xyz")
    )
    comments_path = (
        Path(comments_path)
        if comments_path is not None
        else input_path.with_name(f"{base}_comments.txt")
    )

    try:
        lines = input_path.read_text().splitlines()
    except FileNotFoundError:
        print(f"File not found: {input_path}", file=sys.stderr)
        return None, None

    if len(lines) < 2:
        print("Input does not look like XYZ (missing header lines).", file=sys.stderr)
        return None, None

    header = lines[:2]
    atom_lines = lines[2:]

    cleaned = [header[0], header[1]]
    comment_lines = [header[1]]

    atom_index = 0
    for raw in atom_lines:
        if not raw.strip():
            continue

        # Split on # to separate inline comment.
        main_part, hash_part, comment = raw.partition("#")
        tokens = main_part.split()
        if len(tokens) < 4:
            print(f"Skipping invalid atom line: {raw}", file=sys.stderr)
            continue

        cleaned.append("{:<2} {:>12} {:>12} {:>12}".format(*tokens[:4]))

        if hash_part:
            comment_lines.append(f"Atom {atom_index}: # {comment.strip()}")
        else:
            comment_lines.append(f"Atom {atom_index}: (no comment)")

        atom_index += 1

    clean_path.write_text("\n".join(cleaned) + "\n")
    comments_path.write_text("\n".join(comment_lines) + "\n")

    # Ensure group read access for new files.
    for path in (clean_path, comments_path):
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IRGRP)

    return clean_path, comments_path


def extract_nprocs(orcar_input_file):
    """Extract number of processors from an ORCA input file (PAL or %pal nprocs)."""
    try:
        text = Path(orcar_input_file).read_text()
    except FileNotFoundError:
        return 1
    return extract_nprocs_from_text(text)


def extract_nprocs_from_text(text):
    """Extract number of processors from ORCA input text (PAL or %pal nprocs)."""
    lines = text.splitlines()

    pal_pattern = re.compile(r"\bPAL\s*([0-9]+)", re.IGNORECASE)
    nprocs_pattern = re.compile(r"\bnprocs\s*([0-9]+)", re.IGNORECASE)

    in_pal_block = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("!"):
            match = pal_pattern.search(line)
            if match:
                return int(match.group(1))

        if line.lower().startswith("%pal"):
            # Handle both styles:
            #   %pal nprocs 16
            #   %pal
            #     nprocs 16
            match = nprocs_pattern.search(line)
            if match:
                return int(match.group(1))
            in_pal_block = True
            continue

        if in_pal_block:
            match = nprocs_pattern.search(line)
            if match:
                return int(match.group(1))
            if line.lower() == "end":
                in_pal_block = False

    return 1


def generate_orca_job_script(
    template_root, scheduler, id_dir, input_filename, basename, nprocs, mem_gb
):
    """Generate a scheduler-specific ORCA job script from template."""
    scheduler_dir = template_root / f"{scheduler}-scripts"
    job_template = scheduler_dir / "orca-job.script"
    if not job_template.exists():
        print(f"  Error: Job template not found: {job_template}")
        return None

    env_path = resources.project_root() / ".env"

    # Slurm wants "64G"; PBS wants "64gb". [MEM] is scheduler-formatted here.
    mem_token = f"{mem_gb}gb" if scheduler == "pbs" else f"{mem_gb}G"

    generated_job = id_dir / f"generated-{basename}-orca.script"
    templates.render(
        job_template,
        generated_job,
        {
            "NPROCS": nprocs or 1,
            "BASENAME": basename,
            "INPUT_FILE": input_filename,
            "PIPELINE_ENV": env_path,
            "MEM": mem_token,
            # Regenerable ORCA scratch, excluded from the copy-back (fix #8).
            "SCRATCH_EXCLUDE": "|".join(config.SCRATCH_EXCLUDE_GLOBS),
        },
        executable=True,
        ensure_trailing_newline=True,
    )
    return generated_job


# ---------------------------------------------------------------------------
# ORCA stages
# ---------------------------------------------------------------------------
# A "stage" is one ORCA run: a mode (which template) plus whether it runs AnFreq.
# Normally there is exactly one. With --pre, cheaper stages run first and each
# subsequent stage starts from the previous stage's *optimized* geometry, chained
# by a scheduler afterok dependency.
#
# Each stage keeps its own run dir under the shared group dir, named for its mode
# exactly like a stand-alone run:
#
#     batch-root/CPA_G_OH_OH/
#       CPA_G_OH_OH-quick-ca-fixed/     <- stage 1, from <id>_clean.xyz
#       CPA_G_OH_OH-caopt-anfreq/       <- stage 2, from stage 1's <run_id>.xyz
#
# rather than several inputs sharing one dir. That way every existing scan
# (orca_check, layout.iter_id_dirs, the auto-rerun triage) sees each stage as the
# ordinary ORCA run it is, needs no changes, and can diagnose or rerun a stage on
# its own.


class StageResult:
    """What one prepared ORCA stage left on disk, and how to chain to the next."""

    def __init__(self, mode, anfreq, run_id, run_dir, job_script=None, geometry_out=None):
        self.mode = mode
        self.anfreq = anfreq
        self.run_id = run_id
        self.run_dir = run_dir
        # None for modes that run no ORCA at all (nothing to submit or chain).
        self.job_script = job_script
        # Where ORCA will write this stage's optimized geometry; what the next
        # stage's *xyzfile points at. It does not exist yet at prepare time --
        # the afterok dependency is what guarantees it does before stage N+1 runs.
        self.geometry_out = geometry_out
        self.job_id = None


# Filename of the sidecar one stage leaves behind naming the stage queued after it.
NEXT_STAGE_FILENAME = "{run_id}-next-stage.json"


def next_stage_path(run_dir, run_id):
    """Path of the "what runs after me" sidecar for an ORCA run directory."""
    return Path(run_dir) / NEXT_STAGE_FILENAME.format(run_id=run_id)


def _record_next_stage(previous, stage, dry_run):
    """Record, in *previous*'s run dir, which stage was queued afterok on it.

    Only the scheduler knows about the chain, and it forgets the moment a stage
    fails: the dependent job is killed as DependencyNeverSatisfied. When
    :mod:`xas_pipeline.cli.rerun_orca` then resubmits the failed stage, this
    sidecar is what tells it there was a next stage to re-queue onto the new job
    id -- otherwise a recoverable SCF hiccup in stage 1 silently drops stage 2.
    """
    if previous is None:
        return
    path = next_stage_path(previous.run_dir, previous.run_id)
    path.write_text(
        json.dumps(
            {
                "next_run_id": stage.run_id,
                "next_run_dir": str(stage.run_dir),
                "next_job_script": stage.job_script.name if stage.job_script else None,
                "next_job_id": stage.job_id,
                "dry_run": bool(dry_run),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o644)


def plan_stages(final_mode, pre_modes, anfreq_override):
    """Build the ordered stage list for one invocation.

    ``anfreq_override`` (None / True / False) applies to the FINAL stage only.
    Intermediate stages never run analytic frequencies: their geometry is about to
    move again, so the Hessian would describe a structure that is discarded, and
    AnFreq is by far the most expensive part of these runs.
    """
    stages = [(mode, False) for mode in pre_modes]
    final_anfreq = (
        anfreq_default_for(final_mode) if anfreq_override is None else anfreq_override
    )
    stages.append((final_mode, final_anfreq))
    return stages


def process_xyz_file(
    xyz_file,
    template_dir,
    output_root,
    dry_run,
    template_mode,
    scheduler,
    *,
    anfreq=None,
    geometry_source=None,
    depend_afterok=None,
):
    """Prepare (and unless *dry_run*, submit) one ORCA stage for one XYZ file.

    ``anfreq`` overrides :data:`MODE_ANFREQ_DEFAULT` for this stage; ``None`` uses
    the mode default. ``geometry_source`` is the geometry the ORCA input should
    read -- ``None`` means this stage's own cleaned copy of *xyz_file*, and a path
    means a previous stage's optimized output. ``depend_afterok`` is a list of job
    ids this stage's submission must wait on.

    Returns a :class:`StageResult` on success and ``None`` on failure (both keep
    the historical truthiness of the old bool return).
    """
    # Extract ID from full XYZ stem (no truncation at underscores).
    filename = os.path.basename(xyz_file)
    id_name = Path(filename).stem
    
    print(f"\nProcessing {filename} -> ID: {id_name}")

    # Every mode gets its own run dir, named "<id>-<mode>" and nested under a
    # group dir named for the structure, so caopt-anfreq/free/hopt-spring runs from the
    # same XYZ sit side by side instead of overwriting one another.
    output_base = layout.run_id_for(id_name, template_mode)
    id_dir = layout.run_dir_for(output_root, id_name, template_mode)
    id_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created directory: {id_dir}")
    
    # Copy XYZ file to the directory (leave original in place)
    dest_xyz = id_dir / filename
    shutil.copy2(xyz_file, dest_xyz)
    os.chmod(dest_xyz, 0o644)
    print(f"  Copied {filename} to {id_dir}/")

    if template_mode == "backbone":
        source_pc = Path(xyz_file).with_suffix(".pc")
        dest_pc = id_dir / f"{output_base}.pc"
        shutil.copy2(source_pc, dest_pc)
        os.chmod(dest_pc, 0o644)
        print(f"  Copied {source_pc.name} to {id_dir}/ as {dest_pc.name}")
    
    # Clean XYZ and write comments sidecar
    print("  Cleaning XYZ and extracting comments...")
    clean_path = id_dir / f"{output_base}_clean.xyz"
    comments_path = id_dir / f"{output_base}_comments.txt"
    clean_result, comments_result = clean_xyz_and_comments(
        dest_xyz,
        clean_path=clean_path,
        comments_path=comments_path,
    )
    if clean_result is None or comments_result is None:
        print("  Warning: Failed to clean XYZ or write comments file")
    
    # Get charge and multiplicity from XYZ header
    charge, multiplicity = extract_charge_multiplicity(xyz_file)
    if charge is None or multiplicity is None:
        header_lines = Path(xyz_file).read_text().splitlines()
        found_line2 = header_lines[1] if len(header_lines) >= 2 else "<missing>"
        print(
            "  ERROR: Missing charge and/or multiplicity in XYZ header (line 2) of "
            f"{xyz_file}.\n"
            "         Line 2 must contain a charge token (CHARGE=, CHARGE_ROUNDED=, "
            "or ROUNDED_CHARGE=) and a multiplicity token (MULTIPLICITY= or MULT=), "
            f"e.g. 'id={id_name} CHARGE_ROUNDED=0 MULTIPLICITY=1'.\n"
            f"         Found line 2: {found_line2!r}\n"
            "         No ORCA input or job script was generated for this structure."
        )
        return None

    # Modes with no ORCA stage stop here: the run dir is scaffolded (geometry
    # copied, comments extracted, charge/multiplicity validated) but no ORCA input
    # and no job script are written, so nothing downstream tries to submit one.
    #
    # The charge/multiplicity check above still runs even though FEFF never uses
    # either value: it is a cheap validity gate on the XYZ header, and the numbers
    # are recorded in the comments sidecar.
    #
    # <run_id>.xyz is written explicitly as the geometry of record. chem.xyz
    # .select_run_xyz prefers that name and only falls back to filtering the
    # directory by exclusion; with no ORCA to write it, the fallback would be the
    # only thing standing between CORVUS and the wrong coordinates. Naming it here
    # puts these modes on the same primary path as every optimized run.
    if template_mode in layout.NO_ORCA_MODES:
        geometry_of_record = id_dir / f"{output_base}.xyz"
        source_geometry = clean_result if clean_result is not None else dest_xyz
        shutil.copy2(source_geometry, geometry_of_record)
        os.chmod(geometry_of_record, 0o644)
        print(f"  Wrote geometry of record: {geometry_of_record.name}")
        print(
            f"    Mode '{template_mode}' runs no ORCA stage; Hessian will be "
            "interpolated from ligand spring models before CORVUS"
        )
        return StageResult(
            template_mode, False, output_base, id_dir, geometry_out=geometry_of_record
        )

    # Copy and modify template
    template_file = template_dir / TEMPLATE_FILE_BY_MODE[template_mode]
    output_file = id_dir / f"{output_base}.in"
    
    if not template_file.exists():
        print(f"  Error: Template file not found: {template_file}")
        return None
    
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Size-adaptive per-process memory. %MaxCore (an ORCA input directive) is
    # filled here via the [MAXCORE] placeholder, exactly like [CHARGE]/[MEM]; the
    # scheduler --mem (a Slurm/PBS directive) is derived from it and templated into
    # the job script. Both are sized from the atom count so large systems get
    # enough memory for the RIJCOSX/AnFreq step without over-requesting for small.
    nprocs_for_mem = extract_nprocs_from_text(template_content)
    natoms = count_atoms_xyz(clean_result) if clean_result is not None else None
    if natoms is None:
        natoms = count_atoms_xyz(dest_xyz)
    if natoms is None:
        maxcore_mb = orca_maxcore_mb(ORCA_MAXCORE_TIERS[-1][0])  # conservative default
        print("  Warning: could not determine atom count; using a conservative %MaxCore")
    else:
        maxcore_mb = orca_maxcore_mb(natoms)
    mem_gb = orca_mem_gb(nprocs_for_mem, maxcore_mb)
    had_maxcore_placeholder = '[MAXCORE]' in template_content

    # AnFreq: the mode default unless this stage was told otherwise.
    anfreq_enabled = anfreq_default_for(template_mode) if anfreq is None else bool(anfreq)

    # The geometry ORCA reads. Stage 1 reads its own cleaned copy of the input
    # XYZ; a later stage reads the previous stage's optimized output, which does
    # not exist yet and is guaranteed to by the afterok dependency. Absolute
    # either way, because the job script runs ORCA from a scratch directory.
    if geometry_source is None:
        geometry_path = id_dir / f"{output_base}_clean.xyz"
    else:
        geometry_path = Path(geometry_source).resolve()

    # Replace simple placeholders (INDICES / CA_ATOM handled specially below)
    template_content = templates.fill(template_content, {
        "CHARGE": charge,
        "MULTIPLICITY": multiplicity,
        "PDB_ID": output_base,
        "ID_DIR": id_dir,
        "MAXCORE": maxcore_mb,
        "ANFREQ": anfreq_directive(anfreq_enabled),
        "GEOMETRY": geometry_path,
    })

    if template_mode in {"xtb-free", "xtb-constrained"}:
        comments_file = id_dir / f"{output_base}_comments.txt"
        qm_atoms_spec = get_xtb_qm_indices(
            comments_file,
            include_nco_groups=(template_mode == "xtb-constrained"),
        )

        if qm_atoms_spec is not None:
            template_content = template_content.replace('[INDICES]', qm_atoms_spec)
        else:
            print("  Warning: Could not determine [INDICES] for xtb QMATOMS")
    
    constrained_atoms = []
    if template_mode in {"caopt-anfreq", "quick-ca-fixed", "backbone", "xtb-constrained"}:
        # Extract constrained atoms from comments file
        comments_file = id_dir / f"{output_base}_comments.txt"
        if template_mode in {"caopt-anfreq", "quick-ca-fixed"}:
            constrained_atoms = extract_ca_atoms(comments_file, atom_type="CA")
        elif template_mode == "xtb-constrained":
            constrained_atoms = get_xtb_constrained_atoms(comments_file)
        elif template_mode == "backbone":
            constrained_atoms = get_coord_true_nco_atoms(comments_file)
        else:
            constrained_atoms = []

        # Build constraint lines based on [CA_ATOM] placeholder
        ca_constraints = []
        for atom_num in constrained_atoms:
            constraint_line = f"  {{C {atom_num} C}}  # Freeze all coordinates (X, Y, Z) of atom {atom_num}"
            ca_constraints.append(constraint_line)

        # Replace the single [CA_ATOM] line with multiple lines (one per CA atom)
        if '[CA_ATOM]' in template_content:
            if ca_constraints:
                template_lines = template_content.splitlines()
                replaced_lines = []
                replaced = False
                for line in template_lines:
                    if '[CA_ATOM]' in line and not replaced:
                        replaced_lines.extend(ca_constraints)
                        replaced = True
                    else:
                        replaced_lines.append(line)
                template_content = "\n".join(replaced_lines)
            else:
                print("  Warning: No constrained atoms found to replace [CA_ATOM]")
        elif ca_constraints:
            print("  Warning: No [CA_ATOM] placeholder found in template")
            print(f"  Found {len(ca_constraints)} atoms to constrain")
    
    # Ensure final newline (some ORCA versions can misread the last line without it)
    if not template_content.endswith("\n"):
        template_content += "\n"

    # Write modified template
    with open(output_file, 'w') as f:
        f.write(template_content)
    os.chmod(output_file, 0o644)
    print(f"  Created input file: {output_file}")
    print(f"    CHARGE={charge}, MULTIPLICITY={multiplicity}")
    default_anfreq = anfreq_default_for(template_mode)
    anfreq_note = "" if anfreq_enabled == default_anfreq else " (overridden)"
    print(f"    AnFreq={'on' if anfreq_enabled else 'off'}{anfreq_note}")
    if geometry_source is not None:
        print(f"    Starting geometry: {geometry_path} (previous ORCA stage)")
    if anfreq_enabled and template_mode in SPRING_HESSIAN_MODES:
        print(
            "    Warning: this mode's Hessian is interpolated from ligand spring "
            "models before CORVUS, so the AnFreq .hess would be overwritten"
        )
    if had_maxcore_placeholder:
        print(
            f"    Atoms={natoms if natoms is not None else '?'}: "
            f"%MaxCore={maxcore_mb} MB/proc x {nprocs_for_mem} proc -> --mem={mem_gb}G"
        )
    else:
        print(
            f"    Warning: template {template_file.name} has no [MAXCORE] placeholder; "
            f"no %MaxCore set (scheduler --mem still requested as {mem_gb}G)"
        )
    if template_mode == "hopt-anfreq":
        print("    Only optimizing hydrogen atoms")
    elif template_mode == "carved-anfreq":
        print("    Running single-point style template")
    elif template_mode == "no-constraints":
        print("    Running unconstrained optimization template")
    elif template_mode == "backbone":
        print("    Running backbone point-charge template")
    elif template_mode == "xtb-free":
        print("    Running XTB free template with COORD=TRUE non-CA/N/C/O QM region")
    elif template_mode == "quick":
        print("    Running quick optimization (no CA fixing)")
    elif template_mode == "quick-ca-fixed":
        print(f"    Quick CA-fixed optimization; CA atoms to freeze: {len(constrained_atoms)}")
    elif template_mode == "xtb-constrained":
        print("    Running XTB constrained template with COORD=TRUE full QM region")
    elif template_mode == "carved-spring":
        print("    Single point for the energy (no AnFreq); Hessian will be "
              "interpolated from ligand spring models before CORVUS")
    elif template_mode == "hopt-spring":
        print("    Optimizing hydrogen atoms only (no AnFreq); Hessian will be "
              "interpolated from ligand spring models before CORVUS")
    else:
        print(f"    CA atoms to freeze: {len(constrained_atoms)}")
    
    # Generate and (optionally) submit scheduler job script. nprocs was already
    # parsed from the input text above (nprocs_for_mem); reuse it.
    nprocs = nprocs_for_mem
    generated_job = generate_orca_job_script(
        template_dir,
        scheduler,
        id_dir,
        f"{output_base}.in",
        output_base,
        nprocs,
        mem_gb,
    )
    if generated_job is None:
        return None

    stage = StageResult(
        template_mode,
        anfreq_enabled,
        output_base,
        id_dir,
        job_script=generated_job,
        # ORCA names its outputs after the input stem, so the optimized geometry
        # lands here and the job script copies it back out of scratch.
        geometry_out=id_dir / f"{output_base}.xyz",
    )

    sched = _sched.get_scheduler(scheduler)
    dep_flag = (
        sched.dependency_flag("afterok", [str(j) for j in depend_afterok])
        if depend_afterok
        else []
    )
    print(f"  Generated job script: {generated_job.name}")
    if dry_run:
        dep_note = f" (would depend afterok:{':'.join(map(str, depend_afterok))})" if depend_afterok else ""
        print(f"  Dry run: generated {generated_job.name} (submission skipped){dep_note}")
        return stage

    submit_cmd = [sched.submit_command, *dep_flag, generated_job.name]
    if depend_afterok:
        print(f"  Depends on: afterok:{':'.join(map(str, depend_afterok))}")
    print(f"  Submitting with {sched.submit_command}...")
    result = subprocess.run(
        submit_cmd,
        cwd=id_dir,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(f"  submission output:\n{result.stdout}")
    if result.stderr:
        print(f"  submission stderr:\n{result.stderr}")
    if result.returncode != 0:
        print(f"  Warning: job submission failed (exit code {result.returncode})")
        # A stage that never reached the queue has no job id for the next stage to
        # wait on. Reporting failure stops the chain here rather than submitting a
        # dependent stage that would start from a geometry nothing will write.
        return None

    try:
        stage.job_id = sched.parse_job_id(result.stdout)
    except ValueError as exc:
        # Submitted, but we cannot name the job. Harmless on its own; main() turns
        # it into a hard stop only if a further stage needed to depend on it.
        print(f"  Warning: {exc}")

    return stage


def main():
    parser = argparse.ArgumentParser(
        description='Process XYZ files for ORCA calculations with specified ligand composition.'
    )
    parser.add_argument('path', type=str, help='Directory containing XYZ files or a single XYZ file')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for ID folders (default: parent of input XYZ directory)')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--H', action='store_true', help='Use orca-template-h-only.in')
    mode_group.add_argument('--single', action='store_true', help='Use orca-template-single-point.in')
    mode_group.add_argument('--free', action='store_true', help='Use orca-template-no-constraints.in')
    mode_group.add_argument('--backbone', action='store_true', help='Use orca-template-backbone-charges.in (requires matching .pc file)')
    mode_group.add_argument('--quick', action='store_true', help='Use orca-template-quick.in (quick optimization, no CA fixing)')
    mode_group.add_argument('--quick-ca-fixed', action='store_true', help='Use orca-template-quick-ca-fixed.in (quick CA-fixed optimization)')
    mode_group.add_argument('--xtb-free', action='store_true', help='Use orca-template-xtb-free.in (COORD=TRUE non-CA/N/C/O QM region)')
    mode_group.add_argument('--xtb-constrained', action='store_true', help='Use orca-template-xtb-constrained.in (COORD=TRUE full QM region with constraints)')
    mode_group.add_argument('--interp', action='store_true', help='Use orca-template-interp-hopt.in (optimize hydrogens only, no AnFreq; the Hessian is interpolated from ligand spring models instead)')
    mode_group.add_argument('--interp-raw', action='store_true', help='No ORCA stage at all: the geometry is used as handed in and the Hessian is interpolated from ligand spring models. Nothing checks the geometry before FEFF does.')
    mode_group.add_argument('--caopt-spring', action='store_true', help='As --interp-raw, for input geometries that are already CA-fixed-optimized (e.g. a caopt-anfreq result): no ORCA stage, spring-model Hessian, run dirs named <id>-caopt-spring.')
    parser.add_argument(
        '--pre',
        action='append',
        default=[],
        metavar='MODE',
        choices=sorted(TEMPLATE_FILE_BY_MODE),
        help=(
            'Run MODE as a cheaper ORCA stage BEFORE the selected mode, chained '
            'afterok, with the next stage starting from this one\'s optimized '
            'geometry. Repeatable; stages run in the order given. Each stage gets '
            'its own <id>-<mode> run dir. Pre-stages never run AnFreq. '
            'Typical: --pre quick-ca-fixed (B97-3c pre-optimization before the '
            'default PBE0 caopt-anfreq).'
        ),
    )
    anfreq_group = parser.add_mutually_exclusive_group()
    anfreq_group.add_argument(
        '--anfreq', dest='anfreq', action='store_true', default=None,
        help='Force "! AnFreq" ON for the final stage, whatever its mode default is.',
    )
    anfreq_group.add_argument(
        '--no-anfreq', dest='anfreq', action='store_false', default=None,
        help=(
            'Force "! AnFreq" OFF for the final stage: the optimization runs but no '
            '.hess is produced (so a later CORVUS stage has no ORCA Hessian).'
        ),
    )
    parser.add_argument('-n', '--dry-run', action='store_true', help='Generate job script but skip submission')
    parser.add_argument(
        '--scheduler',
        choices=sorted(SCHEDULER_SUBMIT_COMMAND),
        default=_default_scheduler(),
        help='Scheduler backend used for templates and submission command (default: pbs)',
    )

    args = parser.parse_args()

    template_mode = "caopt-anfreq"
    if args.H:
        template_mode = "hopt-anfreq"
    elif args.single:
        template_mode = "carved-anfreq"
    elif args.free:
        template_mode = "no-constraints"
    elif args.backbone:
        template_mode = "backbone"
    elif args.quick:
        template_mode = "quick"
    elif args.quick_ca_fixed:
        template_mode = "quick-ca-fixed"
    elif args.xtb_free:
        template_mode = "xtb-free"
    elif args.xtb_constrained:
        template_mode = "xtb-constrained"
    elif args.interp:
        # --interp now means "optimize the hydrogens", not "single point". The old
        # single-point behaviour is still the `interp` mode (its template and
        # suffix remain registered so existing run dirs keep their meaning), but no
        # flag produces it any more.
        template_mode = "hopt-spring"
    elif args.interp_raw:
        template_mode = "asis-spring"
    elif args.caopt_spring:
        template_mode = "caopt-spring"

    # --pre validation. All of these are user errors that would otherwise produce a
    # chain that cannot run, so fail before anything is written to disk.
    if args.pre:
        if template_mode in layout.NO_ORCA_MODES:
            print(
                f"ERROR: --pre cannot be combined with mode '{template_mode}', which "
                "runs no ORCA stage; there would be nothing for the pre-stage to feed."
            )
            sys.exit(1)
        chain = list(args.pre) + [template_mode]
        duplicates = [m for i, m in enumerate(chain) if m in chain[:i]]
        if duplicates:
            print(
                f"ERROR: mode(s) {sorted(set(duplicates))} appear more than once in the "
                f"stage chain {' -> '.join(chain)}. Each stage needs its own <id>-<mode> "
                "run dir, so a mode can only appear once."
            )
            sys.exit(1)

    stages = plan_stages(template_mode, args.pre, args.anfreq)

    if len(stages) > 1:
        print(
            "Stage chain: "
            + " -> ".join(
                f"{mode}({'AnFreq' if af else 'no AnFreq'})" for mode, af in stages
            )
        )
        print("  Each stage starts from the previous stage's optimized geometry (afterok).")
        print("  Pre-stages never run AnFreq; --anfreq/--no-anfreq applies to the final stage.")
    else:
        print(f"Template mode: {template_mode}")
        mode, final_anfreq = stages[0]
        if args.anfreq is not None:
            print(f"AnFreq: {'on' if final_anfreq else 'off'} (overriding the {mode} default)")
    print(f"Scheduler: {args.scheduler}")
    
    # Templates ship as package data (orca-templates/, {slurm,pbs}-scripts/).
    template_dir = resources.template_root()
    
    # Find all XYZ files in the specified directory
    target_path = Path(args.path)
    target_path_was_absolute = target_path.is_absolute()
    if not target_path.exists():
        print(f"ERROR: Path not found: {target_path}")
        sys.exit(1)

    target_path_resolved = target_path.resolve()

    if target_path.is_file():
        if target_path.suffix.lower() != ".xyz":
            print(f"ERROR: File is not an XYZ file: {target_path}")
            sys.exit(1)
        xyz_files = [target_path_resolved]
        input_base_dir = target_path_resolved.parent
    else:
        xyz_files = list(target_path.glob("*.xyz"))
        input_base_dir = target_path_resolved

        if not xyz_files:
            print(f"No XYZ files found in {target_path}")
            sys.exit(0)

    # Determine output root (default to one directory up from input XYZ directory)
    if args.out_dir:
        raw_out_dir = Path(args.out_dir)
        if not raw_out_dir.is_absolute() and str(raw_out_dir).startswith("home/"):
            fixed_out_dir = Path("/") / raw_out_dir
            print(
                "Warning: --out-dir looks like an absolute path missing a leading '/'. "
                f"Using: {fixed_out_dir}"
            )
            output_root = fixed_out_dir.resolve()
        else:
            output_root = raw_out_dir.resolve()
            if not raw_out_dir.is_absolute():
                print(
                    "Warning: --out-dir is relative. "
                    f"It resolves to: {output_root}"
                )
    else:
        output_root = input_base_dir.parent.resolve()
        print(f"Defaulting --out-dir to parent of input XYZ directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if input_base_dir != output_root:
        output_was_absolute = Path(args.out_dir).is_absolute() if args.out_dir else True
        print(
            "Warning: Input XYZ location and output directory differ: "
            f"input={input_base_dir} ({'absolute' if target_path_was_absolute else 'relative'} path), "
            f"output={output_root} ({'absolute' if output_was_absolute else 'relative'} path)"
        )

    if template_mode == "backbone":
        missing_pc = [str(x.with_suffix('.pc')) for x in xyz_files if not x.with_suffix('.pc').exists()]
        if missing_pc:
            print("ERROR: --backbone requires a matching .pc file for each XYZ input.")
            for pc in missing_pc:
                print(f"  Missing: {pc}")
            sys.exit(1)
    
    print(f"\nFound {len(xyz_files)} XYZ file(s) to process")
    print(f"Output root: {output_root}")
    
    failed_files = []
    for xyz_file in xyz_files:
        previous = None
        for index, (mode, stage_anfreq) in enumerate(stages):
            stage = process_xyz_file(
                xyz_file,
                template_dir,
                output_root,
                args.dry_run,
                mode,
                args.scheduler,
                anfreq=stage_anfreq,
                # Stage 1 starts from the input XYZ; every later stage starts from
                # the geometry the stage before it will have optimized.
                geometry_source=None if previous is None else previous.geometry_out,
                depend_afterok=(
                    [previous.job_id]
                    if previous is not None and previous.job_id is not None
                    else None
                ),
            )
            if stage is None:
                failed_files.append(xyz_file)
                break
            if index + 1 < len(stages) and not args.dry_run and stage.job_id is None:
                # Without the parent's job id the next stage would be submitted with
                # no dependency and would read a geometry file that does not exist
                # yet. Stop the chain and report it rather than queue a doomed job.
                print(
                    f"  ERROR: stage '{mode}' has no scheduler job id, so the "
                    f"remaining stage(s) cannot be chained onto it. Submit them by "
                    f"hand once {stage.run_id} finishes."
                )
                failed_files.append(xyz_file)
                break
            _record_next_stage(previous, stage, args.dry_run)
            previous = stage

    if failed_files:
        print(
            f"\nERROR: {len(failed_files)} of {len(xyz_files)} XYZ file(s) failed to "
            "produce an ORCA job script (see errors above):"
        )
        for f in failed_files:
            print(f"  - {f}")
        sys.exit(1)

    print("\nProcessing complete!")

if __name__ == "__main__":  # `python -m xas_pipeline...` entry
    raise SystemExit(main())
