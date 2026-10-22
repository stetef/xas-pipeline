"""Prepare ORCA runs: per structure and mode, a run dir with the ORCA input and job script.

Each structure gets ``<out_root>/<id>/<id>-<mode>/`` holding a copy of the XYZ,
``<run_id>_clean.xyz`` (coordinates only), ``<run_id>_comments.txt`` (the
per-atom ``# ...`` tags), ``<run_id>.in`` and ``generated-<run_id>-orca.script``.
Modes without an ORCA stage get the scaffolding plus ``<run_id>.xyz`` as the
geometry of record, and no input or job script.

``--pre MODE`` chains cheaper ORCA stages before the final one: each stage has its
own run dir, starts from the previous stage's optimized geometry, is submitted
``afterok`` on it, and never runs AnFreq. A ``<run_id>-next-stage.json`` sidecar
records the chain so a rerun of a failed stage can re-queue the next one.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from xas_pipeline import batch_log, jobscripts, layout, modes, resources, templates
from xas_pipeline.chem import xyz as chem_xyz
from xas_pipeline.logs import setup_cli_logging
from xas_pipeline.scheduler import Scheduler, add_scheduler_argument, get_scheduler

log = logging.getLogger(__name__)


class PrepError(Exception):
    """A structure cannot be prepared (bad input); nothing should be submitted for it."""


# ── Memory sizing ─────────────────────────────────────────────────────────────
# %MaxCore is per process (MB) and sized by atom count, so large clusters get
# enough for the RIJCOSX/AnFreq step without small ones over-requesting. The
# scheduler request makes nprocs * MaxCore ~70 % of the allocation (ORCA exceeds
# MaxCore transiently and needs memory outside it), with a floor.
ORCA_MAXCORE_TIERS = [(50, 1000), (90, 1800), (140, 2800), (200, 4200)]  # (natoms <=, MB)
ORCA_MAXCORE_ABOVE = 5600
ORCA_MEM_MAXCORE_FRACTION = 0.70
ORCA_MEM_FLOOR_GB = 16


def orca_maxcore_mb(natoms: int) -> int:
    for upper, maxcore in ORCA_MAXCORE_TIERS:
        if natoms <= upper:
            return maxcore
    return ORCA_MAXCORE_ABOVE


def orca_mem_gb(nprocs: int, maxcore_mb: int) -> int:
    nprocs = max(1, int(nprocs or 1))
    return max(ORCA_MEM_FLOOR_GB, math.ceil(nprocs * maxcore_mb / ORCA_MEM_MAXCORE_FRACTION / 1024))


def extract_nprocs_from_text(text: str) -> int:
    """Process count from ``! PALn`` or ``%pal nprocs n [end]``; 1 if absent."""
    pal = re.compile(r"\bPAL\s*([0-9]+)", re.IGNORECASE)
    nprocs = re.compile(r"\bnprocs\s*([0-9]+)", re.IGNORECASE)
    in_pal = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("!") and (m := pal.search(line)):
            return int(m.group(1))
        if line.lower().startswith("%pal"):
            if m := nprocs.search(line):
                return int(m.group(1))
            in_pal = True
            continue
        if in_pal:
            if m := nprocs.search(line):
                return int(m.group(1))
            if line.lower() == "end":
                in_pal = False
    return 1


# ── AnFreq ────────────────────────────────────────────────────────────────────
# Every template carries an [ANFREQ] placeholder; the "on" text is the block the
# templates used to hard-code.
ANFREQ_ON_TEXT = "# Do frequency calculation (produce .hess file)\n! AnFreq"
ANFREQ_OFF_TEXT = "# No frequency calculation: this run produces no .hess file"


def anfreq_directive(enabled: bool) -> str:
    return ANFREQ_ON_TEXT if enabled else ANFREQ_OFF_TEXT


# ── XYZ cleaning and atom tags ────────────────────────────────────────────────


def clean_xyz_and_comments(input_path: Path, clean_path: Path, comments_path: Path) -> None:
    """Split an annotated XYZ into coordinates and a per-atom tag sidecar.

    ``<sym> x y z  # TAGS`` lines become ``<sym> x y z`` in *clean_path* and
    ``Atom <i>: # TAGS`` (or ``(no comment)``) in *comments_path*, whose first
    line repeats the XYZ comment line.
    """
    lines = Path(input_path).read_text().splitlines()
    if len(lines) < 2:
        raise PrepError(f"{input_path} does not look like XYZ (missing header lines)")
    cleaned, comments = lines[:2], [lines[1]]
    for raw in lines[2:]:
        if not raw.strip():
            continue
        main_part, hash_part, comment = raw.partition("#")
        tokens = main_part.split()
        if len(tokens) < 4:
            log.warning("Skipping invalid atom line: %s", raw)
            continue
        comments.append(f"Atom {len(cleaned) - 2}: " + (f"# {comment.strip()}" if hash_part else "(no comment)"))
        cleaned.append("{:<2} {:>12} {:>12} {:>12}".format(*tokens[:4]))
    for path, content in ((clean_path, cleaned), (comments_path, comments)):
        path.write_text("\n".join(content) + "\n")
        path.chmod(path.stat().st_mode | stat.S_IRGRP)


@dataclass(frozen=True)
class AtomTag:
    index: int
    atom: str | None  # ATOM= (PDB atom name)
    bonded_atom: str | None  # BONDEDATOM= (for H: the heavy atom it sits on)
    coord: str | None  # COORD=TRUE/FALSE (coordinates the metal)


_BACKBONE = {"N", "C", "O", "CA"}


def _tag(name: str, line: str) -> str | None:
    match = re.search(rf"\b{name}=([^\s#]+)", line, re.IGNORECASE)
    return match.group(1).upper() if match else None


def read_atom_tags(comments_path: Path) -> list[AtomTag]:
    """The ``Atom <i>:`` lines of a comments sidecar, with their tags."""
    tags = []
    for line in Path(comments_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "Atom":
            continue
        try:
            index = int(parts[1].rstrip(":"))
        except ValueError:
            continue
        tags.append(AtomTag(index, _tag("ATOM", line), _tag("BONDEDATOM", line), _tag("COORD", line)))
    return tags


def _on_backbone(tag: AtomTag) -> bool:
    return tag.atom in _BACKBONE or tag.bonded_atom in _BACKBONE


def constrained_atoms(tags: list[AtomTag], rule: str) -> list[int]:
    """Atoms to freeze, in file order, for a mode's ``constraints`` rule."""
    if rule == "ca":
        chosen = [t.index for t in tags if t.atom == "CA"]
    elif rule == "backbone":
        chosen = [t.index for t in tags if t.coord == "TRUE" and _on_backbone(t)]
    elif rule == "xtb":
        chosen = [t.index for t in tags if t.coord == "FALSE" or (t.coord == "TRUE" and _on_backbone(t))]
    else:
        raise ValueError(f"Unknown constraint rule {rule!r}")
    return list(dict.fromkeys(chosen))


def qm_region_ranges(tags: list[AtomTag], rule: str) -> str:
    """The xTB QMATOMS index ranges (``{0:3} {7}``) for a mode's ``qm_region`` rule."""
    indices = sorted({
        t.index for t in tags
        if t.coord == "TRUE" and (rule == "all" or not _on_backbone(t))
    })
    ranges, start = [], None
    for position, idx in enumerate(indices):
        if start is None:
            start = idx
        if position + 1 == len(indices) or indices[position + 1] != idx + 1:
            ranges.append(f"{{{start}}}" if start == idx else f"{{{start}:{idx}}}")
            start = None
    return " ".join(ranges)


# ── One ORCA stage ────────────────────────────────────────────────────────────


@dataclass
class Stage:
    """One prepared run: its dir, job script (None without ORCA) and output geometry."""

    mode: str
    anfreq: bool
    run_id: str
    run_dir: Path
    job_script: Path | None
    #: Where this stage's optimized geometry will be: the next stage's input.
    geometry_out: Path
    job_id: str | None = None


def render_orca_input(template_text: str, *, mode: modes.Mode, tags: list[AtomTag], tokens: dict) -> str:
    """Fill an ORCA template: simple tokens, the xTB QM region, frozen atoms."""
    text = templates.fill(template_text, tokens)
    if mode.qm_region:
        spec = qm_region_ranges(tags, mode.qm_region)
        if not spec:
            raise PrepError(f"no COORD=TRUE atoms for the {mode.name} QM region ([INDICES])")
        text = text.replace("[INDICES]", spec)
    if mode.constraints and "[CA_ATOM]" in text:
        atoms = constrained_atoms(tags, mode.constraints)
        if not atoms:
            raise PrepError(f"no atoms match the {mode.name} constraint rule ([CA_ATOM])")
        lines = text.splitlines()
        at = next(i for i, line in enumerate(lines) if "[CA_ATOM]" in line)
        lines[at:at + 1] = [
            f"  {{C {idx} C}}  # Freeze all coordinates (X, Y, Z) of atom {idx}" for idx in atoms
        ]
        text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"


def prepare_stage(
    xyz_file: Path,
    out_root: Path,
    mode_name: str,
    sched: Scheduler,
    *,
    anfreq: bool | None = None,
    geometry_source: Path | None = None,
) -> Stage:
    """Write one structure's run dir for *mode_name*. Raises :class:`PrepError`.

    *anfreq* overrides the mode default. *geometry_source* is the geometry ORCA
    reads: None for this stage's own cleaned input, else a previous stage's
    optimized output (which the afterok dependency guarantees will exist).
    """
    mode = modes.get(mode_name)
    xyz_file = Path(xyz_file)
    id_name = xyz_file.stem
    run_id = layout.run_id_for(id_name, mode.name)
    run_dir = layout.run_dir_for(out_root, id_name, mode.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s -> %s", xyz_file.name, run_dir)

    dest_xyz = run_dir / xyz_file.name
    shutil.copy2(xyz_file, dest_xyz)
    os.chmod(dest_xyz, 0o644)
    if mode.needs_point_charges:
        dest_pc = run_dir / f"{run_id}.pc"
        shutil.copy2(xyz_file.with_suffix(".pc"), dest_pc)
        os.chmod(dest_pc, 0o644)

    clean_path = run_dir / f"{run_id}_clean.xyz"
    comments_path = run_dir / f"{run_id}_comments.txt"
    clean_xyz_and_comments(dest_xyz, clean_path, comments_path)

    charge, multiplicity = chem_xyz.extract_charge_multiplicity(xyz_file)
    if charge is None or multiplicity is None:
        lines = xyz_file.read_text().splitlines()
        raise PrepError(
            f"missing charge and/or multiplicity on line 2 of {xyz_file}: it needs a charge "
            "token (CHARGE=, CHARGE_ROUNDED= or ROUNDED_CHARGE=) and a multiplicity token "
            f"(MULTIPLICITY= or MULT=), e.g. 'id={id_name} CHARGE_ROUNDED=0 MULTIPLICITY=1'. "
            f"Found: {lines[1] if len(lines) > 1 else '<missing>'!r}"
        )

    if not mode.template:
        # No ORCA: name the geometry of record explicitly, so chem.xyz.select_run_xyz
        # takes its primary path rather than filtering the directory.
        geometry = run_dir / f"{run_id}.xyz"
        shutil.copy2(clean_path, geometry)
        os.chmod(geometry, 0o644)
        log.info("  no ORCA stage (%s); geometry of record %s", mode.name, geometry.name)
        return Stage(mode.name, False, run_id, run_dir, None, geometry)

    template_text = (resources.template_root() / "orca-templates" / mode.template).read_text()
    nprocs = extract_nprocs_from_text(template_text)
    natoms = chem_xyz.count_atoms_xyz(clean_path) or chem_xyz.count_atoms_xyz(dest_xyz)
    maxcore_mb = orca_maxcore_mb(natoms if natoms is not None else ORCA_MAXCORE_TIERS[-1][0])
    mem_gb = orca_mem_gb(nprocs, maxcore_mb)
    anfreq_on = mode.anfreq if anfreq is None else bool(anfreq)
    geometry_path = clean_path if geometry_source is None else Path(geometry_source).resolve()

    try:
        text = render_orca_input(
            template_text,
            mode=mode,
            tags=read_atom_tags(comments_path),
            tokens={
                "CHARGE": charge,
                "MULTIPLICITY": multiplicity,
                "PDB_ID": run_id,
                "ID_DIR": run_dir,
                "MAXCORE": maxcore_mb,
                "ANFREQ": anfreq_directive(anfreq_on),
                "GEOMETRY": geometry_path,
            },
        )
    except PrepError as exc:
        raise PrepError(f"{xyz_file.name}: {exc}") from None
    input_path = run_dir / f"{run_id}.in"
    input_path.write_text(text)
    os.chmod(input_path, 0o644)

    job_script = jobscripts.orca_job(sched, run_dir, run_id, nprocs=nprocs, mem_gb=mem_gb)
    log.info(
        "  %s: charge %s, multiplicity %s, AnFreq %s, %s atoms -> %%MaxCore %d MB x %d proc, %d GB",
        input_path.name, charge, multiplicity, "on" if anfreq_on else "off",
        natoms if natoms is not None else "?", maxcore_mb, nprocs, mem_gb,
    )
    if geometry_source is not None:
        log.info("  starting geometry: %s (previous stage)", geometry_path)
    if anfreq_on and mode.spring_hessian:
        log.warning("  AnFreq is on, but this mode's spring Hessian will replace the .hess")
    return Stage(mode.name, anfreq_on, run_id, run_dir, job_script, run_dir / f"{run_id}.xyz")


# ── Stage chains ──────────────────────────────────────────────────────────────


def plan_stages(final_mode: str, pre_modes=(), anfreq_override: bool | None = None) -> list[tuple[str, bool]]:
    """``[(mode, anfreq), ...]``: pre-stages never run AnFreq; the override hits the last."""
    final_anfreq = modes.get(final_mode).anfreq if anfreq_override is None else anfreq_override
    return [(mode, False) for mode in pre_modes] + [(final_mode, final_anfreq)]


def next_stage_path(run_dir: Path, run_id: str) -> Path:
    """The sidecar naming the stage queued after this run (read by xas-rerun-orca)."""
    return Path(run_dir) / f"{run_id}-next-stage.json"


def _record_next_stage(previous: Stage, stage: Stage, dry_run: bool) -> None:
    path = next_stage_path(previous.run_dir, previous.run_id)
    path.write_text(json.dumps({
        "next_run_id": stage.run_id,
        "next_run_dir": str(stage.run_dir),
        "next_job_script": stage.job_script.name if stage.job_script else None,
        "next_job_id": stage.job_id,
        "dry_run": bool(dry_run),
    }, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)


def validate(xyz_files: list[Path], final_mode: str, pre_modes=()) -> None:
    """Reject input that cannot produce a runnable chain, before anything is written."""
    if pre_modes and final_mode in modes.NO_ORCA_MODES:
        raise PrepError(f"--pre cannot feed mode '{final_mode}', which runs no ORCA stage")
    chain = [*pre_modes, final_mode]
    duplicates = sorted({m for i, m in enumerate(chain) if m in chain[:i]})
    if duplicates:
        raise PrepError(f"mode(s) {duplicates} appear twice in {' -> '.join(chain)}; each stage needs its own run dir")
    if any(modes.get(m).needs_point_charges for m in chain):
        missing = [str(x.with_suffix(".pc")) for x in xyz_files if not x.with_suffix(".pc").exists()]
        if missing:
            raise PrepError("point-charge mode needs a .pc file per XYZ; missing:\n  " + "\n  ".join(missing))


def prepare_chain(
    xyz_file: Path,
    out_root: Path,
    final_mode: str,
    sched: Scheduler,
    *,
    pre_modes=(),
    anfreq: bool | None = None,
    submit: bool = False,
) -> list[Stage]:
    """Prepare (and with *submit*, queue afterok-chained) every stage for one structure.

    Submissions are recorded in ``<out_root>/batch-jobs.log``, which the ORCA
    job's auto-rerun hook needs to find and supersede the run's jobs.
    """
    stages: list[Stage] = []
    previous = None
    for index, (mode_name, stage_anfreq) in enumerate(plan_stages(final_mode, pre_modes, anfreq)):
        stage = prepare_stage(
            xyz_file, out_root, mode_name, sched, anfreq=stage_anfreq,
            geometry_source=previous.geometry_out if previous else None,
        )
        if submit and stage.job_script is not None:
            afterok = [previous.job_id] if previous and previous.job_id else []
            stage.job_id = batch_log.submit_and_log(
                sched, Path(out_root) / batch_log.LOG_NAME, f"orca-{stage.run_id}", stage.job_script,
                stage.run_dir, afterok=afterok,
            )
            log.info("  submitted %s: job %s%s", stage.job_script.name, stage.job_id,
                     f" (afterok {afterok[0]})" if afterok else "")
        if previous is not None:
            _record_next_stage(previous, stage, dry_run=not submit)
        stages.append(stage)
        previous = stage
        if submit and index + 1 < len(plan_stages(final_mode, pre_modes, anfreq)) and stage.job_id is None:
            raise PrepError(f"stage {stage.run_id} has no job id to chain the next stage onto")
    return stages


def discover_xyz_files(path: Path) -> tuple[list[Path], Path]:
    """Sorted ``.xyz`` inputs at *path* (a file or a directory) and their directory."""
    path = Path(path).expanduser()
    if not path.exists():
        raise PrepError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".xyz":
            raise PrepError(f"Not an XYZ file: {path}")
        return [path.resolve()], path.resolve().parent
    files = sorted(p.resolve() for p in path.glob("*.xyz"))
    if not files:
        raise PrepError(f"No .xyz files in {path}")
    return files, path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare (and submit) ORCA runs for XYZ structures.")
    parser.add_argument("path", type=Path, help="An XYZ file or a directory of them")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Batch root for the run dirs (default: parent of the XYZ directory)")
    modes.add_mode_arguments(parser)
    parser.add_argument(
        "--pre", action="append", default=[], metavar="MODE", choices=list(modes.ORCA_MODES),
        help="Run MODE as a cheaper ORCA stage first (repeatable, in order); each next stage "
             "starts from its optimized geometry. Pre-stages never run AnFreq. "
             "Typical: --pre quick-ca-fixed.",
    )
    anfreq = parser.add_mutually_exclusive_group()
    anfreq.add_argument("--anfreq", dest="anfreq", action="store_true", default=None,
                        help='Force "! AnFreq" on for the final stage.')
    anfreq.add_argument("--no-anfreq", dest="anfreq", action="store_false",
                        help='Force "! AnFreq" off for the final stage (no ORCA .hess).')
    parser.add_argument("-n", "--dry-run", action="store_true", help="Write everything, submit nothing")
    add_scheduler_argument(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_cli_logging()
    sched = get_scheduler(args.scheduler)
    try:
        xyz_files, input_dir = discover_xyz_files(args.path)
        validate(xyz_files, args.mode, args.pre)
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_root = (args.out_dir.expanduser() if args.out_dir else input_dir.parent).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        batch_log.init(out_root / batch_log.LOG_NAME, sched)
    plan = plan_stages(args.mode, args.pre, args.anfreq)
    log.info("Stages: %s; scheduler %s; %d structure(s) -> %s",
             " -> ".join(f"{m}({'AnFreq' if af else 'no AnFreq'})" for m, af in plan),
             sched.name, len(xyz_files), out_root)

    failed = []
    for xyz in xyz_files:
        try:
            prepare_chain(xyz, out_root, args.mode, sched, pre_modes=args.pre,
                          anfreq=args.anfreq, submit=not args.dry_run)
        except (PrepError, RuntimeError, ValueError) as exc:
            log.error("ERROR: %s", exc)
            failed.append(xyz)
    if failed:
        log.error("%d of %d structure(s) failed: %s", len(failed), len(xyz_files),
                  ", ".join(f.name for f in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
