"""Pack a CORVUS run done in node scratch and pick out what goes back to the shared filesystem.

The CORVUS job (``corvus-job.script``) runs CORVUS/FEFF in a node-local scratch dir. FEFF leaves ~130-180 files
(and writes GBs of temporaries on the way), which WekaFS handles badly at campaign scale: many small files, read back
days later one by one. So instead of copying the tree back, this stage, still on the node, writes into
``<scratch>/.return/`` only:

``<run_id>-corvus-<mode>.tar.zst``
    the whole scratch dir (inputs, ``Corvus3_cfavg_<mode>/`` tree, CORVUS logs), minus the regenerable FEFF
    temporaries the cleanup stage would delete anyway (``dmdw.out``, ``*.bin``, ``gg.dat``; ``gg.bin`` is kept, see
    cleanup.FEFF_KEEP). One file. The listing is
    checked against what was packed before anything is deleted. Unpack: ``tar -I zstd -xf <name>``.
``Corvus.cfavg_<mode>.out``
    the combined spectrum (postprocess and the dead-XANES check read it).
``Corvus3_cfavg_<mode>/<Corvus1Zn_*_FEFF>/{xanes,exafs}/xmu.dat`` (+ ``exafs/chi.dat``)
    a slim tree with just the FEFF tables, at the same paths as before, so postprocess (which publishes them as
    ``xmu-xanes-``, ``xmu-exafs-`` and ``chi-exafs-<id>.dat``), ``corvus_diagnosis`` and the auto-rerun work unchanged.
``feff-paths-<run_id>.tsv``
    the scattering paths of the EXAFS leg (``exafs/paths.dat``, 20 MB of text) reduced to which atoms make up each
    path: one row per path FEFF used, with its legs, degeneracy, effective length, the sigma^2 FEFF used (from the ``xmu.dat``
    header) and the scatterers as 0-based atom indices into ``corvus-begin-<run_id>.xyz``.
``<run_id>-corvus-<mode>-node.txt``
    what was packed and sent back, with the archive's size and sha256.

Exit status 0 means ``.return/`` is complete (whatever CORVUS itself did: a failed run is packed too, so its logs are
kept); non-zero means packing failed and the job script copies the scratch tree back the old way instead.

    python -m xas_pipeline.stages.corvus_pack <scratch dir> --run-id <id> [--mode xas]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xas_pipeline.corvus_diagnosis import CFAVG_OUTPUT_TEMPLATE
from xas_pipeline.logs import setup_cli_logging

log = logging.getLogger(__name__)

RETURN_DIR = ".return"
# Never packed: the return staging dir and the scratch heartbeat (node-env.sh).
SKIP_TOP = {RETURN_DIR, ".heartbeat"}
# Regenerable FEFF temporaries; the same deny-list as xas-cleanup's FEFF_SCRATCH_GLOBS, with the same exceptions.
PACK_EXCLUDE = ("dmdw.out", "*.bin", "gg.dat")
PACK_KEEP = ("gg.bin",)  # = cleanup.FEFF_KEEP (test_corvus_pack checks they agree)
# The FEFF tables the slim tree keeps (relative to the FEFF dir); postprocess publishes all three.
SLIM_TABLES = ("xanes/xmu.dat", "exafs/xmu.dat", "exafs/chi.dat")
# A path leg maps to a cluster atom when their positions (absorber at the origin) agree this well, in Angstrom.
# paths.dat prints 6 decimals; corvus-begin xyz carries ~14.
MATCH_TOL_A = 2e-3


# ── paths.dat -> which atoms make each path ─────────────────────────────────


@dataclass
class FeffPath:
    index: int
    nleg: int
    degeneracy: float
    reff: float
    legs: list[tuple[np.ndarray, int, str]]  # (xyz in the FEFF frame, ipot, label), last leg = absorber


_PATH_HEAD = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+index, nleg, degeneracy, r=\s*([\d.]+)")
_LEG = re.compile(r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+'([^']*)'")
_SIG2_ROW = re.compile(r"^#\s+(\d+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)")


def read_paths_dat(path: Path) -> list[FeffPath]:
    paths: list[FeffPath] = []
    current: FeffPath | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _PATH_HEAD.match(line)
            if m:
                current = FeffPath(int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), [])
                paths.append(current)
                continue
            m = _LEG.match(line)
            if m and current is not None and len(current.legs) < current.nleg:
                xyz = np.array([float(m.group(i)) for i in (1, 2, 3)])
                current.legs.append((xyz, int(m.group(4)), m.group(5).strip()))
    return paths


def read_sig2(xmu: Path) -> dict[int, tuple[float, float]]:
    """{path index: (sigma^2 used, curved-wave amplitude ratio %)} from an xmu.dat header."""
    out: dict[int, tuple[float, float]] = {}
    if not xmu.is_file():
        return out
    on = False
    with open(xmu, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "sig2 tot" in line:
                on = True
                continue
            m = _SIG2_ROW.match(line) if on else None
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def write_paths_table(feff_dir: Path, xyz_path: Path, dest: Path) -> int:
    """Write the path -> atoms table; returns the number of rows. Atoms that match nothing are written as -1."""
    paths = read_paths_dat(feff_dir / "exafs" / "paths.dat")
    sig2 = read_sig2(feff_dir / "exafs" / "xmu.dat")
    symbols, coords = read_xyz(xyz_path)
    absorber = _absorber_index(paths, symbols, coords)
    rel = coords - coords[absorber]
    unmatched = rows = 0
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(f"# FEFF exafs paths of {xyz_path.name}: which atoms each path visits.\n")
        fh.write(f"# atoms = 0-based indices into {xyz_path.name} (absorber {symbols[absorber]}{absorber}); "
                 "the absorber ends every path and is not repeated. sig2 = sigma^2 FEFF used (A^2, xmu.dat header);\n")
        fh.write("# amp_ratio = curved-wave amplitude ratio (%). Only the paths FEFF used (listed in the xmu.dat header) are\n"
                 "# written; with no such list (failed run) every path in paths.dat is. Atoms matching nothing are -1.\n")
        fh.write("path\tnleg\tdegeneracy\treff_A\tsig2_A2\tamp_ratio\tatoms\telements\n")
        for p in paths:
            if sig2 and p.index not in sig2:
                continue
            idx, els = [], []
            for xyz, _ipot, label in p.legs[:-1]:
                d = np.linalg.norm(rel - xyz, axis=1)
                j = int(np.argmin(d))
                if d[j] > MATCH_TOL_A:
                    j = -1
                    unmatched += 1
                idx.append(str(j))
                els.append(label)
            rows += 1
            s2, amp = sig2.get(p.index, (float("nan"), float("nan")))
            fh.write(f"{p.index}\t{p.nleg}\t{p.degeneracy:g}\t{p.reff:.4f}\t{s2:.5f}\t{amp:.2f}\t"
                     f"{','.join(idx)}\t{','.join(els)}\n")
    if unmatched:
        log.warning("%s: %d path legs matched no atom of %s within %g A", dest.name, unmatched, xyz_path.name, MATCH_TOL_A)
    return rows


def _absorber_index(paths: list[FeffPath], symbols: list[str], coords: np.ndarray) -> int:
    """The cluster atom FEFF put at the origin: the one whose shift makes the first path's scatterer line up."""
    candidates = [i for i, s in enumerate(symbols) if s.capitalize() == "Zn"] or list(range(len(symbols)))
    if not paths or not paths[0].legs:
        return candidates[0]
    first = paths[0].legs[0][0]
    best = min(candidates, key=lambda i: np.min(np.linalg.norm(coords - coords[i] - first, axis=1)))
    return best


# ── pack + return set ────────────────────────────────────────────────────────


def _excluded(rel: Path) -> bool:
    if rel.parts[0] in SKIP_TOP:
        return True
    return rel.name not in PACK_KEEP and any(rel.match(g) for g in PACK_EXCLUDE)


def pack(scratch: Path, archive: Path) -> list[str]:
    """tar.zst of the scratch dir (minus SKIP_TOP / PACK_EXCLUDE); returns the member list after verifying it."""
    members = sorted(
        str(p.relative_to(scratch))
        for p in scratch.rglob("*")
        if (p.is_file() or p.is_symlink()) and not _excluded(p.relative_to(scratch))
    )
    if not members:
        raise RuntimeError(f"nothing to pack in {scratch}")
    listfile = archive.with_name(archive.name + ".members")
    listfile.write_text("\0".join(members) + "\0", encoding="utf-8")
    try:
        subprocess.run(["tar", "-C", str(scratch), "-I", "zstd -3 -T0", "--null", "-T", str(listfile),
                        "-cf", str(archive)], check=True)
        listed = subprocess.run(["tar", "-I", "zstd", "-tf", str(archive)], check=True, capture_output=True,
                                text=True).stdout.splitlines()
    finally:
        listfile.unlink(missing_ok=True)
    if sorted(listed) != members:
        raise RuntimeError(f"archive listing differs from what was packed ({len(listed)} vs {len(members)} entries)")
    return members


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_return(scratch: Path, run_id: str, mode: str = "xas") -> Path:
    """Fill ``<scratch>/.return/`` (see the module docstring); returns it."""
    ret = scratch / RETURN_DIR
    if ret.exists():
        shutil.rmtree(ret)
    ret.mkdir()
    sent: list[str] = []

    archive = ret / f"{run_id}-corvus-{mode}.tar.zst"
    members = pack(scratch, archive)
    sent.append(archive.name)

    cfavg = scratch / CFAVG_OUTPUT_TEMPLATE.format(mode=mode)
    if cfavg.is_file():
        shutil.copy2(cfavg, ret / cfavg.name)
        sent.append(cfavg.name)

    mode_root = scratch / f"Corvus3_cfavg_{mode}"
    feff_dirs = sorted(mode_root.glob("Corvus1Zn_*_FEFF")) if mode_root.is_dir() else []
    for feff_dir in feff_dirs:
        for table in SLIM_TABLES:
            src = feff_dir / table
            if src.is_file():
                rel = src.relative_to(scratch)
                (ret / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, ret / rel)
                sent.append(str(rel))

    xyz = scratch / f"corvus-begin-{run_id}.xyz"
    if feff_dirs and (feff_dirs[0] / "exafs" / "paths.dat").is_file() and xyz.is_file():
        table = ret / f"feff-paths-{run_id}.tsv"
        try:
            n = write_paths_table(feff_dirs[0], xyz, table)
            sent.append(f"{table.name} ({n} paths)")
        except Exception as exc:  # the table is a convenience; never lose the run over it
            log.warning("could not write %s: %s", table.name, exc)
            table.unlink(missing_ok=True)

    note = ret / f"{run_id}-corvus-{mode}-node.txt"
    note.write_text(
        f"CORVUS ran in node scratch {scratch} on {socket.gethostname()} (job {os.environ.get('SLURM_JOB_ID', '?')}).\n"
        f"Packed {len(members)} files into {archive.name}: {archive.stat().st_size} bytes, sha256 {sha256(archive)}.\n"
        f"Left out (regenerable): {', '.join(PACK_EXCLUDE)} (kept: {', '.join(PACK_KEEP)}).\n"
        "Sent back loose:\n" + "".join(f"  {s}\n" for s in sent),
        encoding="utf-8",
    )
    return ret


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack a node-scratch CORVUS run and stage what goes back.")
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", default="xas")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_cli_logging()
    try:
        ret = build_return(args.scratch.resolve(), args.run_id, args.mode)
    except Exception as exc:
        log.error("packing %s failed: %s", args.scratch, exc)
        return 1
    log.info("staged %s", ret)
    return 0


if __name__ == "__main__":
    sys.exit(main())
