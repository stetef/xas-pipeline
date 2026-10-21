"""Auto-rerun remedies for failed ORCA runs: which one, and its edit to the input.

The ladder is short and physics-neutral: more memory for an out-of-memory
failure, and an optimization restarted from its last geometry for one that did
not converge -- each reusing the previous orbitals when a ``.gbw`` exists. SCF
non-convergence is escalated to a human rather than retried with solver aids
(tetef01, 2026-09-21: auto-rerun recovers from resource problems; it does not
re-tune the calculation, so the input stays the one a human wrote). Silent hangs
are the watchdog's business, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

from xas_pipeline.diagnosis import Evidence, FailureKind

#: Automatic reruns per structure (the counter persists in the run dir).
MAX_ATTEMPTS = 2

#: Marks the block an applied remedy inserts into the ORCA input.
REMEDY_MARKER = "# --- auto-rerun remedy:"

# %MaxCore (and the job's memory request) multiplier per OOM attempt.
_OOM_MULTIPLIERS = {1: 1.6, 2: 2.5}


@dataclass
class Remedy:
    label: str
    #: Start from the previous orbitals (``! MOREAD`` + ``%moinp``).
    use_moread: bool = False
    #: Restart the optimization from the last geometry, not the original one.
    opt_restart: bool = False
    #: Multiply %MaxCore (and the job's memory request) by this.
    maxcore_mult: float = 1.0


def select_remedy(kind: FailureKind, evidence: Evidence, attempt: int,
                  max_attempts: int = MAX_ATTEMPTS) -> Remedy | None:
    """The remedy for the 1-based *attempt*, or None: escalate instead.

    *max_attempts* can only shorten the ladder; each kind has its own rungs.
    """
    if attempt < 1 or attempt > min(max_attempts, MAX_ATTEMPTS):
        return None
    restart = evidence.n_opt_cycles >= 2
    if kind is FailureKind.OOM and attempt in _OOM_MULTIPLIERS:
        mult = _OOM_MULTIPLIERS[attempt]
        return Remedy(f"oom-maxcore-x{mult:g}", use_moread=evidence.gbw_present,
                      opt_restart=restart, maxcore_mult=mult)
    if kind is FailureKind.OPT_NONCONVERGENCE and attempt == 1:
        return Remedy("opt-restart", use_moread=evidence.gbw_present, opt_restart=True)
    return None


def _swap_geometry(xyzfile_line: str, new_geometry: str) -> str:
    """Replace the file on a ``*xyzfile <charge> <mult> <path>`` line.

    A path with a ``/`` is used as is; a bare name keeps the line's directory.
    (A ``--pre`` stage's line points into the previous stage's dir, so callers
    pass full paths.)
    """
    parts = xyzfile_line.split()
    if len(parts) < 2:
        return xyzfile_line
    old = parts[-1]
    if "/" not in new_geometry and "/" in old:
        new_geometry = f"{old.rsplit('/', 1)[0]}/{new_geometry}"
    parts[-1] = new_geometry
    return " ".join(parts)


def apply_remedy(in_text: str, remedy: Remedy, *, gbw_name: str | None = None,
                 last_geometry_name: str | None = None) -> str:
    """The ORCA input *in_text* with *remedy* applied.

    MOREAD (when *gbw_name* is given): ``! MOREAD`` after the first ``!`` line
    and ``%moinp "<gbw>"`` before the geometry, behind a marker comment.
    ``%maxcore`` is scaled; an opt restart points ``*xyzfile`` at
    *last_geometry_name*.
    """
    moread = remedy.use_moread and gbw_name
    out: list[str] = []
    keywords_added = False
    for line in in_text.splitlines():
        stripped = line.strip()
        if remedy.maxcore_mult != 1.0 and stripped.lower().startswith("%maxcore"):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].isdigit():
                line = f"%maxcore {int(round(int(parts[1]) * remedy.maxcore_mult))}"
        if stripped.startswith("*xyzfile"):
            if moread:
                out += [f"{REMEDY_MARKER} {remedy.label} ---", f'%moinp "{gbw_name}"', ""]
            if remedy.opt_restart and last_geometry_name:
                line = _swap_geometry(line, last_geometry_name)
        out.append(line)
        if moread and not keywords_added and stripped.startswith("!"):
            out.append("! MOREAD")
            keywords_added = True
    return "\n".join(out) + "\n"
