#!/usr/bin/env python3
"""Apply a :class:`~xas_pipeline.remedy.Remedy` to an existing ORCA input file.

Pure text transform (no I/O), so it is fully unit-testable. Given the already
rendered ``<id>.in`` (charge/mult/geometry already substituted) and a Remedy, it
returns the edited input for the resubmission:

* extra simple keywords -> ``! <kw>`` lines inserted after the first ``!`` line
  (``MOREAD`` and ``SCFStabilityAnalysis`` are added here when requested);
* ``%moinp "<gbw>"`` + a ``%scf ... end`` block inserted just before the
  ``*xyzfile`` geometry line, behind a ``# --- auto-rerun remedy: <label> ---``
  marker;
* ``%maxcore N`` multiplied by ``maxcore_mult`` (rounded);
* on ``opt_restart``, the geometry filename on the ``*xyzfile`` line is swapped
  for the last-completed geometry.
"""

from __future__ import annotations

from xas_pipeline.remedy import Remedy

REMEDY_MARKER = "# --- auto-rerun remedy:"


def _swap_geometry_filename(xyzfile_line: str, new_filename: str) -> str:
    """Replace the geometry file on a ``*xyzfile <charge> <mult> <path>`` line.

    Only the basename is swapped; the directory (if any) is preserved.
    """
    parts = xyzfile_line.split()
    if len(parts) < 2:
        return xyzfile_line
    old_path = parts[-1]
    if "/" in old_path:
        directory = old_path.rsplit("/", 1)[0]
        parts[-1] = f"{directory}/{new_filename}"
    else:
        parts[-1] = new_filename
    return " ".join(parts)


def apply_remedy(
    in_text: str,
    remedy: Remedy,
    *,
    gbw_name: str | None = None,
    last_geometry_name: str | None = None,
) -> str:
    """Return ``in_text`` with ``remedy`` applied. Idempotent inputs are not assumed."""
    extra_keywords = list(remedy.keywords)
    if remedy.use_moread and gbw_name:
        extra_keywords.append("MOREAD")
    if remedy.stability_analysis:
        extra_keywords.append("SCFStabilityAnalysis")

    # Block inserted just before *xyzfile: marker + %moinp + %scf...end.
    pre_geometry: list[str] = [f"{REMEDY_MARKER} {remedy.label} ---"]
    if remedy.use_moread and gbw_name:
        pre_geometry.append(f'%moinp "{gbw_name}"')
    if remedy.scf_lines:
        pre_geometry.append("%scf")
        pre_geometry.extend(f"  {line}" for line in remedy.scf_lines)
        pre_geometry.append("end")

    out: list[str] = []
    keywords_inserted = False
    for line in in_text.splitlines():
        stripped = line.strip()

        if remedy.maxcore_mult != 1.0 and stripped.lower().startswith("%maxcore"):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].isdigit():
                new_val = int(round(int(parts[1]) * remedy.maxcore_mult))
                line = f"%maxcore {new_val}"

        if stripped.startswith("*xyzfile"):
            # Emit the remedy block immediately before the geometry spec.
            if len(pre_geometry) > 1:  # more than just the marker
                out.extend(pre_geometry)
                out.append("")
            if remedy.opt_restart and last_geometry_name:
                line = _swap_geometry_filename(line, last_geometry_name)

        out.append(line)

        # Insert extra keywords right after the first "!" keyword line.
        if not keywords_inserted and stripped.startswith("!") and extra_keywords:
            out.extend(f"! {kw}" for kw in extra_keywords)
            keywords_inserted = True

    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    return text
