"""Placeholder-fill engine for pipeline job/input templates.

Every template (ORCA input, ORCA/Corvus job scripts, corvus-wrapper and
postprocess scripts) uses ``[UPPER_SNAKE]`` placeholder tokens:

- :func:`fill` swaps each ``[TOKEN]`` for its value in a string.
- :func:`render` reads a template file, fills it, and writes the result,
  optionally making it executable / newline-terminated.

The engine is case-sensitive and does not enforce the naming convention.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

# A placeholder left in rendered text: ``[UPPER_SNAKE]``.
UNFILLED_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")


def fill(text: str, mapping: Mapping[str, object]) -> str:
    """Replace each ``[KEY]`` placeholder in *text* with ``str(mapping[KEY])``.

    Keys are token names *without* the surrounding brackets. Replacements are
    applied in iteration order of *mapping*; callers that depend on ordering
    (e.g. one token's value could contain another token) should pass an
    ordered mapping.
    """
    for token, value in mapping.items():
        text = text.replace(f"[{token}]", str(value))
    return text


def render(
    template_path: os.PathLike | str,
    dest_path: os.PathLike | str,
    mapping: Mapping[str, object],
    *,
    executable: bool = False,
    ensure_trailing_newline: bool = False,
) -> Path:
    """Fill the template at *template_path* and write it to *dest_path*.

    Returns the destination :class:`~pathlib.Path`. With
    ``ensure_trailing_newline`` a missing final newline is appended (matches the
    generated job/wrapper scripts); with ``executable`` the result is chmod
    ``0o755``. Raises ValueError, writing nothing, if a placeholder is left unfilled.
    """
    content = fill(Path(template_path).read_text(encoding="utf-8"), mapping)
    unfilled = sorted(set(UNFILLED_RE.findall(content)))
    if unfilled:
        raise ValueError(f"{Path(template_path).name}: unfilled placeholder(s) {', '.join(unfilled)}")
    if ensure_trailing_newline and not content.endswith("\n"):
        content += "\n"
    dest = Path(dest_path)
    dest.write_text(content, encoding="utf-8")
    if executable:
        dest.chmod(0o755)
    return dest
