"""Pure chemistry parsers/transforms extracted from the flat scripts.

Modules here are side-effect-free (they read a passed-in path or transform
in-memory data and return values); the I/O shells that write files stay in the
stage code. Split out so the parsing/numeric logic is unit-testable directly.

- :mod:`periodic` -- element symbol/number/mass tables and lookups
- :mod:`xyz`      -- XYZ geometry parsing (ORCA + Corvus conventions)
- :mod:`hessian`  -- ORCA ``.hess`` (`$HESSIAN` block) parsing
- :mod:`feff`     -- FEFF/Corvus output tables and chi(k)->chi(R) FFT
"""

from __future__ import annotations
