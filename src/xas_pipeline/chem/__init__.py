"""Pure chemistry parsers/transforms extracted from the flat scripts.

Modules here are side-effect-free (they read a passed-in path or transform
in-memory data and return values); the I/O shells that write files stay in the
stage code. Split out so the parsing/numeric logic is unit-testable directly.

- :mod:`periodic` -- element symbol/number/mass tables and lookups
- :mod:`xyz`      -- XYZ geometry parsing (ORCA + Corvus conventions)
- :mod:`hessian`  -- ORCA ``.hess`` (`$HESSIAN` block) parsing
- :mod:`feff`     -- FEFF/Corvus output tables and chi(k)->chi(R) FFT

Two modules are *vendored* from DW_Interpolation rather than written here, and
back the ``--interp`` mode (a Hessian from ligand spring models instead of an
ORCA analytic-frequency run). Their numerics are kept as upstream wrote them;
re-vendor rather than editing them in place:

- :mod:`springs`        -- interpolate per-ligand spring constants onto a cluster
- :mod:`spring_hessian` -- build an ORCA-format ``.hess`` from a spring model
"""

from __future__ import annotations
