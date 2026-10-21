"""Physical constants and unit conversions.

Two Bohr radii coexist on purpose, each pinned by the outputs it produces:

- :data:`BOHR_TO_ANG` (CODATA 2018) -- the spring-model code and its ``.hess``
  reading/writing;
- :data:`BOHR_TO_ANG_DYM` -- the ``.dym`` dynamical-matrix coordinates handed to
  FEFF's DMDW module.

Using one for the other moves written coordinates in the 8th decimal, so callers
name the one they mean.
"""

from __future__ import annotations

BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
BOHR_TO_ANG_DYM = 0.52917724899

HARTREE_TO_EV = 27.211396132
HARTREE_TO_J = 4.3597447222071e-18
BOHR_TO_M = 5.29177210903e-11
AMU_TO_KG = 1.66053906660e-27
C_CM_PER_S = 2.99792458e10

HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / BOHR_TO_ANG**2
HARTREE_BOHR2_TO_NM = HARTREE_TO_J / BOHR_TO_M**2
