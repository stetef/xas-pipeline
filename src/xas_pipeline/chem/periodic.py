"""Element tables and lookups.

- :data:`ATOMIC_SYMBOLS` / :data:`ATOMIC_MASSES_AMU` (Z = 1..54) -- the ``.dym``
  and CORVUS inputs. Extend both together for heavier atoms.
- :data:`SPRING_MASSES_AMU` -- the masses the spring-model route mass-weights its
  Hessian with. A separate, rounder table; merging the two changes the floored
  spring Hessian, so they stay distinct.
- :data:`COVALENT_RADII` -- Alvarez 2008 (doi:10.1039/b801115j), for bond detection.
"""

from __future__ import annotations

ATOMIC_SYMBOLS = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
}

ATOMIC_MASSES_AMU = {
    1: 1.00794,
    2: 4.002602,
    3: 6.941,
    4: 9.012182,
    5: 10.811,
    6: 12.0107,
    7: 14.0067,
    8: 15.9994,
    9: 18.9984032,
    10: 20.1797,
    11: 22.98976928,
    12: 24.3050,
    13: 26.9815386,
    14: 28.0855,
    15: 30.973762,
    16: 32.065,
    17: 35.453,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955912,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938045,
    26: 55.845,
    27: 58.933195,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.64,
    33: 74.92160,
    34: 78.96,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90585,
    40: 91.224,
    41: 92.90638,
    42: 95.96,
    43: 98.0,
    44: 101.07,
    45: 102.90550,
    46: 106.42,
    47: 107.8682,
    48: 112.411,
    49: 114.818,
    50: 118.710,
    51: 121.760,
    52: 127.60,
    53: 126.90447,
    54: 131.293,
}

ATOMIC_NUM_TO_SYMBOL = {
    z: sym[0] + sym[1:].lower() for sym, z in ATOMIC_SYMBOLS.items()
}


def atomic_number_from_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("Empty atom token")
    if token.isdigit():
        return int(token)
    sym = token.upper()
    if sym not in ATOMIC_SYMBOLS:
        raise ValueError(
            f"Unknown element symbol '{token}'. Extend ATOMIC_SYMBOLS/ATOMIC_MASSES_AMU."
        )
    return ATOMIC_SYMBOLS[sym]


def atomic_mass_amu(z: int) -> float:
    if z not in ATOMIC_MASSES_AMU:
        raise ValueError(
            f"Unknown atomic mass for Z={z}. Extend ATOMIC_MASSES_AMU for this element."
        )
    return float(ATOMIC_MASSES_AMU[z])


def canonical_symbol_from_token(token: str) -> str:
    z = atomic_number_from_token(token)
    if z not in ATOMIC_NUM_TO_SYMBOL:
        raise ValueError(f"Missing canonical symbol for atomic number {z}")
    return ATOMIC_NUM_TO_SYMBOL[z]


COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.61, "Fe": 1.52, "Co": 1.50, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
    "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03,
    "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90,
    "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62,
    "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
    "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90,
    "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}
DEFAULT_COVALENT_RADIUS = 1.50

SPRING_MASSES_AMU = {
    "H": 1.008, "He": 4.003,
    "Li": 6.941, "Be": 9.012, "B": 10.811, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62,
    "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.96,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293,
    "Cs": 132.905, "Ba": 137.327, "La": 138.905,
    "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
    "Tb": 158.925, "Dy": 162.500, "Ho": 164.930,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Hf": 178.49, "Ta": 180.948, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217,
    "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.980,
}
