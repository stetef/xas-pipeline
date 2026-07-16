"""Element symbol/number/mass tables and pure lookup helpers.

Extracted verbatim from prepare-corvus.py. Tables cover Z=1..54 (through Xe),
which spans every element the pipeline encounters; extend both
:data:`ATOMIC_SYMBOLS` and :data:`ATOMIC_MASSES_AMU` together for heavier atoms.
"""

from __future__ import annotations

ANGSTROM_PER_BOHR = 0.52917724899

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
