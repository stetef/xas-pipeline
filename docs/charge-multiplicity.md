# Input XYZ files: charge, multiplicity and atom tags

`xas-run-batch` and `xas-prepare-orca` take one `.xyz` file or a directory of them
(`*.xyz`, sorted). The file stem is the structure id and the first part of every
run id.

## Charge and multiplicity (line 2)

The second line of the XYZ file must contain a charge token and a multiplicity
token:

- charge: `CHARGE_ROUNDED=<int>`, `ROUNDED_CHARGE=<int>` or `CHARGE=<int>`
- multiplicity: `MULTIPLICITY=<int>` or `MULT=<int>`

```
id=2j6a_ZN_cluster1 CHARGE_ROUNDED=-2 MULTIPLICITY=1
```

Rules, as the parser (`xas_pipeline.chem.xyz.extract_charge_multiplicity`) applies
them:

- Tokens are upper case and case-sensitive: `charge=-2` is not recognised.
- When several charge tokens are present, the first one on the line wins, whatever
  its name.
- Only the integer part of the value is read: `CHARGE=-1.6` is read as `-1`.
- A file without both tokens is refused before anything is written or submitted,
  with the offending line in the error message.

The values go into the ORCA input's `*xyzfile <charge> <multiplicity> <file>` line
unchanged. A charge/multiplicity that is impossible for the electron count fails
ORCA within seconds; the failure hook classifies it as `charge_mult` and escalates
it to a human (see [recovery.md](recovery.md)).

## Atom tags

Anything after `#` on an atom line is a tag list. prepare-orca writes the bare
coordinates to `<run id>_clean.xyz` (what ORCA reads) and the tags, one line per
atom, to `<run id>_comments.txt`. Three tags are used; others (`RES=`, `CHAIN=`,
`RESSEQ=`, ...) are carried along but ignored:

| Tag | Meaning | Used by |
|---|---|---|
| `ATOM=` | PDB atom name (`CA`, `SG`, `NE2`, ...) | `CA` atoms are frozen in the CA-fixed modes; `N`, `C`, `O`, `CA` define the backbone |
| `BONDEDATOM=` | for a hydrogen, the heavy atom it sits on | a hydrogen on a backbone atom counts as backbone |
| `COORD=TRUE` / `COORD=FALSE` | whether the residue coordinates the metal | `backbone`, `xtb-free`, `xtb-constrained` |

Which atoms each mode freezes and which it puts in the QM region is listed in
[modes-and-hessians.md](modes-and-hessians.md). A mode whose rule matches no atom
is refused for that structure.

## Other inputs

- `backbone` mode needs `<stem>.pc` (ORCA point-charge file) next to each XYZ.
- The absorber is the first Zn atom in the file; prepare-corvus refuses a geometry
  with no Zn.
