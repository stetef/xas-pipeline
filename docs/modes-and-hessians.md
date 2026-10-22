# Modes, staged ORCA runs and spring Hessians

The mode registry is `src/xas_pipeline/modes.py`. Every mode table, the mode flags
of `xas-run-batch` and `xas-prepare-orca`, and the run-dir naming are derived from
it. The README has the summary table; this page has the details.

## What each mode does

| Mode | Geometry | AnFreq by default | Frozen atoms | Notes |
|---|---|---|---|---|
| `caopt-anfreq` | TightOPT, PBE0 | yes | CA atoms | the default |
| `quick` | OPT, B97-3c | no | none | cheap; no `.hess` |
| `quick-ca-fixed` | OPT, B97-3c | no | CA atoms | cheap; typical `--pre` stage |
| `hopt-anfreq` | TightOPT, hydrogens only | yes | none | |
| `carved-anfreq` | single point | yes | none | |
| `no-constraints` | TightOPT | yes | none | |
| `backbone` | TightOPT with point charges | yes | COORD=TRUE atoms on the backbone | needs `<stem>.pc` per XYZ |
| `xtb-free` | QM/xTB TightOPT | yes | none | QM region: COORD=TRUE atoms off the backbone |
| `xtb-constrained` | QM/xTB TightOPT | no | COORD=FALSE atoms and COORD=TRUE backbone atoms | QM region: all COORD=TRUE atoms |
| `carved-spring` | single point | no | none | spring Hessian; `--mode carved-spring` only |
| `hopt-spring` | TightOPT, hydrogens only | no | none | spring Hessian (`--interp`) |
| `asis-spring` | none (no ORCA) | n/a | n/a | spring Hessian (`--interp-raw`) |
| `caopt-spring` | none (no ORCA) | n/a | n/a | spring Hessian (`--caopt-spring`) |

"Backbone" means an atom tagged `ATOM=N`, `C`, `O` or `CA`, or a hydrogen whose
`BONDEDATOM=` is one of those. The tags are described in
[charge-multiplicity.md](charge-multiplicity.md). A constrained mode whose rule
matches no atom, or an xTB mode with no COORD=TRUE atom, is refused before
anything is written.

`caopt-spring` runs no ORCA. It is for geometries that are already CA-fixed
optimized (for example a `caopt-anfreq` result): the name only labels the run, and
the claim is not checked.

ORCA memory: `%MaxCore` is set per process from the atom count (≤50 atoms:
1000 MB, ≤90: 1800, ≤140: 2800, ≤200: 4200, above: 5600), and the job's memory
request is sized so that nprocs × MaxCore is about 70 % of it, with a 16 GB floor.

## Naming and older run dirs

A run id is `<structure id>-<mode>`. It names the run dir and every file in it
(`<run id>.in`, `<run id>.hess`, `xas-<run id>.dat`, ...), so runs of several modes
of one structure can sit side by side in one batch and in the download directory.

Run dirs from older batches use other suffixes. They are still recognised and
mapped to the mode they meant, but never written:

| Old suffix | Mode |
|---|---|
| `-ca-fixed` | `caopt-anfreq` |
| `-H-only`, `-h-only` | `hopt-anfreq` |
| `-single-point` | `carved-anfreq` |
| `-interp` | `carved-spring` |
| `-interp-hopt` | `hopt-spring` |
| `-opt-interp` | `caopt-spring` |
| `-interp-raw` | `asis-spring` |

A bare `<id>/` run dir (no suffix) is treated as having an ORCA Hessian. When a
new mode is added to such a batch, its run dir is created inside the old one
(`<id>/<id>-<mode>/`); every stage finds both.

## Staged runs and the AnFreq toggle (`xas-prepare-orca`)

Every ORCA template has an `[ANFREQ]` placeholder, filled from the mode's default.
`--anfreq` / `--no-anfreq` override it for the final stage:

```bash
# The CA-fixed PBE0 optimization without the analytic frequencies:
xas-prepare-orca xyz_dir --out-dir batch --scheduler slurm --no-anfreq
```

A `--no-anfreq` run writes no `.hess`, so a CORVUS job pointed at it fails in
prepare-corvus with a missing-Hessian error. The toggle buys the geometry, not the
Debye-Waller factors.

`--pre MODE` runs a cheaper ORCA stage first and starts the next stage from its
optimized geometry, submitted `afterok`:

```bash
# B97-3c CA-fixed pre-optimization, then the PBE0 CA-fixed optimization with AnFreq:
xas-prepare-orca xyz_dir --out-dir batch --scheduler slurm --pre quick-ca-fixed
```

```
batch/CPA_G_OH_OH/
  CPA_G_OH_OH-quick-ca-fixed/     stage 1, reads <run id>_clean.xyz
  CPA_G_OH_OH-caopt-anfreq/       stage 2, reads stage 1's <run id>.xyz
```

- `--pre` is repeatable; stages run in the order given, and the mode chosen by the
  usual flags runs last. `--pre` accepts only modes that have an ORCA template.
- Each stage has its own run dir, so every scan, check and rerun treats it as an
  ordinary ORCA run.
- The next stage's `*xyzfile` is an absolute path into the previous stage's dir,
  naming a file that does not exist yet; the `afterok` dependency guarantees it
  will.
- Pre-stages never run AnFreq: their geometry moves again, so their Hessian would
  describe a discarded structure. `--anfreq` / `--no-anfreq` apply to the final
  stage only.
- A mode may appear only once in a chain, and `--pre` cannot be combined with a
  mode that runs no ORCA. Both are rejected before anything is written.
- Each pre-stage writes `<run id>-next-stage.json` naming the next stage. When a
  stage fails, the scheduler drops the stage queued after it; `xas-rerun-orca`
  reads this file and re-chains the next stage onto the resubmitted job.

`xas-prepare-orca` submits the ORCA stages only (use `-n`/`--dry-run` to write
without submitting). It records its submissions in `batch-jobs.log`, so the ORCA
failure hook can resubmit these runs, but it submits no CORVUS, postprocess or
watchdog job; run `xas-submit-corvus` on the batch once the final stage has
finished. `xas-run-batch` always submits a single ORCA stage with the mode's
AnFreq default, because the CORVUS job it chains must depend on the final stage
and needs to know whether an ORCA Hessian will exist.

## Spring Hessians

In the spring modes the CORVUS job runs `xas_pipeline.stages.interp_hessian` before
prepare-corvus. It:

1. takes the run's geometry (`<run id>.xyz`: ORCA's optimized geometry, or the
   input geometry in the no-ORCA modes), the same file prepare-corvus uses, so the
   Hessian and the `.dym` describe the same atoms in the same order;
2. finds every occurrence of each ligand model in the cluster by subgraph
   isomorphism and interpolates the ligand's spring constants to the observed bond
   lengths, writing the merged constants to `spring.model`;
3. builds the 3N × 3N Hessian from those constants and the bond vectors and writes
   it as `<run id>.hess` in ORCA format, read downstream like a real ORCA Hessian.

`spring.model` is the only record of what the subgraph search matched; it is kept.
Run the stage by hand with `xas-interp-hessian <run_dir> [--run-id ID]`.

Why `hopt-spring` optimizes the hydrogens: heavy-atom positions in carved clusters
are crystallographic, but the protons are added by a protonation step, and FEFF's
muffin-tin construction is what notices. In a survey of 14 route-A failures in
`clustering-validation/4cys/first-set`, every one aborted on a heavy-atom···H
overlap (12 C···H, 2 S···H, never heavy···heavy) with `MOVRLP-1` or
`tell authors to INCREASE NOVP`. Relaxing the protons moves exactly the atoms
responsible.

`asis-spring` skips ORCA entirely. On a carved cluster it gives the same spectrum
as the single-point `carved-spring` mode (a single point moves no atoms) at none of
the cost. The trade-off: ORCA is otherwise the only stage that looks at the
geometry before FEFF does, so a malformed cluster fails later and with a worse
error.

### Ligand models

Ligand files are taken, in order of precedence, from `--ligand FILE` /
`--ligand-dir DIR` on `xas-interp-hessian`; the run dir's own `interp-ligands/`;
the packaged `src/xas_pipeline/data/interp-ligands/` (`ZnCys`, `ZnHis`,
`ZnHis_2`). The two histidine files are not alternatives: the two ring nitrogens
coordinate differently, so both are searched.

`xas-run-batch --ligand-dir DIR` uses every `.interp` file in `DIR` instead of the
packaged set. A `{stem}` in `DIR` is replaced by each structure's XYZ stem, for one
database per structure (`--ligand-dir /db/{stem}`). Every structure's directory is
checked before anything is written, and the files are copied into the run dir's
`interp-ligands/`, so reruns and `xas-submit-corvus` use the same database.

A ligand with no model contributes no springs; those atoms are then coupled only
by the inter-ligand floor below. When two ligand files match the same pair, the
later file's value is used and a warning is logged.

### What to watch

- **Imaginary modes.** The stage diagonalizes the Hessian and reports imaginary
  modes beyond the six translations and rotations. With the inter-ligand floor off,
  some are expected in the spring Hessian: the packaged models contain negative spring constants (8 of 36
  pairs in one `ZnCys` reference model, 38 of 120 in `ZnHis`), and each reference
  model is slightly unstable at its own reference geometry (1 imaginary mode for
  `ZnCys`, 3–4 for the `ZnHis` pair, worst around −100 cm⁻¹), so such modes are
  largely inherited from the models, not a sign that the interpolation misfired.
  Imaginary modes before the eigenvalue floor are reported as a `NOTE:`; a
  `WARNING:` means some survived the floor and the Debye-Waller factors are
  suspect.
- **Inter-ligand coupling** (`--add-inter-ligand SCALE`, default 1.0). The ligand
  models describe nothing between ligands, which leaves the cluster
  under-constrained: on the test cluster, 9 floppy zero modes on top of the 6
  translations/rotations, and 4 imaginary modes. The floor sets every pair's
  constant to at least SCALE × a hydrogen bond (0.005 Ha/Bohr²). With it on, the
  null space is exactly 6 and the raw Hessian has no imaginary modes. The cost: it
  couples distant atoms that are not physically correlated, which slightly lowers
  the Debye-Waller σ² of long FEFF paths. Because the floor is a minimum, it also
  lifts the models' negative constants to the floor value. `0` disables it.
- **Eigenvalue floor** (`--min-freq-scale SCALE`, default 1.0). Raises every
  mass-weighted eigenvalue below SCALE × max(eigenvalue) / 1000² to that floor,
  pins the six translation/rotation modes to zero and rebuilds the Hessian. `0`
  writes the raw Hessian. It is only safe when the null space is six-dimensional:
  with `--add-inter-ligand 0` the extra floppy modes are degenerate with the
  translations/rotations, the six that get pinned are an arbitrary basis of that
  subspace, and the acoustic sum rule breaks (~6e-6 Ha/Bohr² on the test cluster).
  Turn both off together, not one. `--no-freq-check` skips the diagonalization and
  therefore the floor as well.
- **Extrapolation clamp** (`--extrap-limits LOW HIGH`, default −2 2). Each pair's
  constant is interpolated in α = (d − d₁)/(d₂ − d₁), where d₁ < d₂ are the two
  reference bond lengths (the reader orders them, so α = 0 is the shorter one, not
  necessarily the natural-charge model). α is clamped to the limits: two
  references whose bond lengths barely differ otherwise send α to huge values,
  which in log space runs away exponentially (one pair on the test cluster came
  out at 0.90 Ha/Bohr², stiffer than the Zn–S bond beside it). `--no-extrapolate`
  clamps to [0, 1].
- **Interpolation form.** Log-linear in k when both constants are positive, k falls
  as d grows, the pair is bonded in the cluster and neither atom is H; linear
  otherwise. `spring.model` lists every atom pair; pairs no ligand covers get
  k = 0 there and are raised by the inter-ligand floor when the Hessian is built.
- **Hydrogen-swap degeneracies.** A ligand with interchangeable hydrogens matches
  once per H permutation (48 matches for `ZnCys` on a Cys₄ cluster). These describe
  the same bonds, are averaged and reported as one count. A disagreement involving
  a heavy atom is a real ambiguity in the ligand definition and is warned about.

Other options: `--bond-factor` / `--bond-cutoff` (bonded when
d ≤ 1.2 × (rᵢ + rⱼ) by default), `--zero-negative` (drop negative interpolated
constants), `--skip-modes` (how many lowest-|ν| modes the imaginary-mode report
ignores, default 6).

## Building `.interp` ligand models

Database building is separate from the batch: nothing in the job chain runs it. A
ligand's `.interp` comes from two ORCA Hessians of the same atoms in the same
order, one per charge state, natural charge first:

```bash
xas-build-interp natural.hess modified.hess -o /db/<structure>/ZnCys.interp
```

- The geometry is read from each `.hess` file's `$atoms` block. The natural-charge
  geometry becomes the `.interp` reference; at inference the ligand's bond graph is
  built from it.
- The two fitted spring models are kept as `<stem>.natural.springs` and
  `<stem>.modified.springs` next to the output (or in `--springs-dir`).
- Every atom pair is kept by default, as in the packaged models; `--cutoff` and
  `--bonded-only` narrow it.
- A bond present in only one charge state is warned about; `--strict` makes it an
  error. `--no-check` skips the Hessian reconstruction report.
- From Python: `xas_pipeline.chem.spring_model.build_interp_from_hess`.

Point `xas-run-batch --ligand-dir` at the directory that holds the result.
