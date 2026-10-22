# Installing CORVUS and building FEFF10

The CORVUS job runs `run-corvus` from the CORVUS package, which drives FEFF10 and
its DMDW module. prepare-corvus also uses FEFF10's `dym2feffinp` to centre the
dynamical matrix on the absorbing Zn.

## CORVUS

CORVUS ([times-software/Corvus](https://github.com/times-software/Corvus)) is not on
PyPI and is deliberately not a dependency or a `uv.lock` entry. Install it from git
into the pipeline's `.venv`:

```bash
uv pip install "corvus @ git+https://github.com/times-software/Corvus.git"
```

To pin a branch, tag or commit, append `@<ref>`; add `--reinstall` when switching
refs, otherwise uv considers the requirement satisfied and skips it:

```bash
uv pip install --reinstall "corvus @ git+https://github.com/times-software/Corvus.git@xas_input"
```

The pipeline is developed against CORVUS's `xas_input` branch: one
`cfavg_target{xas}` run reads `xanes.in` and `exafs.in` and writes the combined
spectrum `Corvus.cfavg_xas.out`.

Because CORVUS is not in `uv.lock`, a plain `uv sync` removes it. Use
`uv sync --inexact`, or repeat the install after a full sync. The installed ref is
recorded in `.venv/lib/python*/site-packages/corvus-*.dist-info/direct_url.json`.

The CORVUS job finds `run-corvus` through `CORVUS_RUNNER` in `.env`, else in the
pipeline `.venv`, `~/.venv`, the `corvusenv` conda environment or `PATH`.

## FEFF10

FEFF10 is built from source on the `inters` branch; there is no installer. Build
the MPI binaries into `bin/MPI/` and point CORVUS at them. The build needs your
site's OpenMPI on `PATH`/`LD_LIBRARY_PATH` and, for the `inters` branch, the
`-mcmodel=large` compiler flag: that branch has COMMON blocks totalling more than
2 GB, so without it the link fails with `relocation truncated to fit:
R_X86_64_PC32 against ... COMMON` (`-mcmodel=medium` is not enough).

```bash
git clone https://github.com/times-software/feff10 feff10-git-installed
cd feff10-git-installed

# 1. Reuse a working compiler config. The inters branch ships only
#    Compiler.mk.default (mpif90); a known-good one uses MPIF90 = mpifort.
#    Copy the Compiler.mk from a working FEFF10 (10.0.0) build.
cp <your-feff10-10.0.0>/src/Compiler.mk src/Compiler.mk

# 2. Put the site's OpenMPI and hcoll libraries on PATH/LD_LIBRARY_PATH. These
#    are the installs the pipeline uses at run time (PIPELINE_OPENMPI_ROOT and
#    PIPELINE_HPCX_ROOT in .env): `set -a; source .env; set +a` to reuse them.
export PATH="$PIPELINE_OPENMPI_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$PIPELINE_OPENMPI_ROOT/lib:$PIPELINE_HPCX_ROOT/hcoll/lib:$LD_LIBRARY_PATH"

# 3. Add -mcmodel=large to MPIFLAGS in src/Compiler.mk (by hand).

# 4. Build. `clean` is required after any flag change: make does not track flags.
#    DEP/*.mk are committed, so no `make deps` is needed.
make -C src clean && make -C src mpi
```

`make mpi` compiles all parallel binaries, including `dym2feffinp`, into
`bin/MPI/`. Then:

- set both the `feff` and `dmdw` keys in `~/.Corvus/corvus.conf` to that `bin/MPI/`
  directory;
- set `DYM2FEFFINP_BIN=<checkout>/bin/MPI/dym2feffinp` in the pipeline's `.env`.

prepare-corvus runs `dym2feffinp` on short scratch file names: it keeps file names
in a fixed-length Fortran buffer and, past about 50 characters, silently writes to
a truncated name while still exiting 0. Without `dym2feffinp`, prepare-corvus falls
back to CORVUS's Python DYM writer (`corvus.dmdw.writeDym`); if
`DYM2FEFFINP_BIN` is set but not executable, it logs a warning and falls back too.
