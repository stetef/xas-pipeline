# Node scratch and node-local software

On S3DF every shared filesystem (`/sdf/data`, `/sdf/home`, `/sdf/group`, `/sdf/scratch`) is WekaFS. It streams big
files well, but it is slow with many small files, files read back days later, and many jobs looking up the same
directory. Two campaign incidents (2026-11, `Riti-CSD-calcs/CAMPAIGN-LESSONS.md`) shaped this:

- FEFF writes ~130–180 files per run and reads/writes GBs of temporaries.
- `/sdf/group/.../openmpi/lib` became a metadata hot spot under ~800 jobs: `mpirun -n 16 /bin/true` took 28 s instead
  of 0.3 s, and every CORVUS run starts ~14 MPI programs.

So the Slurm job scripts compute in node-local `/lscratch` and send back a few files. Both pieces fall back to the old
in-place behaviour on any problem (no job id, no `/lscratch`, < 20 GB free, copy-in failure).

## CORVUS job

1. The run dir's top-level files are copied to `/lscratch/$USER/xas-pipeline/<job>-corvus-<run id>-xas/`.
2. CORVUS/FEFF run there. The live log `corvus-<run id>-xas.out` stays in the run dir (the watchdog reads its mtime).
3. On the node, `xas_pipeline.stages.corvus_pack` writes the return set:

   | sent back | what |
   |---|---|
   | `<run id>-corvus-xas.tar.zst` | the whole scratch run, minus `dmdw.out`, `*.bin` (except `gg.bin`, kept), `gg.dat` (regenerable; the same files `xas-cleanup` deletes). Unpack: `tar -I zstd -xf <file>` |
   | `Corvus.cfavg_xas.out` | combined spectrum |
   | `Corvus3_cfavg_xas/Corvus1Zn_*_FEFF/{xanes,exafs}/xmu.dat`, `exafs/chi.dat` | slim tree at the usual paths, so postprocess, diagnosis and the auto-rerun are unchanged |
   | `feff-paths-<run id>.tsv` | one row per EXAFS path FEFF used: legs, degeneracy, r_eff, σ² used, amplitude ratio, and the scatterers as 0-based indices into `corvus-begin-<run id>.xyz` (from `exafs/paths.dat`, 20 MB → ~90 kB). Postprocess copies it into `output-<run id>/` |
   | `<run id>-gg-check.tsv` | FMS's `gg.bin` (xanes, exafs) checked for the two FEFF artifacts (CR/LF header byte, all-zero section); the gate fails and auto-reruns on it (docs/recovery.md) |
   | `<run id>-corvus-xas-node.txt` | node, job, archive size + sha256, what came back |

   A failed CORVUS run is packed too (its logs are in the archive). If packing itself fails, the scratch tree is
   copied back unpacked, as before.
4. Every file lands under a temporary name and is renamed into place; an existing directory of the same name is moved
   aside as `<name>.rerun-node-<job>` (deleted by `xas-cleanup`). If any copy fails the job exits 74.

`xas-rerun-corvus` moves the archive, the paths table and the note aside with the other CORVUS outputs.

## ORCA job

ORCA already computed in node scratch; now:

- Its scratch is a per-job dir under `/lscratch/$USER/xas-pipeline/` (same heartbeat/cleanup as CORVUS).
- The log is written in scratch and copied to the run dir every `XAS_LOG_SYNC_S` (300 s) when it has grown, and once
  more at the end. The watchdog's hang test (log silent for 1 h *and* no CPU) still works.
- Every `XAS_ORCA_CHECKPOINT_S` (1800 s), and on TERM/INT (preemption, `scancel`, time limit), `<run id>.xyz`,
  `_trj.xyz`, `.opt` and `.gbw` are copied to `<run dir>/orca-checkpoint/`. After a normal finish the results are
  copied back as before and the checkpoint dir is removed. A killed run keeps it, so it can be restarted from its last
  geometry. Restarting is still a manual step (or a resubmission from the original input, as before).

## Restarts and dead nodes

| event | CORVUS | ORCA |
|---|---|---|
| preemption, `scancel`, time limit (TERM) | scratch dropped, run dir untouched; the watchdog's clean resubmit starts from the same state | checkpoint + log saved, scratch dropped |
| hang (watchdog kill) | same as above | same as above |
| node crash / SIGKILL (no trap runs) | run dir untouched; scratch left on that node | last checkpoint (≤ 30 min old) and log (≤ 5 min old) in the run dir |

On S3DF a cancel, preemption (`PreemptMode=CANCEL`) or time limit sends TERM to the batch shell only, not to its
children, and kills everything 30 s later (`KillWait`). So the scripts run CORVUS and ORCA in the background in their
own process groups: the batch shell's trap stops the whole group (MPI ranks included), then saves/cleans up within
the 30 s. The CORVUS wrapper forwards TERM to the CORVUS job script and waits for it.

`/lscratch` is not cleaned when a job ends (checked 2026-11-13). Each scratch dir has a `.heartbeat` its job touches
every 5 min; the next pipeline job on that node deletes any whose heartbeat is older than 2 h. A 7-day ORCA job keeps
its heartbeat fresh, so it is never touched.

## Node-local MPI + FEFF bundle (optional)

Set `PIPELINE_NODE_BUNDLE=<dir>/<tag>.tar` in `.env`. The first pipeline job on a node copies and checks it (sha256)
and extracts it to `/lscratch/$USER/xas-pipeline-sw/<tag>` (~1 min, once per node, under a lock). The job scripts then
take `mpirun`, the OpenMPI/HCOLL libraries, FEFF and `dym2feffinp` from there, and CORVUS reads the bundle's
`.Corvus/corvus.conf` (it runs with `HOME` set to the bundle's `home/`). Build one with

    PIPELINE_ENV=<pipeline>/.env bash src/xas_pipeline/data/sw/build-node-bundle.sh <out dir> [tag]

It copies the OpenMPI, HCOLL and FEFF installs named in `.env` and rewrites their ELF RUNPATHs to the node path, so
the loader never looks in `/sdf/group`. Rebuild under a new tag whenever one of those installs changes. ORCA itself
(15 GB) is not in the bundle yet.

## Knobs

| variable | default | |
|---|---|---|
| `XAS_NODE_SCRATCH` | `1` | `0` = run in place (old behaviour) |
| `XAS_SCRATCH_ROOT` | `/lscratch/$USER/xas-pipeline` | |
| `XAS_SCRATCH_MIN_GB` | `20` | free space needed, else run in place |
| `XAS_SCRATCH_STALE_MIN` | `120` | heartbeat age after which a leftover dir is deleted |
| `XAS_LOG_SYNC_S` | `300` | ORCA log copy interval |
| `XAS_ORCA_CHECKPOINT_S` | `1800` | ORCA checkpoint interval |
| `PIPELINE_NODE_BUNDLE` | unset | node-local MPI + FEFF bundle |
| `XAS_NODE_SW_ROOT` | `/lscratch/$USER/xas-pipeline-sw` | must match the bundle's build root |

The PBS templates share the CORVUS logic; without `/lscratch` (or a job id) they run in place as before.
