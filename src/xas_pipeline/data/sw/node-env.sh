# node-env.sh -- sourced by the ORCA and CORVUS job scripts. Keeps a job's chatty I/O off the shared filesystem.
#
# Why: every shared path on S3DF (/sdf/group, /sdf/home, /sdf/data, /sdf/scratch) is WekaFS. It streams big files
# well, but metadata (lookups/opens/stats) on one busy directory and first reads of idle files get slow under hundreds
# of jobs. Two campaign incidents set the design (Riti-CSD-calcs/CAMPAIGN-LESSONS.md):
#   * FEFF writes ~170 files and GBs of temporaries per run: run it in node-local /lscratch and send back one packed
#     archive plus the few files the pipeline reads.
#   * /sdf/group/.../openmpi/lib became a metadata hot spot under ~800 jobs (mpirun 28 s instead of 0.3 s): run
#     MPI + FEFF from a bundle extracted once per node into /lscratch, with ELF RUNPATHs rewritten to that path.
#
# Two independent pieces, both optional and both falling back to the old behaviour on any problem:
#
#   xas_node_bundle_env     stage $PIPELINE_NODE_BUNDLE (built by build-node-bundle.sh) on this node, export
#                           PATH/LD_LIBRARY_PATH/OPAL_PREFIX/... and XAS_CORVUS_HOME (the HOME CORVUS must run with
#                           so it reads the bundle's .Corvus/corvus.conf). Unset bundle -> no-op, returns 1.
#   xas_scratch_make TAG    make a fresh per-job scratch dir under $XAS_SCRATCH_ROOT, set XAS_SCRATCH_DIR to it and
#                           start its heartbeat. Call it directly, never inside $(...) (a subshell would lose the
#                           variables and the heartbeat). Returns 1 when node scratch is off, missing, or too small.
#   xas_scratch_drop        stop the heartbeat and delete the scratch dir.
#
# Knobs (environment, e.g. in the pipeline .env):
#   XAS_NODE_SCRATCH=0          run in place on the shared filesystem, exactly as before
#   XAS_SCRATCH_ROOT            default /lscratch/$USER/xas-pipeline
#   XAS_SCRATCH_MIN_GB          default 20 (free space required, else run in place)
#   PIPELINE_NODE_BUNDLE        path to <tag>.tar (next to <tag>.tar.sha256); unset = use the group install
#   XAS_NODE_SW_ROOT            default /lscratch/$USER/xas-pipeline-sw (must match the bundle's build root)
#
# Leftovers: /lscratch is NOT cleaned when a job ends (checked 2026-11-13: per-job dirs outlive the job). A job that
# dies without its EXIT/TERM trap running (node crash, SIGKILL) leaves its scratch dir behind. Every scratch dir has a
# .heartbeat touched every XAS_HEARTBEAT_S (300 s) while its job lives; xas_scratch_make deletes any sibling whose
# heartbeat is older than XAS_SCRATCH_STALE_MIN (120 min). A 7-day ORCA job keeps touching its heartbeat, so age of
# the directory itself never matters.

_xas_log() { echo "[node-env $(date -u +%FT%TZ)] $*" >&2; }

xas_node_bundle_env() {
    local tar=${PIPELINE_NODE_BUNDLE:-}
    [[ -n "$tar" ]] || return 1
    if [[ ! -f "$tar" || ! -f "$tar.sha256" ]]; then
        _xas_log "bundle $tar (or its .sha256) missing: using the group install"
        return 1
    fi
    local tag parent root t0=$SECONDS
    tag=$(basename "$tar" .tar)
    parent=${XAS_NODE_SW_ROOT:-/lscratch/$USER/xas-pipeline-sw}
    root=$parent/$tag
    if [[ ! -f "$root/.ready" ]]; then
        mkdir -p "$parent" || { _xas_log "cannot create $parent: using the group install"; return 1; }
        (
            flock -w 900 9 || exit 1
            [[ -f "$root/.ready" ]] && exit 0
            tmp=$(mktemp -d "$parent/.stage.XXXXXX") || exit 1
            trap 'rm -rf "$tmp"' EXIT
            cp "$tar" "$tmp/b.tar" || exit 1
            want=$(cut -d' ' -f1 "$tar.sha256") || exit 1
            [[ "$(sha256sum "$tmp/b.tar" | cut -d' ' -f1)" == "$want" ]] || { echo "bundle checksum mismatch" >&2; exit 1; }
            tar -xf "$tmp/b.tar" -C "$tmp" || exit 1
            rm -rf "$root"
            mv "$tmp/$tag" "$root" || exit 1
            touch "$root/.ready"
        ) 9> "$parent/.lock-$tag" || { _xas_log "staging $tag failed: using the group install"; return 1; }
    fi
    # The RUNPATHs inside were rewritten for one fixed directory at build time; refuse a bundle staged elsewhere.
    local built_for
    built_for=$(sed -n 's/^NODE_ROOT=//p' "$root/bundle.env" 2>/dev/null)
    if [[ "$built_for" != "$root" ]]; then
        _xas_log "bundle $tag was built for '$built_for', not $root: using the group install"
        return 1
    fi
    local lp="" p
    IFS=: read -ra _xas_parts <<< "${LD_LIBRARY_PATH:-}"
    for p in "${_xas_parts[@]}"; do
        [[ -z "$p" || "$p" == "${PIPELINE_OPENMPI_ROOT:-/nonexistent}"* || "$p" == "${PIPELINE_HPCX_ROOT:-/nonexistent}"* ]] && continue
        lp+=":$p"
    done
    export LD_LIBRARY_PATH="$root/openmpi/lib:$root/hcoll-lib$lp"
    export PATH="$root/openmpi/bin:$root/feff-bin:$PATH"
    export OPAL_PREFIX=$root/openmpi PRTE_PREFIX=$root/openmpi PMIX_PREFIX=$root/openmpi
    export HWLOC_PLUGINS_PATH=$root/openmpi/lib/hwloc
    export DYM2FEFFINP_BIN=$root/feff-bin/dym2feffinp
    export XAS_CORVUS_HOME=$root/home
    export XAS_NODE_BUNDLE_ROOT=$root
    _xas_log "using node-local $root ($((SECONDS - t0)) s; mpirun $(command -v mpirun))"
}

XAS_SCRATCH_DIR=
_XAS_HB_PID=

xas_scratch_make() {
    local tag=$1 root=${XAS_SCRATCH_ROOT:-/lscratch/$USER/xas-pipeline}
    local job=${SLURM_JOB_ID:-${PBS_JOBID:-}}
    [[ "${XAS_NODE_SCRATCH:-1}" != "0" ]] || { _xas_log "XAS_NODE_SCRATCH=0: running in place"; return 1; }
    [[ -n "$job" ]] || { _xas_log "no batch job id: running in place"; return 1; }
    mkdir -p "$root" 2>/dev/null || { _xas_log "cannot create $root: running in place"; return 1; }
    local avail need=$(( ${XAS_SCRATCH_MIN_GB:-20} * 1024 * 1024 ))
    avail=$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}')
    if [[ -z "$avail" || "$avail" -lt "$need" ]]; then
        _xas_log "only ${avail:-?} kB free under $root: running in place"
        return 1
    fi
    # Reap dirs of jobs that died without cleaning up (heartbeat stale), never a live job's.
    find "$root" -mindepth 2 -maxdepth 2 -name .heartbeat -mmin +"${XAS_SCRATCH_STALE_MIN:-120}" -printf '%h\0' 2>/dev/null \
        | xargs -0 -r rm -rf 2>/dev/null
    local d=$root/${job%%.*}-$tag
    rm -rf "$d"
    mkdir -p "$d" || { _xas_log "cannot create $d: running in place"; return 1; }
    touch "$d/.heartbeat"
    ( while sleep "${XAS_HEARTBEAT_S:-300}"; do touch "$d/.heartbeat" 2>/dev/null || exit 0; done ) </dev/null >/dev/null 2>&1 &
    _XAS_HB_PID=$!
    XAS_SCRATCH_DIR=$d
}

xas_scratch_drop() {
    [[ -n "$_XAS_HB_PID" ]] && kill "$_XAS_HB_PID" 2>/dev/null
    _XAS_HB_PID=
    if [[ -n "$XAS_SCRATCH_DIR" && -d "$XAS_SCRATCH_DIR" ]]; then
        cd / && rm -rf "$XAS_SCRATCH_DIR"
    fi
    XAS_SCRATCH_DIR=
}

# Copy one file into a shared-filesystem directory atomically (temp name + rename), so a reader never sees half a file.
xas_put() {
    local src=$1 dest_dir=$2 name=${3:-$(basename "$1")}
    local tmp="$dest_dir/.$name.part-${SLURM_JOB_ID:-$$}"
    cp -p "$src" "$tmp" && mv -f "$tmp" "$dest_dir/$name" || { rm -f "$tmp"; return 1; }
}
