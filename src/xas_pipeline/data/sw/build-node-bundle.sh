#!/bin/bash
# build-node-bundle.sh -- build the node-local OpenMPI + HCOLL + FEFF10 bundle the job scripts stage per node
# (node-env.sh, $PIPELINE_NODE_BUNDLE). Generalised from Riti-CSD-calcs/scan-spectra-v3/sw/build-mpi-bundle.sh.
#
# Why: /sdf/group/ssrl/sarangi/sw/openmpi/lib became a WekaFS metadata hot spot under ~800 jobs (every lookup
# 0.2-0.45 s, `mpirun -n 16 /bin/true` ~28 s, ~14 MPI launches per CORVUS run). From a node-local copy: 0.5 s.
# The binaries are byte-for-byte the shared builds except their ELF RUNPATH, rewritten (patchelf) from /sdf/group/...
# to the fixed node path, so the loader never probes the shared directory. Compiled-in OpenMPI prefixes are redirected
# at run time by OPAL_PREFIX / PRTE_PREFIX / PMIX_PREFIX / HWLOC_PLUGINS_PATH (node-env.sh sets them).
#
#   bash build-node-bundle.sh <out dir> [tag]      -> <out dir>/<tag>.tar + .sha256   (tag default: ompi-feff-v1)
#   then in the pipeline .env:   PIPELINE_NODE_BUNDLE=<out dir>/<tag>.tar
#
# Sources, from the pipeline .env (or the environment): PIPELINE_OPENMPI_ROOT, PIPELINE_HPCX_ROOT (its hcoll/lib), the
# FEFF bin dir of DYM2FEFFINP_BIN, and $HOME/.Corvus (corvus.conf + corvutils; its feff/dmdw entries are rewritten to
# the bundle). Node root: ${XAS_NODE_SW_ROOT:-/lscratch/$USER/xas-pipeline-sw}/<tag>, recorded in bundle.env; a node
# that would stage it anywhere else refuses it. Rebuild under a NEW tag whenever a source changes.
set -euo pipefail
OUT=${1:?usage: build-node-bundle.sh <out dir> [tag]}
TAG=${2:-ompi-feff-v1}
ENV_FILE=${PIPELINE_ENV:-}
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
OMPI=${PIPELINE_OPENMPI_ROOT:?set PIPELINE_OPENMPI_ROOT (or PIPELINE_ENV=<pipeline .env>)}
HCOLL=${PIPELINE_HPCX_ROOT:?set PIPELINE_HPCX_ROOT}/hcoll/lib
FEFF=$(dirname "${DYM2FEFFINP_BIN:?set DYM2FEFFINP_BIN}")
NODE_ROOT=${XAS_NODE_SW_ROOT:-/lscratch/$USER/xas-pipeline-sw}/$TAG
PE=(uvx --from patchelf==0.17.2.1 patchelf)
[[ -e "$OUT/$TAG.tar" ]] && { echo "$OUT/$TAG.tar exists: pick a new tag" >&2; exit 1; }
mkdir -p "$OUT"

B=$(mktemp -d "${TMPDIR:-/tmp}/nodebundle.XXXXXX"); trap 'rm -rf "$B"' EXIT
R=$B/$TAG
mkdir -p "$R/home/.Corvus/corvutils"
cp -a "$OMPI" "$R/openmpi"
cp -a "$HCOLL" "$R/hcoll-lib"
cp -a "$FEFF" "$R/feff-bin"
cp -a "$HOME/.Corvus/corvutils/." "$R/home/.Corvus/corvutils/"
sed -E "s#^(dmdw|feff)( *):.*#\1\2:$NODE_ROOT/feff-bin#" "$HOME/.Corvus/corvus.conf" > "$R/home/.Corvus/corvus.conf"
{
    echo "NODE_ROOT=$NODE_ROOT"
    echo "BUILT=$(date -u +%FT%TZ) by $USER on $(hostname)"
    echo "OPENMPI=$OMPI"
    echo "HCOLL=$HCOLL"
    echo "FEFF=$FEFF"
} > "$R/bundle.env"

n=0
while IFS= read -r -d '' f; do
    [[ -L "$f" ]] && continue
    head -c4 "$f" 2>/dev/null | grep -q ELF || continue
    rp=$("${PE[@]}" --print-rpath "$f" 2>/dev/null || true)
    [[ "$rp" == *"/sdf/group/"* ]] || continue
    new=${rp//$OMPI/$NODE_ROOT/openmpi}
    new=${new//$HCOLL/$NODE_ROOT/hcoll-lib}
    [[ "$f" == "$R/feff-bin/"* ]] && new="$NODE_ROOT/openmpi/lib:$NODE_ROOT/hcoll-lib"
    [[ "$new" == *"/sdf/group/"* ]] && { echo "unhandled RUNPATH in $f: $rp" >&2; exit 1; }
    "${PE[@]}" --set-rpath "$new" "$f"; n=$((n + 1))
done < <(find "$R" -type f -print0)
echo "patched RUNPATH of $n ELF files"

tar -C "$B" -cf "$OUT/$TAG.tar.tmp" "$TAG"
mv "$OUT/$TAG.tar.tmp" "$OUT/$TAG.tar"
(cd "$OUT" && sha256sum "$TAG.tar" > "$TAG.tar.sha256")
ls -la "$OUT/$TAG.tar"; cat "$OUT/$TAG.tar.sha256"
