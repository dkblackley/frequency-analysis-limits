#!/usr/bin/env bash
# Build the Rust build/run container locally from rust_env.dev and push it to hopper
# as /scratch/dblackle/rust_env.sif (the path every *.slurm script uses).
#
# Usage: ./build_and_push_container.sh
#   REMOTE=hopper REMOTE_SIF=/scratch/dblackle/rust_env.sif ./build_and_push_container.sh
#
# Two ways to build, picked automatically:
#   1. apptainer/singularity installed locally -> build the .sif here, scp it over.
#   2. only docker installed -> build a docker image from the same %post steps,
#      scp the image tarball, and convert it to a .sif on hopper.
set -euo pipefail

cd "$(dirname "$0")"

DEF_FILE="rust_env.dev"
REMOTE="${REMOTE:-hopper}"
REMOTE_SIF="${REMOTE_SIF:-/scratch/dblackle/rust_env.sif}"
REMOTE_DIR="$(dirname "$REMOTE_SIF")"
IMAGE_TAG="rust_env:latest"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Upload under a temporary name and rename at the end, so a job that is already
# running with the old image never sees a half-written file.
REMOTE_TMP="$REMOTE_SIF.tmp.$$"

if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
    SG="$(command -v apptainer || command -v singularity)"
    echo "Building $DEF_FILE with $SG..."
    if [ "$(id -u)" -eq 0 ]; then
        "$SG" build "$WORK_DIR/rust_env.sif" "$DEF_FILE"
    else
        sudo "$SG" build "$WORK_DIR/rust_env.sif" "$DEF_FILE"
    fi

    echo "Copying image to $REMOTE:$REMOTE_SIF..."
    scp "$WORK_DIR/rust_env.sif" "$REMOTE:$REMOTE_TMP"
    ssh "$REMOTE" "mv -f '$REMOTE_TMP' '$REMOTE_SIF'"

elif command -v docker >/dev/null 2>&1; then
    # Reuse the base image and %post steps from rust_env.dev so there is one definition.
    BASE_IMAGE="$(awk '/^From:/ {print $2; exit}' "$DEF_FILE")"
    awk '/^%post/ {on=1; next} /^%/ {on=0} on' "$DEF_FILE" > "$WORK_DIR/post.sh"
    cat > "$WORK_DIR/Dockerfile" <<EOF
FROM $BASE_IMAGE
COPY post.sh /tmp/post.sh
RUN bash -euxo pipefail /tmp/post.sh && rm /tmp/post.sh
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib
EOF

    echo "Building docker image $IMAGE_TAG from $DEF_FILE..."
    docker build --platform linux/amd64 -t "$IMAGE_TAG" "$WORK_DIR"
    docker save "$IMAGE_TAG" -o "$WORK_DIR/rust_env.tar"

    echo "Copying image tarball to $REMOTE and converting to $REMOTE_SIF..."
    scp "$WORK_DIR/rust_env.tar" "$REMOTE:$REMOTE_DIR/rust_env.tar"
    ssh "$REMOTE" "bash -lc 'set -e
        module --quiet purge
        module load hosts/hopper
        module load singularity
        singularity build --force \"$REMOTE_TMP\" docker-archive://$REMOTE_DIR/rust_env.tar
        mv -f \"$REMOTE_TMP\" \"$REMOTE_SIF\"
        rm -f $REMOTE_DIR/rust_env.tar'"
else
    echo "Need apptainer/singularity or docker installed locally." >&2
    exit 1
fi

echo "Done. Submit the build with: ssh $REMOTE 'cd /scratch/dblackle/frequency-analysis-limits && sbatch build_bin.slurm'"
