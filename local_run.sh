#!/bin/bash -l

# ============================================================
# Which experiment to run locally. Mirrors the .slurm scripts,
# but each is pinned to a single configuration so it's a quick
# end-to-end check that the pipeline finishes.
#
#   perfect        -> run_LAMA_perfect.slurm       (cali, 50x50, gaussian, t=3, 100% queries)
#   probs_2d       -> run_LAMA_probs_2d.slurm      (cali, 15x15_run_1, gaussian, t=2, 30% queries)
#   probs_highdim  -> run_LAMA_probs_highdim.slurm (nh, 16x16x14, gaussian, t=2, 30% queries)
#   noise          -> run_LAMA_noise.slurm         (cali, 15x15_run_1, gaussian, t=2, 100% queries, 1% TV noise)
#   all            -> every one of the above, in order
#
# Can also be overridden from the command line: ./local_run.sh probs_2d
# ============================================================
EXPERIMENT="${1:-noise}"

export RUST_LOG="debug"
export RUST_BACKTRACE="1"

BIN=./target/release/frequency_analysis_limits
DB_ROOT="databases"
FAILED_RUNS=()

case "$EXPERIMENT" in
    perfect|probs_2d|probs_highdim|noise|all) ;;
    *)
        echo "Unknown EXPERIMENT '$EXPERIMENT'. Use one of: perfect, probs_2d, probs_highdim, noise, all"
        exit 1
        ;;
esac

echo "Building project..."
cargo rustc --bin frequency_analysis_limits --release -- -Clink-arg=-lprotobuf
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

# run_lama <label> <args...>
# Runs the binary with the given args and records a failure under <label>.
run_lama() {
    local label=$1
    shift
    echo "========================================"
    echo "Experiment: $label"
    echo "Args:       $*"
    echo "========================================"
    $BIN "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "Error: $label failed (exit code $rc)."
    else
        echo "$label completed successfully."
    fi
    echo ""
    return $rc
}

# --- Perfect knowledge: cali @ 50x50, gaussian, t=3 (retry at t=2 like the slurm script) ---
run_perfect() {
    local common=(--dir-path "$DB_ROOT/50x50" --name cali --dist gaussian --padding 0 --query-percent 1.0)
    if ! run_lama "perfect (t=3)" "${common[@]}" --t 3; then
        echo "Retrying perfect with t=2..."
        if run_lama "perfect (t=2 retry)" "${common[@]}" --t 2; then
            FAILED_RUNS+=("perfect | only reached t=2")
        else
            FAILED_RUNS+=("perfect | failed at t=3 and t=2")
        fi
    fi
}

# --- Probabilistic 2D: cali @ 15x15_run_1, gaussian, 30% of queries ---
run_probs_2d() {
    run_lama "probs_2d" \
        --dir-path "$DB_ROOT/15x15_run_1" \
        --name cali \
        --t 2 \
        --dist gaussian \
        --padding 0 \
        --query-percent 0.3 \
        || FAILED_RUNS+=("probs_2d")
}

# --- Probabilistic high-dim: nh @ 16x16x14 (3D), gaussian, 30% of queries ---
run_probs_highdim() {
    run_lama "probs_highdim" \
        --dir-path "$DB_ROOT/16x16x14" \
        --name nh \
        --t 2 \
        --dim 3 \
        --dist gaussian \
        --padding 0 \
        --query-percent 0.3 \
        || FAILED_RUNS+=("probs_highdim")
}

# --- Noisy distribution: cali @ 15x15_run_1, gaussian, all queries, 1% TV noise ---
run_noise() {
    if ! $BIN --help | grep -q -- "--dist-noise"; then
        echo "This binary has no --dist-noise flag: check out the branch with it and rebuild."
        FAILED_RUNS+=("noise | binary missing --dist-noise")
        return
    fi
    run_lama "noise" \
        --dir-path "$DB_ROOT/15x15_run_1" \
        --name cali \
        --t 2 \
        --dist gaussian \
        --padding 0 \
        --query-percent 1.0 \
        --dist-noise 0.01 \
        || FAILED_RUNS+=("noise")
}

case "$EXPERIMENT" in
    perfect)       run_perfect ;;
    probs_2d)      run_probs_2d ;;
    probs_highdim) run_probs_highdim ;;
    noise)         run_noise ;;
    all)
        run_perfect
        run_probs_2d
        run_probs_highdim
        run_noise
        ;;
esac

echo "All runs completed."
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
    echo "The following experiments failed (or fell back to a lower t):"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo "All experiments processed successfully."
fi
