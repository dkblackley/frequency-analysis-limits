#!/bin/bash
# run_me.sh: run this on hopper after every fix. It never deletes results.
#
#   bash run_me.sh                    # 50x50 sampled sweep: 6 datasets x 3 dists x 1,2,5,10,20,30% queries
#   bash run_me.sh drink spitz        # only these datasets
#   HIGHDIM=1 bash run_me.sh          # also the high-dim runs (nh 16x16x14, grid 2D..6D; gaussian, 30%)
#   bash run_me.sh status             # only refresh logs/resource_usage.tsv and print where every run is
#   DRY_RUN=1 bash run_me.sh          # show what it would submit, change nothing
#   PROBS="0.1 0.2" DISTS=uniform bash run_me.sh   # narrow the sweep
#
# What it does, in order:
#   1. git pull --ff-only (NO_PULL=1 skips it); if this script changed, it restarts itself
#   2. makes the *.sh / *.slurm scripts executable again
#   3. if bin/lama is missing or older than the Rust code (or REBUILD=1), submits build_bin.slurm; every
#      job submitted below then waits for that build (afterok) and runs the snapshot it produces
#   4. refreshes logs/resource_usage.tsv from sacct/sstat: per job its state, elapsed / limit, ACTUAL peak
#      memory vs requested, and for running jobs how far the LNS got (from the log)
#   5. submits every config that has no new-format result and no queued/running job, sizing each job from
#      its own history: OOM -> 2x memory (bigmem past NORMAL_MAX_GB), TIMEOUT (slurm killed it, no result)
#      -> 2x time, and a config that completed before asks for its real peak + 50% instead of the default.
#      A run that hit its LNS deadline before OUTPUTS (1000) consistent databases is rerun with the SAME
#      time limit (more time did not help: runs converge in < 1 h or stall). First attempts ask small
#      (10 GB, 1-8 h by dataset size) so they start quickly.
#   6. prints the status table
# A result counts as done only if results_*.json was written by the pair-LNS code (it has an "lns" field);
# older results are copied to limits/archive_pre_lns/ before their config is rerun.
# Everything it prints is also saved in logs/run_me/<timestamp>.log.

set -uo pipefail

# ---------------------------------------------------------------- sweep definition
GRID_2D="${GRID_2D:-50x50}"
NAMES_2D=(drink spitz busstop cali highway shopparis)
# (arrays get their own names: bash cannot export arrays, so reusing PROBS etc. lost them on the restart)
read -r -a DISTS_ARR <<< "${DISTS:-uniform gaussian beta}"
read -r -a PROBS_ARR <<< "${PROBS:-0.01 0.02 0.05 0.1 0.2 0.3}"
HIGHDIM_DIST="${HIGHDIM_DIST:-gaussian}"
read -r -a HIGHDIM_PROBS_ARR <<< "${HIGHDIM_PROBS:-0.3}"

# ---------------------------------------------------------------- resources
# Measured: Spitz 50x50 peaks at 4.1 GB, nearly all of it the 50x50 query distribution (same for every
# dataset); the old 240 GB was the t = 2 tables, gone now. Ask small so jobs start fast: an OOM doubles it.
CPUS_2D="${CPUS_2D:-16}"
MEM_2D_GB="${MEM_2D_GB:-10}"
OUTPUTS="${OUTPUTS:-1000}"   # distinct consistent databases per run (--lns-outputs)
NORMAL_MAX_GB=170      # normal nodes have 180 GB; above this go to bigmem
BIGMEM_MAX_GB=1000
MAX_TIME_H=160         # partitions allow 7 days
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
SETUP_MARGIN_FRAC=0.15 # the LNS gets (1 - this) of the time limit, minus 15 min, so jobs write results in time

DB_ROOT="databases"
TSV="logs/resource_usage.tsv"
LOGDIR="logs/lama"

# ---------------------------------------------------------------- helpers
to_gb() { # 251657732K / 245757.50M / 240G / 1T -> GB (1 decimal)
    awk -v x="$1" 'BEGIN { if (x == "" ) { print ""; exit }
        u = substr(x, length(x)); v = substr(x, 1, length(x) - 1) + 0
        if (u == "K") v /= 1048576; else if (u == "M") v /= 1024; else if (u == "T") v *= 1024
        else if (u != "G") v = x / 1073741824
        printf "%.1f", v }'
}
to_hours() { # [D-]HH:MM:SS -> hours
    awk -v t="$1" 'BEGIN { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
        n = split(t, b, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + b[i]
        printf "%.2f", d * 24 + s / 3600 }'
}

# config line: key name dir dim dist prob default_mem_gb default_time_h cpus
configs() {
    local names=("$@") n d p
    [[ ${#names[@]} -eq 0 ]] && names=("${NAMES_2D[@]}")
    for n in "${names[@]}"; do
        for d in "${DISTS_ARR[@]}"; do
            for p in "${PROBS_ARR[@]}"; do
                echo "${n}_${d}_p${p}@${GRID_2D} $n $GRID_2D 2 $d $p $MEM_2D_GB $(time_2d "$n" "$p") $CPUS_2D"
            done
        done
    done
    if [[ "${HIGHDIM:-0}" == 1 ]]; then
        for p in "${HIGHDIM_PROBS_ARR[@]}"; do
            # nh/grid <=5D: the query distribution is <= 4M queries, a few GB. 6D is 86M queries (old runs
            # needed hundreds of GB just for that), and its search is likely too slow to converge in time.
            # times from the 2026-11-17 runs: nh used all of 12 h, 4D all of 4 h (367 of 1000 outputs),
            # 2D/3D finished in minutes; 5D/6D never got past the start solve
            echo "nh_${HIGHDIM_DIST}_p${p}@16x16x14 nh 16x16x14 3 $HIGHDIM_DIST $p 16 24 32"
            local dim dir
            for dim in 2 3 4 5 6; do
                dir=$(printf '6%.0s' $(seq 1 "$dim") | sed 's/./&x/g; s/x$//')
                if [[ $dim -le 3 ]]; then echo "grid${dim}d_${HIGHDIM_DIST}_p${p}@$dir grid $dir $dim $HIGHDIM_DIST $p 8 4 16"
                elif [[ $dim -eq 4 ]]; then echo "grid${dim}d_${HIGHDIM_DIST}_p${p}@$dir grid $dir $dim $HIGHDIM_DIST $p 8 12 16"
                elif [[ $dim -eq 5 ]]; then echo "grid${dim}d_${HIGHDIM_DIST}_p${p}@$dir grid $dir $dim $HIGHDIM_DIST $p 32 24 32"
                else echo "grid${dim}d_${HIGHDIM_DIST}_p${p}@$dir grid $dir $dim $HIGHDIM_DIST $p 600 48 48"; fi
            done
        done
    fi
}

# first-attempt time limit (h): the search scales with the number of records, and 1-2% queries mean
# bigger candidate sets. A run that hits its deadline before OUTPUTS databases is rerun with the same time.
time_2d() {
    local n=$1 p=$2 h
    case "$n" in spitz) h=1 ;; drink) h=2 ;; *) h=4 ;; esac
    awk -v p="$p" 'BEGIN { exit !(p + 0 <= 0.02) }' && h=$((h * 2))
    echo "$h"
}

job_name() { echo "LAMa_${1%@*}_${1#*@}"; }
result_file() { echo "$DB_ROOT/$3/$2/limits/results_${2}_${5}_p${6}.json"; } # key name dir dim dist prob
recon_file() { echo "$DB_ROOT/$3/$2/limits/${2}_${5}_p${6}_reconstruction.json"; }
# none: no result from the pair-LNS code | done | more: it hit its deadline before OUTPUTS consistent
# databases (rerun with 2x time) | exhausted: finished early with fewer (no more neighbours to find; done)
result_state() {
    local f; f=$(result_file "$@")
    [[ -f "$f" ]] && grep -q '"lns"' "$f" || { echo none; return; }
    python3 - "$f" "$OUTPUTS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); l = d["lns"]; want = int(sys.argv[2])
if l["output_is_consistent"] and l["returned"] >= want: print("done")
elif l["hit_deadline"]: print("more")
else: print("exhausted")
PY
}
result_line() {
    python3 - "$(result_file "$@")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); l = d["lns"]
print(f"{l['returned']} DBs ({'all consistent' if l['output_is_consistent'] else 'NOT consistent: best found'}), "
      f"true DB in output={l['true_db_in_output']}, true DB pairs outside eps={l['true_db_pairs_outside']}, "
      f"{d['time_taken'] / 3600:.2f}h{', hit deadline' if l['hit_deadline'] else ''}")
PY
}

progress_of() { # log file -> short description of where the run is
    local f=$1 line
    [[ -f "$f" ]] || { echo "no log yet"; return; }
    line=$(grep -aE "LNS progress|t = 1 start|True database|Starting LAMA!|Making observed|Sampling [0-9]+ queries|Beginning to set up|LAMa completely finished|panicked|Killed" "$f" | tail -1)
    case "$line" in
        *"LNS progress"*)   echo "LNS ${line#*LNS progress: }" | cut -c1-110 ;;
        *"t = 1 start"*)    echo "LNS started" ;;
        *"True database"*)  echo "scoring the true DB" ;;
        *"Starting LAMA!"*) echo "t=1 tables" ;;
        *"Making observed"*|*"Sampling"*) echo "sampling queries" ;;
        *"Beginning to set up"*) echo "building the query distribution" ;;
        *"finished"*)       echo "writing results" ;;
        *panicked*)         echo "PANIC: $(grep -a -A1 panicked "$f" | tail -1 | cut -c1-90)" ;;
        *)                  echo "starting" ;;
    esac
}

# ---------------------------------------------------------------- 4. resource_usage.tsv
refresh_usage() {
    mkdir -p logs
    local since; since=$(cat logs/run_me/since 2>/dev/null || echo 2026-11-17)
    local tmp; tmp=$(mktemp)
    # job lines and their .batch step (which carries MaxRSS)
    sacct -u "$USER" -S "$since" -n -P \
        --format=JobID,JobName%80,State,Elapsed,Timelimit,ReqMem,MaxRSS,AllocCPUS,Submit 2>/dev/null > "$tmp"
    {
        printf "job_name\tjob_id\tstate\telapsed_h\tlimit_h\tpeak_gb\treq_gb\tcpus\tsubmitted\tprogress\n"
        awk -F'|' '
            $1 ~ /\.batch$/ { id = $1; sub(/\.batch$/, "", id); rss[id] = $7; next }
            $1 ~ /\./ { next }
            $2 ~ /^LAMa_/ { order[++n] = $1; line[$1] = $0 }
            END { for (i = 1; i <= n; i++) { id = order[i]; print line[id] "|" rss[id] } }' "$tmp" |
        while IFS='|' read -r id name state elapsed limit reqmem _ cpus submit rss; do
            state=${state%% *}
            if [[ "$state" == RUNNING ]]; then
                rss=$(sstat -n -P -j "${id}.batch" --format=MaxRSS 2>/dev/null | head -1)
                prog=$(progress_of "$LOGDIR/${name}_${id}.err")
            elif [[ "$state" == PENDING ]]; then prog="queued"
            else prog=$(progress_of "$LOGDIR/${name}_${id}.err"); fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$id" "$state" "$(to_hours "$elapsed")" \
                "$(to_hours "$limit")" "$(to_gb "$rss")" "$(to_gb "$reqmem")" "$cpus" "$submit" "$prog"
        done
    } > "$TSV"
    rm -f "$tmp"
}

# history of one job name from the tsv: attempts, last state, last req mem/time, max peak
history_of() {
    awk -F'\t' -v n="$1" 'NR > 1 && $1 == n {
            a++; st = $3; req = $7; lim = $5; if ($6 + 0 > pk) pk = $6 + 0; if (st == "COMPLETED" && $6 + 0 > okpk) okpk = $6 + 0 }
        END { printf "%d %s %s %s %s %s\n", a + 0, (st == "" ? "-" : st), (req == "" ? 0 : req), (lim == "" ? 0 : lim), pk + 0, okpk + 0 }' "$TSV"
}

# ---------------------------------------------------------------- main
main() {
    local self; self=$(readlink -f "$0")
    cd "$(dirname "$self")" || exit 1
    local mode=submit
    if [[ "${1:-}" == status ]]; then mode=status; shift; fi

    # 1. update (before the log starts, so a restart doesn't log twice)
    if [[ $mode == submit && -z "${NO_PULL:-}" && -z "${DRY_RUN:-}" ]]; then
        local before; before=$(git hash-object "$self")
        echo "== git pull"
        git pull --ff-only || { echo "git pull failed (local changes? see git status). Fix it, or NO_PULL=1 bash run_me.sh"; exit 1; }
        if [[ "$before" != "$(git hash-object "$self")" ]]; then
            echo "run_me.sh changed, restarting it"
            NO_PULL=1 exec bash "$self" "$@"
        fi
    fi

    local ts; ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p logs/run_me "$LOGDIR"
    exec > >(tee -a "logs/run_me/$ts.log") 2>&1
    echo "run_me.sh $(date +%FT%T)  mode: $mode${DRY_RUN:+  (DRY_RUN: nothing is submitted)}"
    git log -1 --format='   code: %h %ci %s'

    # 2. exec bits
    chmod u+x ./*.sh ./*.slurm 2>/dev/null || true

    # queued/running LAMa jobs, by name
    local queued; queued=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep '^LAMa_' | sort -u)

    # 3. binary
    local dep=""
    if [[ $mode == submit ]]; then
        echo "== binary"
        local why=""
        if [[ -n "${REBUILD:-}" ]]; then why="REBUILD=1"
        elif [[ ! -x bin/lama ]]; then why="no bin/lama"
        elif [[ -n "$(find src Cargo.toml Cargo.lock -newer "$(readlink -f bin/lama)" -print -quit 2>/dev/null)" ]]; then why="older than the Rust code"
        fi
        local building; building=$(squeue -u "$USER" -h -n LAMA_build -o %i | head -1)
        if [[ -n "$building" ]]; then
            echo "   build $building already queued/running; new jobs wait for it"; dep="--dependency=afterok:$building"
        elif [[ -z "$why" ]]; then
            echo "   up to date: $(readlink bin/lama)"
        elif [[ -n "${DRY_RUN:-}" ]]; then
            echo "   would build ($why)"
        else
            local bid; bid=$(sbatch --parsable build_bin.slurm) || { echo "   build submit FAILED"; exit 1; }
            echo "   building ($why): job $bid; new jobs wait for it (afterok)"
            dep="--dependency=afterok:$bid"
        fi
    fi

    # 4. usage
    echo "== $TSV"
    refresh_usage
    echo "   $(($(wc -l < "$TSV") - 1)) LAMa jobs since $(cat logs/run_me/since 2>/dev/null || echo 2026-11-17)"

    # 5. submit
    local n_done=0 n_queued=0 n_sub=0 n_giveup=0
    local status_lines=()
    while read -r key name dir dim dist prob mem time cpus; do
        local jn; jn=$(job_name "$key")
        local res; res=$(result_file "$key" "$name" "$dir" "$dim" "$dist" "$prob")
        local rstate; rstate=$(result_state "$key" "$name" "$dir" "$dim" "$dist" "$prob")
        if [[ "$rstate" == done || "$rstate" == exhausted ]]; then
            n_done=$((n_done + 1))
            status_lines+=("DONE     $key  $(result_line "$key" "$name" "$dir" "$dim" "$dist" "$prob")")
            continue
        fi
        if grep -qx "$jn" <<< "$queued"; then
            n_queued=$((n_queued + 1))
            status_lines+=("QUEUED   $key  $(awk -F'\t' -v n="$jn" '$1 == n { p = $3 " " $10 " (peak " $6 "/" $7 " GB, " $4 "/" $5 " h)" } END { print p }' "$TSV")")
            continue
        fi
        [[ $mode == status ]] && { status_lines+=("TODO     $key  $(history_of "$jn" | awk '{ print ($1 ? $1 " attempt(s), last " $2 : "never run") }')"); continue; }

        # size from this config's history
        local attempts last req lim peak okpeak
        read -r attempts last req lim peak okpeak <<< "$(history_of "$jn")"
        if (( attempts >= MAX_ATTEMPTS )); then
            n_giveup=$((n_giveup + 1))
            status_lines+=("GAVE UP  $key  $attempts attempts, last $last: see $LOGDIR/${jn}_*.err (MAX_ATTEMPTS=$MAX_ATTEMPTS)")
            continue
        fi
        # hit its LNS deadline before OUTPUTS consistent databases: rerun with the default time (no doubling)
        case "$last" in
            OUT_OF_MEMORY) mem=$(awk -v r="$req" -v p="$peak" -v m="$mem" 'BEGIN { x = 2 * r; if (1.5 * p > x) x = 1.5 * p; if (m > x) x = m; printf "%d", x + 0.5 }') ;;
            TIMEOUT)       time=$(awk -v l="$lim" -v t="$time" 'BEGIN { x = 2 * l; if (t > x) x = t; printf "%d", x + 0.5 }') ;;
        esac
        # a config that completed before (e.g. an older code version): its real peak + 50%, at least 8 GB
        if [[ "$last" != OUT_OF_MEMORY ]] && awk -v p="$okpeak" 'BEGIN { exit !(p > 0) }'; then
            mem=$(awk -v p="$okpeak" 'BEGIN { x = 1.5 * p; if (x < 8) x = 8; printf "%d", x + 0.5 }')
        fi
        (( time > MAX_TIME_H )) && time=$MAX_TIME_H
        local part=normal
        (( mem > NORMAL_MAX_GB )) && part=bigmem
        (( mem > BIGMEM_MAX_GB )) && mem=$BIGMEM_MAX_GB
        local lns_h; lns_h=$(awk -v t="$time" -v f="$SETUP_MARGIN_FRAC" 'BEGIN { x = t * (1 - f) - 0.25; if (x < 0.25) x = 0.25; printf "%.2f", x }')
        local args="--dir-path $DB_ROOT/$dir --name $name --t 2 --dim $dim --dist $dist --padding 0 --query-percent $prob --lns-hours $lns_h --lns-outputs $OUTPUTS"

        # keep earlier results next to the new ones (old code, or a pair-LNS run that ran out of time)
        local old; old=$(dirname "$res")/archive_pre_lns
        [[ "$rstate" == more ]] && old=$(dirname "$res")/archive_short_runs/$(date +%Y%m%d_%H%M%S)
        if [[ -f "$res" && -z "${DRY_RUN:-}" ]]; then
            mkdir -p "$old"
            cp -n "$res" "$old/" 2>/dev/null
            cp -n "$(recon_file "$key" "$name" "$dir" "$dim" "$dist" "$prob")" "$old/" 2>/dev/null
        fi

        local desc="${mem}G ${cpus}cpu ${time}h $part (LNS ${lns_h}h)$([[ $attempts -gt 0 ]] && echo ", attempt $((attempts + 1)) after $last")$([[ $rstate == more ]] && echo ", last run hit its deadline")"
        if [[ -n "${DRY_RUN:-}" ]]; then
            status_lines+=("WOULD    $key  $desc")
            continue
        fi
        local jid
        jid=$(sbatch --parsable $dep --job-name="$jn" --partition="$part" --cpus-per-task="$cpus" --mem="${mem}G" \
            --time="$(awk -v t="$time" 'BEGIN { printf "%d-%02d:00:00", int(t / 24), t % 24 }')" \
            --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
            --export=ALL,LAMA_ARGS="$args" run_lama_one.slurm) || { status_lines+=("SUBMIT FAILED  $key"); continue; }
        n_sub=$((n_sub + 1))
        status_lines+=("SUBMIT   $key  job $jid: $desc")
    done < <(configs "$@")

    # 6. summary
    echo
    echo "== status ($n_done done, $n_queued queued/running, $n_sub submitted now, $n_giveup gave up)"
    printf '%s\n' "${status_lines[@]}" | sort
    echo
    echo "Per job (actual peak memory vs requested, how far it got): column -t -s\$'\\t' $TSV | less -S"
    echo "Saved to logs/run_me/$ts.log. Later: bash run_me.sh status"
}

# main is called on the last line, so bash has read the whole file before git pull can change it
main "$@"
