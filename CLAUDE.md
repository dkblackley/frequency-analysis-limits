# AGENT.md: how this repository works

Read this first. It describes what each file does, where results live, how experiments are run, and the
non-obvious things that have bitten people before. Last updated 2026-11-17 (NDSS rebuttal, Reviewer 356A).

## What the project is

LAMa ("Leakage Abuse via MAtching") reconstructs a database of 2D/3D/nD points from range-query leakage
(how often record tuples are returned together). It is compared against Even Less, Remin/Remin+ (embedding
baselines, run elsewhere, outputs only stored here) and Falzon et al. (constructed analytically, see below).

Two settings:
- **Ideal / "perfect"** (`--query-percent 1.0`): every query observed, exact frequencies. LAMa enumerates
  every database within epsilon (~1e-6). Uses the table-based translator + exhaustive Pumpkin enumeration.
- **Realistic / sampled** (`--query-percent < 1`): a fraction of queries is sampled i.i.d. from the query
  distribution; epsilon comes from a VC bound; LAMa outputs databases whose every t-tuple is within epsilon.
  Since 2026-11-17 this uses the pair LNS in `src/LAMA/lazy_lns.rs` (see below).

## Layout

```
src/main.rs                 CLI (clap). Calls LAMA::lama_attack, optional --post (GeoJSON export).
src/lib.rs                  Type aliases: Value=i64, Record=Vec<i64>, DomPair=(Record,Record) = a query/MBQ.
src/LAMA/mod.rs             lama_attack: loads the DB, builds the query distribution, samples queries,
                            builds validate_candidate, runs t=1, then either the perfect pipeline
                            (translator t>=2 + Solver) or the sampled pair LNS; writes results.
                            Also enclosing_sums (prefix-sum of query weights over enclosing queries).
src/LAMA/lazy_lns.rs        Sampled setting at t=2: min-error LNS that scores pairs on demand (no t=2
                            tables), then phase-2 sampling of distinct fully-consistent databases.
src/LAMA/translator.rs      Builds per-t-tuple tables of allowed plaintext assignments (process_t1,
                            process_cpsat_global for t>=2). Perfect setting only for t>=2 now.
src/LAMA/mini_solver.rs     CpModel (tables + all-different) on Pumpkin: exhaustive enumeration
                            (solve_partition) and the OLD sampled search min_error_lns (no longer called
                            by lama_attack for t=2; still used by process_cpsat_global if limit is set).
src/LAMA/solver.rs          Final solve of the perfect pipeline; maps solutions back to records.
src/LAMA/selector.rs        Query sampling (sample_percent_responses), epsilon (VC bound), responses.
src/LAMA/query.rs           QueryDistribution: uniform / gaussian / beta / flat weights, noise (TV),
                            cumulative (enclosing) probability per MBQ.
src/LAMA/utility.rs         dominates/encloses/get_mbq, DistributionType parsing.
src/dataloader/two_d.rs     TwoDMap (2D datasets from databases/<grid>/<name>/<name>.json).
src/dataloader/three_d.rs   ThreeDMap (nh, New Hampshire 3D).
src/dataloader/tester.rs    testDB: synthetic grid, testDB::new(dim, 6, 35) = 6^dim grid, 35% density, seed 42.
src/dataloader/mod.rs       flatten_nd/unflatten_nd (a record's "encrypted" id IS its flattened coordinate).
src/dataloader/raw_data.rs  Big embedded constant arrays (raw datasets). Don't read it whole.
src/plotting/mod.rs         DbResult (results_*.json schema), ReconstructionDataPoint.
src/plotting/post.rs        --post: GeoJSON export, Procrustes alignment, MSE (Rust side).
tests/integration.rs        End-to-end tests on small DBs. NOTE: end_to_end_sampled drives the OLD
                            table-based sampled path itself, not lama_attack / lazy_lns.
plotting/                   Python plots/tables for the paper (see "Plotting" below).
run_me.sh                   Cluster driver (hopper): pull, build, submit missing configs, track resources.
run_lama_one.slurm          One LAMa config; submitted by run_me.sh with all sizes on the command line.
build_bin.slurm             Builds in the singularity container; snapshots to bin/lama_<ts>_<hash>, bin/lama symlink.
run_LAMA_*.slurm            Older array scripts (perfect, probs_2d, probs_highdim, noise, noise_sweep).
local_run.sh, local_run_noise_sweep.sh   Local equivalents of the slurm scripts.
databases/                  Inputs AND outputs (gitignored). See "Data and file names".
figures/                    Intermediate JSON for figures (mostly not on the laptop).
```

## The algorithm, as implemented

1. **Query distribution** (`query.rs`). All M = prod_d n_d(n_d+1)/2 axis-aligned range queries
   (grid index 0..n-1 per axis) in the order `Selector::get_dom_pairs` returns them. Query i has weight:
   uniform 1; gaussian N(i; mu=M/2, sigma=M/5) pdf; beta Beta(2,1) pdf at (i+0.5)/M (= linear).
   CAVEAT: get_dom_pairs uses rayon par_bridge, so the ORDER (hence which query gets which gaussian/beta
   weight) differs between runs. Within a run attacker and sampler share the order.
   Expected probability of a tuple = sum of P(query) over queries enclosing the tuple's MBQ
   (prefix-sum precompute in `precompute_cumulative_results`).
2. **Sampling** (`selector.rs::sample_percent_responses`): ceil(M * p) queries i.i.d. (thread_rng, NOT
   seeded). Observed frequency of an MBQ = fraction of sampled queries enclosing it (mod.rs, via
   `enclosing_sums`, exactly equal to the old brute-force count; unit test `enclosing_sums_match_brute_force`).
3. **Epsilon**: `calculate_epsilon_real_vc(d, m, delta=0.1)` = sqrt(0.5/m * (d + ln(1/delta))), with
   d = `simple_bound_corollary2`. As implemented its SUKP capacity is the total number of incidences, so the
   optimum is always q = m and d = floor(log2 m) + 1: epsilon depends only on m. Computed in closed form
   (`simple_bound_corollary2_closed_form`, unit test checks it equals the LP). The LP version overflowed
   HiGHS' i32 row index at 6D. (The docstring says capacity should be the largest response; not changed.)
4. **validate_candidate(enc_tuple, plaintexts)** = |observed(enc MBQ) - expected(plaintext MBQ)| <= eps.
   Cost of a valid row = round(err^2 * 1e5) (small integers, many 0s).
5. **t = 1** (`Translator::process_t1`): for every record, every plaintext value within eps -> candidates C_v.
6a. **Perfect, t >= 2**: translator builds every t-tuple table (pairs of candidates within eps), Pumpkin
    enumerates all databases. Memory O(n^t |C|^t): fine with exact frequencies (|C| tiny).
6b. **Sampled** (`lazy_lns.rs`, only t = 2; `--t 3` is treated as 2):
    - phase 1: Pumpkin picks an all-different assignment from the t=1 tables (shared start), then each of
      `--lns-restarts` (64) restarts does LNS steps: free a random violated pair + random records (4 total),
      re-solve in Pumpkin minimising (pairs outside eps, then error), keep only if exactly better. Pairs with
      both records freed keep at most `--lns-max-rows` (1000) lowest-cost rows (tested 500/1000/2000/20000
      on Spitz 50x50 10%: 20000 never converged in 4 min, <=2000 did; 1000 found the most distinct DBs).
      Each restart does 1500 steps; with `--lns-hours` it continues past 1500 while still inconsistent,
      but (since 2026-11-18) is abandoned after `--lns-stall-steps` (1500) steps without improving, and
      new restarts (fresh seeds, past 64) keep starting until the deadline or the output target. Before
      that, only the first #CPU restarts ever ran under a deadline (stuck ones ground until it), which is
      why failed Nov-17 runs report 17 databases. The t = 1 start is ALWAYS a Pumpkin solve, 60 s (was 5 s: 40
      of 132 Nov-17 jobs found no start and wrote nothing, although exit code 0 / COMPLETED). If it still
      finds none, lama_attack returns RestartNeeded and main.rs reruns the WHOLE pipeline (new query
      sample), up to `--max-pipeline-attempts` (10), later attempts getting what is left of --lns-hours.
      results `lns.pipeline_attempt` says which attempt produced the result. The user wants every
      solve to stay a Pumpkin CSP solve: do NOT add non-CSP shortcuts (e.g. a bipartite matching start).
    - phase 2: from a fully consistent database, `sample_neighbour` re-assigns a few random records to a
      DIFFERENT assignment keeping every touched pair within eps (values in shuffled order) and records
      every distinct database, until `--lns-outputs` (1000) are found or the deadline passes.
    - output: the distinct fully consistent databases sorted by total error (index 0 = lowest error). If
      none were found: the restarts' best databases, flagged `output_is_consistent: false`.
    - also scores the TRUE database under the same objective (`true_db_*` fields).
    Why: the old sampled path built every t=2 table (1.4e9 rows for Shanghai 50x50 30%) and OOMed at 240 GB
    for every dataset except Amsterdam/Spitz; 1% OOMed/timed out; 5% panicked when no consistent DB was found.
    Measured new: Spitz 50x50 10% uniform: 1000 consistent DBs, 1.5 GB peak, ~6 min on 16 laptop threads.
    The outputs are a walk around the few restarts that converged (12 of 64 there), so they are correlated.
    Under UNIFORM the 8 symmetric images of the truth have identical frequencies, so outputs cluster around
    reflections/rotations (Spitz: one cluster MSE ~300, others 1650-1965); lowest error != truth there.
    In that test the truth had error 0 (better than every output, best 13) but was not reached.

## Data and file names (databases/, gitignored, also on hopper /scratch/dblackle/frequency-analysis-limits)

- Inputs: `databases/<grid>/<name>/<name>.json`. Grids: 15x15, 20x20 ... 50x50, 16x16x14 (nh), synthetic
  dirs 5x5, 6x6 ... 6x6x6x6x6x6 (name `grid`, data generated by testDB, dir only holds outputs).
- Dataset names -> paper names: busstop=Shanghai, shopparis=Paris, cali=California, drink=Amsterdam,
  highway=Manhattan, spitz=Spitz, nh=New Hampshire (3D). Records at 50x50: spitz 91, drink 371,
  cali 975, busstop 1002, shopparis 1112, highway 1584.
- LAMa outputs in `<grid>/<name>/limits/`:
  - perfect: `<name>_<dist>_e0_d0.9_reconstruction.json` / `results_<name>_<dist>_e0_d0.9.json`
    (old naming; current code writes `_p1`), `e0_d0.001` variants for some grids.
  - sampled: `<name>_<dist>_p<q>_reconstruction.json` and `results_<name>_<dist>_p<q>.json`, q in Rust f64
    Display (0.01, 0.1, 0.3; 1.0 -> "1").
  - noisy distribution: `..._p1_noise<tv>...` (15x15 TV sweep).
  - `archive_pre_lns/`: results from the old sampled code, copied by run_me.sh before rerunning;
    `archive_short_runs/<ts>/`: pair-LNS runs that hit their deadline before 1000 outputs.
- Reconstruction JSON: list of databases; each a list of {"true": [...], "reconstructed": [...],
  "unscaled_true": null}, one per record (same record order in every database).
- results JSON (`DbResult`): `number_of_reconstructions` is the number of RECORDS (historical bug);
  use `number_of_databases` (new) or the length of the reconstruction JSON. `eps`, `num_queries_used`,
  `time_taken`, `translator_meta` (constraints/recons per t), and for sampled runs `lns`:
  returned, target_outputs, output_is_consistent, restarts_consistent, best/true_db errors and pairs
  outside eps, true_db_in_output, hit_deadline, lns_seconds, lns_steps.
- Baselines (not produced here): `<grid>/<name>/even_less/<name>_prob<pct>.0_<dist>_<grid>_even_less.json`,
  `remin/..._classic.json`, `reminp/..._classic.json`; single embedding each ({"true","reconstructed"}),
  arbitrary rotation/scale. At 15x15 they exist for 5,10,15,20,25,30,100%; at 50x50 only 100%.
- 15x15 sampled LAMa runs (Apr 2026) used the OLD enumeration code capped at 10000 databases; 50x50
  drink/spitz 10-30% (Nov 5) used the old min_error_lns; everything run_me.sh produces uses lazy_lns.
- The paper's Table II LAMa numbers came from `databases/15x15_average`, which is NOT on the laptop.

## Running experiments (hopper)

Access: `ssh hopper` needs Duo; ~/.ssh/config has ControlMaster, so open one with `ssh -fN hopper` and
other ssh calls reuse it. Repo: /scratch/dblackle/frequency-analysis-limits, container
/scratch/dblackle/rust_env.sif (built from rust_env.dev).

```
bash run_me.sh                    # 50x50 sweep: 6 datasets x uniform/gaussian/beta x 1,2,5,10,20,30%
HIGHDIM=1 bash run_me.sh          # + nh 16x16x14 and grid 2D..6D (gaussian, 30%)
PROBS="0.03" bash run_me.sh       # other query fractions; DISTS=..., OUTPUTS=..., GRID_2D=...
bash run_me.sh status             # refresh logs/resource_usage.tsv + status table, submit nothing
DRY_RUN=1 bash run_me.sh          # show what would be submitted
```
run_me.sh: git pull --ff-only -> build_bin.slurm if bin/lama is older than src (jobs wait afterok) ->
`logs/resource_usage.tsv` (per job: state, elapsed/limit h, ACTUAL peak GB vs requested, progress from the
log) -> submits each config without a finished pair-LNS result and not queued, job name
`LAMa_<name>_<dist>_p<q>_<grid>`, logs in `logs/lama/<jobname>_<jobid>.{out,err}`. Sizing: first attempt
10 GB / 16 CPUs / 1 h (spitz), 2 h (drink), 4 h (others), x2 at <=2%; OOM -> 2x memory (bigmem > 170 GB);
TIMEOUT -> 2x time; a run that hit its LNS deadline before 1000 outputs -> rerun with the SAME time
(runs converge in < 1 h or stall; more time did not help). High-dim limits: nh 24 h, grid 2-3D 4 h, 4D 12 h,
5D 24 h, 6D 48 h. The LNS gets
85% of the limit minus 15 min, so jobs normally finish and write results. Re-run run_me.sh to resubmit
whatever failed; it never deletes results. Partitions: normal (7 days, 180 GB+ nodes), bigmem (1 TB).

Old (pre-2026-11-17) failure history, from sacct: job 3275393 (probs_2d array) and 3275408 (highdim).

## Building locally (laptop)

`.cargo/config.toml` adds `-l protobuf` (OR-tools leftover) and sets target-dir to
`~/.cargo-target/frequency_analysis_limits`. Locally: `RUSTFLAGS="" cargo build --release`
(binary in that target dir). good_lp is HiGHS-only (CBC not needed). Tests:
`RUSTFLAGS="" cargo test --release --lib -- enclosing_sums_match_brute_force corollary2_closed_form_matches_lp`.
Writes go to `<--dir-path>/<name>/limits`: run local experiments on a COPY of the input json in a scratch
dir, never on databases/ directly.

## Plotting (plotting/, run from that folder)

Python needs numpy, scipy, matplotlib, tqdm (no system env has scipy; make a venv).
- `main.py`: entry point; blocks for each figure are commented in/out. Constants: colors, labels, datasets.
- `data_loader.py`: loaders + `procrustes_align`, `calculate_mse` (mean squared Euclidean distance, not /dims),
  `compute_dataset_metrics` (Table II: per-record convex hull over candidates -> centroid / diameter-midpoint
  MSE, divided by dims), `load_convex_mse_data(..., grid=, limits_grid=)`.
- `working_rebuttal.py`: prints the rebuttal tables (items 1-5) as markdown; never estimates, prints MISSING.
  Item 2 alignment for baselines: bbox -> [1,N]^2, round, 8 orientations. Note: with 8 symmetric candidates
  the hull centroid/diameter midpoint is always the grid centre, so baseline eps no longer depends on method.
- `falzon.py`: Falzon et al.'s output E(D) built from the true DB (4 * 2^r databases, r reflectable
  components; r = 1 for all six datasets at 50x50). Not a re-run of their attack; no single best guess.
  `plot_falzon_guarantees.py` reuses LAMa's histogram for gaussian/beta in the "Without Guarantees" panels.
- `print_noise_table.py`: TV-noise tables (15x15, p=1, exact frequencies, one run per cell, noise seed 1337).
- Others: grid-vs-MSE, box plots, histograms, running time, spatial reconstructions (names say what).

## Nov-17 sweep outcome (checked 2026-11-18)

~45 configs reached 1000 consistent DBs (mostly spitz/drink, small p; busstop/shopparis some). The rest:
start-solve failure (fixed above) or stalled. Convergence gets HARDER with more queries (tighter eps): at
20-30% cali/highway/shopparis best restarts were 2.5k-85k pairs outside eps. true_db_in_output is False
everywhere. `bash run_me.sh status` only lists configs for the PROBS/HIGHDIM it is given.

## Rebuttal status (2026-11-17)

Items 1-3 answered from existing data (working_rebuttal.py). Item 4: 15x15 1/2/5% LAMa and 30x30 never run;
50x50 sweep submitted via run_me.sh; baselines at 50x50 for <100% do not exist (baseline code not in this
repo). Item 5 (correlated queries) not implemented: needs a new sampler in sample_percent_responses.
