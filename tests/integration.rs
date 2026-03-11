use log::{debug, info};
use std::collections::HashMap;

// Import solver engine from main project:
use frequency_analysis_limits::LAMA::solver::SolverEngine;

/// Helper functions for LAMa tests.
/// Calculates the frequency (number of range queries covering a point) in a 2D grid.
/// In a perfect square grid of size `max_val` x `max_val`, a query [x1, x2] x [y1, y2]
/// covers (x, y) if x1 <= x <= x2 and y1 <= y <= y2.
/// The number of such intervals for 1D is (coordinate + 1) * (max_val - coordinate).
/// This all assumes a uniform response distribution
pub fn calculate_query_coverage(x: u32, y: u32, max_val: u32) -> u32 {
    let x_cov = (x + 1) * (max_val - x);
    let y_cov = (y + 1) * (max_val - y);
    x_cov * y_cov
}

/// Generates a perfectly uniform 2D database grid.
pub fn generate_grid(size: u32) -> Vec<(u32, u32)> {
    let mut grid = Vec::with_capacity((size * size) as usize);
    for x in 0..size {
        for y in 0..size {
            grid.push((x, y));
        }
    }

    grid
}

#[test]
fn test_full_reconstruction_4x4() {
    // Initialize the logger. Run `RUST_LOG=info cargo test -- --nocapture` to see output.
    let _ = env_logger::builder().is_test(true).try_init();

    info!("Starting 4x4 grid E2E test");

    let grid_size = 4;
    let records = generate_grid(grid_size);
    let total_items = records.len() as u32; // 16 items

    info!(
        "Generated {} records for a {}x{} grid",
        total_items, grid_size, grid_size
    );

    // Step 1: Compute query coverage (frequency) for every true domain value. This assumes a
    // uniform query distribution. Really just a map of observed frequency->record ID
    let mut coverage_to_points: HashMap<u32, Vec<i64>> = HashMap::new();
    for x in 0..grid_size {
        for y in 0..grid_size {
            let cov = calculate_query_coverage(x, y, grid_size);
            // Flatten to unique hashabl/record ID for each input. (maybe I should make a reverse map)
            let flattened_val = (x * grid_size + y) as i64;
            coverage_to_points
                .entry(cov)
                .or_default()
                .push(flattened_val);
        }
    }

    // Step 2: Build the Frequency-Pair Matching Table (t1_matches)
    // Map each encrypted record ID to the set of domain values that share its exact frequency.
    let mut t1_matches: HashMap<u32, Vec<i64>> = HashMap::new();
    let mut record_ids = Vec::with_capacity(total_items as usize);

    for (i, &(x, y)) in records.iter().enumerate() {
        let rec_id = i as u32;
        record_ids.push(rec_id);

        // Simulate observing the query frequency for this specific record
        let observed_frequency = calculate_query_coverage(x, y, grid_size);

        // The record's candidate values are all domain points with that same frequency
        let candidates = coverage_to_points.get(&observed_frequency).unwrap().clone();

        debug!(
            "Record {:02} (Freq: {}) has {} candidate(s): {:?}",
            rec_id,
            observed_frequency,
            candidates.len(),
            candidates
        );

        t1_matches.insert(rec_id, candidates);
    }

    info!("Constructed t1_matches frequency table mapping records to candidate domains.");

    // Step 3: Solver Integration
    info!("Initializing Z3 Solver Engine");
    let solver = SolverEngine::new();

    info!("Running constraint satisfaction reconstruction...");
    let solution = solver
        .reconstruct(&record_ids, &t1_matches)
        .expect("Z3 failed to find a valid reconstruction (UNSAT)!");

    info!("Reconstruction successful (SAT). Validating constraints...");

    // Step 4: Validation
    assert_eq!(
        solution.len(),
        total_items as usize,
        "Solution must map every record."
    );

    // Check that we used exactly 16 unique values (AllDifferent constraint held)
    let mut assigned_values: Vec<i64> = solution.iter().map(|&(_, val)| val).collect();
    assigned_values.sort_unstable();
    assigned_values.dedup();
    assert_eq!(
        assigned_values.len(),
        total_items as usize,
        "Solution must be a 1-to-1 permutation (no duplicate values)."
    );

    // Check that every assigned value respects the original frequency bounds
    for &(rec_id, assigned_val) in &solution {
        let allowed_candidates = t1_matches.get(&rec_id).unwrap();
        assert!(
            allowed_candidates.contains(&assigned_val),
            "Record {} assigned to {}, which violates its frequency class {:?}",
            rec_id,
            assigned_val,
            allowed_candidates
        );
    }

    info!("E2E test completed successfully! The solver respected all frequency symmetries.");
}
