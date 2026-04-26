use crate::plotting::ReconstructionDataPoint;
use qhull::Qh;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug)]
pub struct PointConvexHull {
    pub vertices: Vec<Vec<f64>>,
    pub faces: Vec<Vec<usize>>,
}

pub fn get_per_point_convex_hulls(data: &[Vec<ReconstructionDataPoint>]) -> Vec<PointConvexHull> {
    if data.is_empty() || data[0].is_empty() {
        return vec![];
    }

    let num_runs = data.len();
    let num_points = data[0].len();
    let mut all_hulls = Vec::with_capacity(num_points);

    for point_idx in 0..num_points {
        let mut point_cloud = Vec::with_capacity(num_runs);

        for run in data.iter() {
            point_cloud.push(run[point_idx].reconstructed_points.clone());
        }

        // Deduplicate identical points
        let mut unique_cloud = Vec::new();
        for p in &point_cloud {
            if !unique_cloud.contains(p) {
                unique_cloud.push(p.clone());
            }
        }

        let dims = point_cloud[0].len();
        if unique_cloud.len() <= dims {
            all_hulls.push(PointConvexHull {
                vertices: unique_cloud,
                faces: vec![],
            });
            continue;
        }

        // ---------------------------------------------------------
        // Attempt 1: Standard Triangulated Hull
        // ---------------------------------------------------------
        let mut qh_result = Qh::builder()
            .compute(true)
            .qhull_args(&["Qt"])
            .expect("EXPECT") // REMOVED "QJ" entirely
            .build_from_iter(unique_cloud.iter().map(|p| p.iter().copied()));

        // ---------------------------------------------------------
        // Attempt 2: Manual Joggle Fallback (Avoids Rust Pointer Panic)
        // ---------------------------------------------------------
        let mut joggled_cloud = unique_cloud.clone();
        if qh_result.is_err() {
            for (i, pt) in joggled_cloud.iter_mut().enumerate() {
                for (j, val) in pt.iter_mut().enumerate() {
                    // Add deterministic, imperceptible noise (1e-8) to break coplanarity
                    let noise = 1e-8 * (((i * 97 + j * 31) % 101) as f64 / 100.0);
                    *val += noise;
                }
            }

            // Re-run with the joggled points
            qh_result = Qh::builder()
                .compute(true)
                .qhull_args(&["Qt"])
                .expect("EXPECT")
                .build_from_iter(joggled_cloud.iter().map(|p| p.iter().copied()));
        }

        match qh_result {
            Ok(qh) => {
                let mut faces = Vec::new();
                let mut unique_vertices = Vec::new();
                let mut index_map = HashMap::new();

                for simplex in qh.simplices() {
                    if let Some(vertices) = simplex.vertices() {
                        let mut face_indices = Vec::new();

                        for v in vertices.iter() {
                            if let Some(qhull_idx) = v.index(&qh).map(|i: usize| i) {
                                let local_idx = *index_map.entry(qhull_idx).or_insert_with(|| {
                                    let new_idx = unique_vertices.len();
                                    // CRITICAL: Pull from the original unique_cloud!
                                    // This ensures the 1e-8 noise does not leak into your plot/MSE calculations.
                                    unique_vertices.push(unique_cloud[qhull_idx].clone());
                                    new_idx
                                });
                                face_indices.push(local_idx);
                            }
                        }
                        faces.push(face_indices);
                    }
                }

                all_hulls.push(PointConvexHull {
                    vertices: unique_vertices,
                    faces,
                });
            }
            Err(_) => {
                // If it STILL fails (e.g., highly degenerate data), safely fallback to the unique points
                all_hulls.push(PointConvexHull {
                    vertices: unique_cloud,
                    faces: vec![],
                });
            }
        }
    }

    all_hulls
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3d_convex_hull_per_point() {
        let num_runs = 5;
        let mut mock_data = vec![Vec::with_capacity(2); num_runs];

        // ---------------------------------------------------------
        // POINT 0: A 3D Tetrahedron with one point trapped inside.
        // We expect the hull to consist ONLY of the 4 outer points.
        // ---------------------------------------------------------
        let p0_recons = [
            vec![0.0, 0.0, 0.0], // Vertex 1
            vec![1.0, 0.0, 0.0], // Vertex 2
            vec![0.0, 1.0, 0.0], // Vertex 3
            vec![0.0, 0.0, 1.0], // Vertex 4
            vec![0.2, 0.2, 0.2], // TRAPPED INSIDE! (Should NOT be in hull)
        ];

        // ---------------------------------------------------------
        // POINT 1: A 3D Square Pyramid.
        // 4 base points, 1 tip point. All 5 should be on the hull.
        // ---------------------------------------------------------
        let p1_recons = [
            vec![10.0, 10.0, 0.0], // Base 1
            vec![11.0, 10.0, 0.0], // Base 2
            vec![11.0, 11.0, 0.0], // Base 3
            vec![10.0, 11.0, 0.0], // Base 4
            vec![10.5, 10.5, 1.0], // Tip
        ];

        // Construct our [Run][Point] vector structure
        for run_idx in 0..num_runs {
            mock_data[run_idx].push(ReconstructionDataPoint {
                true_points: vec![0.0, 0.0, 0.0],
                reconstructed_points: p0_recons[run_idx].clone(),
                unscaled_points: None,
            });

            mock_data[run_idx].push(ReconstructionDataPoint {
                true_points: vec![10.5, 10.5, 0.0],
                reconstructed_points: p1_recons[run_idx].clone(),
                unscaled_points: None,
            });
        }

        let hulls = get_per_point_convex_hulls(&mock_data);

        // We passed 2 point indices per run, so we should get exactly 2 hulls.
        assert_eq!(hulls.len(), 2);

        // --- Verify Point 0 (Tetrahedron) ---
        // 4 outer vertices, so the hull should contain exactly 4 vertices.
        assert_eq!(
            hulls[0].vertices.len(),
            4,
            "Tetrahedron should have 4 vertices on the hull"
        );

        // Ensure the trapped point (0.2, 0.2, 0.2) is cleanly rejected from the vertices array
        let contains_trapped = hulls[0].vertices.iter().any(|v| {
            (v[0] - 0.2).abs() < 1e-6 && (v[1] - 0.2).abs() < 1e-6 && (v[2] - 0.2).abs() < 1e-6
        });
        assert!(
            !contains_trapped,
            "The inner trapped point must not be included in the hull!"
        );

        // A tetrahedron is bounded by exactly 4 triangular faces
        assert_eq!(hulls[0].faces.len(), 4, "Tetrahedron should have 4 faces");

        // --- Verify Point 1 (Square Pyramid) ---
        // 5 vertices form the boundaries, so all 5 should be returned.
        assert_eq!(
            hulls[1].vertices.len(),
            5,
            "Pyramid should have all 5 vertices on the hull"
        );

        // Quickhull strictly triangulates faces in 3D.
        // A square base is composed of 2 triangles. So 4 sides + 2 base triangles = 6 faces.
        assert_eq!(
            hulls[1].faces.len(),
            6,
            "Square pyramid should be triangulated into 6 faces"
        );
    }
}
