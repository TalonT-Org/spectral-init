#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2};
use sprs::CsMatI;
use spectral_init::metrics::sign_agnostic_max_error;
use serde_json::json;
use spectral_init::{
    normalize_signs_pub,
    scale_and_add_noise_pub,
    solve_eigenproblem_pub,
    DEGENERATE_GAP_THRESHOLD,
};
use std::fs;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::solve_eigenproblem_simd_pub;

const SIMD_PARITY_F32_THRESHOLD: f64 = 4.0 * f32::EPSILON as f64;
const SUBSPACE_ANGLE_SENSITIVITY_RAD: f64 = 1e-6;
const SUBSPACE_ANGLE_FAILURE_RAD: f64 = 0.01;
const BRIDGE_WEIGHTS: [f64; 7] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];

fn results_dir() -> std::path::PathBuf {
    let p = match std::env::var("RESULTS_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("research/2026-03-28-simd-lobpcg-near-degenerate/results"),
    };
    std::fs::create_dir_all(&p).expect("results dir");
    p
}

fn large_epsilon_bridge_laplacian(cluster_size: usize, bridge_weight: f64) -> CsMatI<f64, usize> {
    let cs = cluster_size;
    let n = 2 * cs;
    let deg_regular = (cs - 1) as f64;
    let deg_bridge = deg_regular + bridge_weight;

    // inv_sqrt_deg[i] = 1 / sqrt(degree[i])
    let mut inv_sqrt_deg = vec![1.0 / deg_regular.sqrt(); n];
    inv_sqrt_deg[cs - 1] = 1.0 / deg_bridge.sqrt();
    inv_sqrt_deg[cs] = 1.0 / deg_bridge.sqrt();

    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    for i in 0..n {
        // Collect all (col, weight) pairs for this row, then sort by column.
        let mut row_entries: Vec<(usize, f64)> = Vec::new();
        // Diagonal
        row_entries.push((i, 1.0));
        // Off-diagonal clique/bridge edges
        if i < cs {
            // First clique: connect to all other first-clique nodes
            for j in 0..cs {
                if j != i {
                    let w_off = -1.0 * inv_sqrt_deg[i] * inv_sqrt_deg[j];
                    row_entries.push((j, w_off));
                }
            }
            // Bridge edge: node cs-1 connects to node cs
            if i == cs - 1 {
                let w_bridge = -bridge_weight * inv_sqrt_deg[i] * inv_sqrt_deg[cs];
                row_entries.push((cs, w_bridge));
            }
        } else {
            // Second clique: connect to all other second-clique nodes
            for j in cs..n {
                if j != i {
                    let w_off = -1.0 * inv_sqrt_deg[i] * inv_sqrt_deg[j];
                    row_entries.push((j, w_off));
                }
            }
            // Bridge edge: node cs connects to node cs-1
            if i == cs {
                let w_bridge = -bridge_weight * inv_sqrt_deg[i] * inv_sqrt_deg[cs - 1];
                row_entries.push((cs - 1, w_bridge));
            }
        }
        row_entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, val) in row_entries {
            indices.push(col);
            data.push(val);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::new((n, n), indptr, indices, data)
}

fn epsilon_bridge_sqrt_deg(cluster_size: usize, bridge_weight: f64) -> Array1<f64> {
    let cs = cluster_size;
    let n = 2 * cs;
    let deg_regular = ((cs - 1) as f64).sqrt();
    let deg_bridge = ((cs - 1) as f64 + bridge_weight).sqrt();
    let mut v = vec![deg_regular; n];
    v[cs - 1] = deg_bridge;
    v[cs] = deg_bridge;
    Array1::from_vec(v)
}

/// Compute the maximum principal angle between two column subspaces.
/// Returns arccos(min singular value of A^T * B) in radians.
fn max_subspace_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // M = A^T * B  (k x k matrix, k=3 in our case)
    let m = a.t().dot(b);
    // Compute singular values via eigenvalues of M^T * M
    let mtm = m.t().dot(&m);
    let k = mtm.nrows();
    let faer_mat = faer::Mat::<f64>::from_fn(k, k, |i, j| mtm[[i, j]]);
    let evd = faer_mat
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("eigendecomposition of small k×k matrix");
    let s = evd.S();
    // Singular values of M are sqrt of eigenvalues of M^T*M
    let min_sv_sq = (0..k)
        .map(|i| s.column_vector().iter().nth(i).copied().unwrap_or(0.0).max(0.0))
        .fold(f64::INFINITY, f64::min);
    let min_sv = min_sv_sq.sqrt().min(1.0); // clamp to [0, 1] for arccos
    min_sv.acos()
}

fn dk_bound_predicted(gap: f64, n_iterations: u32, avg_nnz_per_row: usize) -> f64 {
    if gap == 0.0 {
        return f64::INFINITY;
    }
    n_iterations as f64 * avg_nnz_per_row as f64 * f64::EPSILON / gap
}

#[test]
fn test_gap_sweep() {
    todo!()
}

#[test]
fn test_discrete_multi_seed() {
    todo!()
}

#[test]
fn test_solver_level_parity() {
    todo!()
}
