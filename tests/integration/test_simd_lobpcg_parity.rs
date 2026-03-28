#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2};
use sprs::CsMatI;
use spectral_init::metrics::{sign_agnostic_max_error, subspace_gram_det_kd, max_eigenpair_residual};
use serde_json::json;
use spectral_init::{
    normalize_signs_pub,
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
const DISCRETE_WEIGHTS: [f64; 3] = [0.25, 2.5e-4, 2.5e-7];
const SEEDS: [u64; 5] = [42, 123, 777, 1337, 9999];

/// Returns the cluster size for each barbell clique.
/// Override at runtime with the `CLUSTER_SIZE` environment variable for
/// dry-run validation (e.g. `CLUSTER_SIZE=50 cargo test …`).
fn cluster_size() -> usize {
    std::env::var("CLUSTER_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000)
}

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

/// Runs both solvers on `laplacian`, computes all 8 experiment metrics,
/// and returns a `serde_json::Value` record with the 12 required fields.
///
/// `bridge_weight` is recorded verbatim; `n` is the per-clique cluster size
/// used to estimate `avg_nnz_per_row` for the DK bound.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn measure_solver_pair(
    laplacian: &CsMatI<f64, usize>,
    n: usize,
    bridge_weight: f64,
    seed: u64,
) -> serde_json::Value {
    const N_LOBPCG_ITER: u32 = 300;

    // --- Solver invocations ---
    let ((eigenvalues_s, eigvecs_s_raw), level_s) =
        solve_eigenproblem_pub(laplacian, 2, seed);
    let ((eigenvalues_x, eigvecs_x_raw), level_x) =
        solve_eigenproblem_simd_pub(laplacian, 2, seed);

    // --- Residuals (pre-normalization, per Davis-Kahan definition) ---
    let residual_scalar = max_eigenpair_residual(laplacian, &eigenvalues_s, &eigvecs_s_raw);
    let residual_simd   = max_eigenpair_residual(laplacian, &eigenvalues_x, &eigvecs_x_raw);

    // --- Sign normalization ---
    let mut eigvecs_s = eigvecs_s_raw;
    normalize_signs_pub(&mut eigvecs_s);
    let mut eigvecs_x = eigvecs_x_raw;
    normalize_signs_pub(&mut eigvecs_x);

    // --- Subspace comparison ---
    let max_subspace_angle_rad = max_subspace_angle(&eigvecs_s, &eigvecs_x);
    // Cross-Gram det: SIMD as "computed", scalar as "reference"
    let subspace_gram_det = subspace_gram_det_kd(eigvecs_x.view(), eigvecs_s.view());

    // --- f32 parity ---
    let f32_s = eigvecs_s.mapv(|v| v as f32);
    let f32_x = eigvecs_x.mapv(|v| v as f32);
    let f32_max_abs_diff = sign_agnostic_max_error(&f32_x, &f32_s);

    // --- DK bound (use approx_gap from scalar solver eigenvalues) ---
    let approx_gap = (eigenvalues_s[1] - eigenvalues_s[0]).max(0.0);
    // avg nnz per row ≈ cluster_size (complete clique of n nodes, +1 diag)
    let avg_nnz: usize = n;
    let dk_bound = dk_bound_predicted(approx_gap, N_LOBPCG_ITER, avg_nnz);
    let dk_amplification_factor = if approx_gap > 0.0 {
        1.0 / approx_gap
    } else {
        f64::INFINITY
    };

    let level_match = level_s == level_x;

    json!({
        "bridge_weight":           bridge_weight,
        "approx_gap":              approx_gap,
        "level_scalar":            level_s,
        "level_simd":              level_x,
        "level_match":             level_match,
        "max_subspace_angle_rad":  max_subspace_angle_rad,
        "subspace_gram_det":       subspace_gram_det,
        "f32_max_abs_diff":        f32_max_abs_diff,
        "residual_scalar":         residual_scalar,
        "residual_simd":           residual_simd,
        "dk_bound_predicted":      dk_bound,
        "dk_amplification_factor": dk_amplification_factor,
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_gap_sweep() {
    let n = cluster_size();
    let results_dir = results_dir();
    let mut records: Vec<serde_json::Value> = Vec::with_capacity(BRIDGE_WEIGHTS.len());

    for &w in BRIDGE_WEIGHTS.iter() {
        let laplacian = large_epsilon_bridge_laplacian(n, w);
        let rec = measure_solver_pair(&laplacian, n, w, 42);

        // Stdout annotation for noteworthy entries
        let angle = rec["max_subspace_angle_rad"].as_f64().unwrap_or(f64::NAN);
        let level_s = rec["level_scalar"].as_u64().unwrap_or(99);
        let level_x = rec["level_simd"].as_u64().unwrap_or(99);
        if level_s != level_x || angle > SUBSPACE_ANGLE_SENSITIVITY_RAD {
            println!(
                "NOTEWORTHY: w={:.2e}  level_scalar={}  level_simd={}  \
                 max_subspace_angle_rad={:.4e}",
                w, level_s, level_x, angle
            );
        }

        records.push(rec);
    }

    let json_str = serde_json::to_string_pretty(&records).expect("serialization failed");
    std::fs::write(results_dir.join("gap_sweep.json"), json_str)
        .expect("cannot write gap_sweep.json");
    println!("Wrote gap_sweep.json ({} entries)", records.len());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_discrete_multi_seed() {
    let n = cluster_size();
    let results_dir = results_dir();
    let mut records: Vec<serde_json::Value> =
        Vec::with_capacity(DISCRETE_WEIGHTS.len() * SEEDS.len());

    for &w in DISCRETE_WEIGHTS.iter() {
        for &seed in SEEDS.iter() {
            let laplacian = large_epsilon_bridge_laplacian(n, w);
            let mut rec = measure_solver_pair(&laplacian, n, w, seed);
            // Add the seed field required by REQ-P3-005
            rec["seed"] = serde_json::json!(seed);
            records.push(rec);
        }
    }

    let json_str = serde_json::to_string_pretty(&records).expect("serialization failed");
    std::fs::write(results_dir.join("discrete_gaps.json"), json_str)
        .expect("cannot write discrete_gaps.json");
    println!("Wrote discrete_gaps.json ({} entries)", records.len());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_solver_level_parity() {
    let n = cluster_size();
    for &w in BRIDGE_WEIGHTS.iter() {
        let laplacian = large_epsilon_bridge_laplacian(n, w);
        let ((eigenvalues_s, _), level_s) = solve_eigenproblem_pub(&laplacian, 2, 42);
        let (_, level_x) = solve_eigenproblem_simd_pub(&laplacian, 2, 42);

        let approx_gap = (eigenvalues_s[1] - eigenvalues_s[0]).max(0.0);
        let is_degenerate = approx_gap < DEGENERATE_GAP_THRESHOLD;

        if level_s != level_x && level_s <= 3 && level_x <= 3 {
            panic!(
                "Solver level parity violation at bridge_weight={:.2e}: \
                 approx_gap={:.6e} (degenerate={}), level_scalar={}, level_simd={}",
                w, approx_gap, is_degenerate, level_s, level_x
            );
        }
    }
}
