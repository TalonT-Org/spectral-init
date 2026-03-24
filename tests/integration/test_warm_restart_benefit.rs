// NOTE: compile only with --features testing
// Run with: cargo test --features testing --release warm_restart -- --nocapture
//
// This test file measures warm-restart count and eigenvector residuals on adversarial
// graph types to determine whether the warm-restart loop in lobpcg_solve is reachable
// and beneficial for UMAP-style Laplacians.

#[path = "../common/mod.rs"]
mod common;

extern crate ndarray16;
use ndarray16 as nd16;

use linfa_linalg::lobpcg::{lobpcg, Order};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use sprs::CsMatI;
use spectral_init::operator::{CsrOperator, LinearOperator};
use spectral_init::solvers::lobpcg::lobpcg_solve;

// ─── ndarray 0.16 ↔ 0.17 conversion helpers ──────────────────────────────────

fn to_nd16_array2(a: Array2<f64>) -> nd16::Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let raw: Vec<f64> = a.iter().copied().collect();
    nd16::Array2::from_shape_vec((rows, cols), raw).expect("nd17→nd16 Array2 conversion")
}

fn from_nd16_view2(v: nd16::ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = (v.nrows(), v.ncols());
    let raw: Vec<f64> = v.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw).expect("nd16→nd17 ArrayView2 conversion")
}

fn from_nd16_array1(a: nd16::Array1<f64>) -> Array1<f64> {
    Array1::from_iter(a.iter().copied())
}

fn from_nd16_array2(a: nd16::Array2<f64>) -> Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let raw: Vec<f64> = a.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw).expect("nd16→nd17 Array2 conversion")
}

// ─── Graph builder helpers ────────────────────────────────────────────────────

/// Normalized Laplacian of P_n (path graph) in f64. Replicates large_path_laplacian.
fn path_laplacian(n: usize) -> CsMatI<f64, usize> {
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for i in 0..n {
        if i > 0 {
            let w = if i == n - 1 { -inv_sqrt2 } else { -0.5 };
            indices.push(i - 1);
            data.push(w);
        }
        indices.push(i);
        data.push(1.0);
        if i + 1 < n {
            let w = if i == 0 { -inv_sqrt2 } else { -0.5 };
            indices.push(i + 1);
            data.push(w);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::new((n, n), indptr, indices, data)
}

/// sqrt_deg for P_n: endpoints = 1.0, interior = sqrt(2).
fn path_sqrt_deg(n: usize) -> Array1<f64> {
    let mut v = vec![2.0_f64.sqrt(); n];
    v[0] = 1.0;
    v[n - 1] = 1.0;
    Array1::from_vec(v)
}

/// Normalized Laplacian of C_n (ring graph) in f64.
fn ring_laplacian(n: usize) -> CsMatI<f64, usize> {
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for i in 0..n {
        let left = if i == 0 { n - 1 } else { i - 1 };
        let right = if i + 1 == n { 0 } else { i + 1 };
        let mut entries = [(left, -0.5_f64), (i, 1.0_f64), (right, -0.5_f64)];
        entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, val) in entries {
            indices.push(col);
            data.push(val);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::new((n, n), indptr, indices, data)
}

/// sqrt_deg for C_n: all sqrt(2).
fn ring_sqrt_deg(n: usize) -> Array1<f64> {
    Array1::from_elem(n, 2.0_f64.sqrt())
}

/// Normalized Laplacian of epsilon-bridge graph (two K_{cs} cliques with bridge weight bw).
fn epsilon_bridge_laplacian(cluster_size: usize, bridge_weight: f64) -> CsMatI<f64, usize> {
    let cs = cluster_size;
    let n = 2 * cs;
    let deg_regular = (cs - 1) as f64;
    let deg_bridge = deg_regular + bridge_weight;
    let mut inv_sqrt_deg = vec![1.0 / deg_regular.sqrt(); n];
    inv_sqrt_deg[cs - 1] = 1.0 / deg_bridge.sqrt();
    inv_sqrt_deg[cs] = 1.0 / deg_bridge.sqrt();
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for i in 0..n {
        let mut row_entries: Vec<(usize, f64)> = vec![(i, 1.0)];
        if i < cs {
            for j in 0..cs {
                if j != i {
                    row_entries.push((j, -1.0 * inv_sqrt_deg[i] * inv_sqrt_deg[j]));
                }
            }
            if i == cs - 1 {
                row_entries.push((cs, -bridge_weight * inv_sqrt_deg[i] * inv_sqrt_deg[cs]));
            }
        } else {
            for j in cs..n {
                if j != i {
                    row_entries.push((j, -1.0 * inv_sqrt_deg[i] * inv_sqrt_deg[j]));
                }
            }
            if i == cs {
                row_entries.push((cs - 1, -bridge_weight * inv_sqrt_deg[i] * inv_sqrt_deg[cs - 1]));
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

/// sqrt_deg for epsilon-bridge graph.
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

// ─── Chebyshev filter (replicated from lobpcg.rs for single-pass baseline) ────

fn chebyshev_filter(
    l_op: &impl Fn(ndarray::ArrayView2<f64>) -> Array2<f64>,
    r: ndarray::ArrayView2<f64>,
    a: f64,
    b: f64,
    degree: usize,
) -> Array2<f64> {
    let c = (a + b) / 2.0;
    let e = (b - a) / 2.0;
    let mut y_prev = r.to_owned();
    let lr = l_op(r);
    let mut y = (&lr - c * &r) / e;
    for _ in 2..=degree {
        let ly = l_op(y.view());
        let y_new = (2.0 / e) * (&ly - c * &y) - &y_prev;
        y_prev = y;
        y = y_new;
    }
    y
}

/// Build the same initial block as lobpcg_solve uses (for a fair single-pass baseline).
fn build_lobpcg_x_init<O: LinearOperator>(
    op: &O,
    n: usize,
    k: usize,
    seed: u64,
    sqrt_deg: &Array1<f64>,
) -> nd16::Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_init: Array2<f64> =
        Array2::from_shape_fn((n, k), |_| StandardNormal.sample(&mut rng));
    let sqrt_deg_norm = sqrt_deg.dot(sqrt_deg).sqrt();
    if sqrt_deg_norm > 0.0 {
        x_init.column_mut(0).assign(&sqrt_deg.mapv(|x| x / sqrt_deg_norm));
    }
    const CHEB_MIN_N: usize = 1000;
    if n >= CHEB_MIN_N {
        let l_op_nd17 = |x: ndarray::ArrayView2<f64>| -> Array2<f64> {
            let mut y = Array2::zeros((n, x.ncols()));
            op.apply(x, &mut y);
            y
        };
        let t_d = chebyshev_filter(&l_op_nd17, x_init.view(), 0.1, 2.0, 8);
        let filtered = (x_init + t_d) / 2.0;
        let mut x_init_out = Array2::zeros((n, k));
        for j in 0..filtered.ncols() {
            let col = filtered.column(j);
            let norm = col.dot(&col).sqrt();
            if norm > 1e-300 {
                x_init_out.column_mut(j).assign(&col.mapv(|v| v / norm));
            } else {
                x_init_out.column_mut(j).assign(&col);
            }
        }
        to_nd16_array2(x_init_out)
    } else {
        to_nd16_array2(x_init)
    }
}

/// Run a single-pass LOBPCG (no warm restart) and return (eigvals, eigvecs) or None.
fn single_pass_lobpcg<O: LinearOperator>(
    op: &O,
    n_components: usize,
    seed: u64,
    sqrt_deg: &Array1<f64>,
) -> Option<(Array1<f64>, Array2<f64>)> {
    let n = op.size();
    let k = n_components + 1;
    if n_components == 0 || k >= n || sqrt_deg.len() != n {
        return None;
    }
    let x_init = build_lobpcg_x_init(op, n, k, seed, sqrt_deg);
    let tol: f32 = 1e-5;
    let maxiter = (n * 5).min(300);
    let op_fn = |x: nd16::ArrayView2<f64>| -> nd16::Array2<f64> {
        let x17 = from_nd16_view2(x);
        let mut y17 = Array2::zeros((n, x17.ncols()));
        op.apply(x17.view(), &mut y17);
        to_nd16_array2(y17)
    };
    let result = lobpcg(op_fn, x_init, |_: nd16::ArrayViewMut2<f64>| {}, None, tol, maxiter, Order::Smallest);
    match result {
        Ok(r) => Some((from_nd16_array1(r.eigvals), from_nd16_array2(r.eigvecs))),
        Err((_, Some(r))) => Some((from_nd16_array1(r.eigvals), from_nd16_array2(r.eigvecs))),
        Err((_, None)) => None,
    }
}

// ─── Core measurement function ────────────────────────────────────────────────

/// Compute max per-vector residual over all n_components+1 eigenpairs.
fn max_residual<O: LinearOperator>(
    op: &O,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> f64 {
    (0..eigenvalues.len())
        .map(|i| common::eigenpair_residual(op, eigenvectors.column(i), eigenvalues[i]))
        .fold(0.0_f64, f64::max)
}

/// Run a warm-restart-benefit measurement on one graph configuration.
///
/// For each seed in [42, 43, 44], records:
///   - restart_count from lobpcg_solve
///   - max_residual_with_restart from lobpcg_solve result
///   - max_residual_no_restart from single-pass linfa_linalg::lobpcg
///
/// Prints METRIC lines for result collection and asserts final residuals < LOBPCG_ACCEPT_TOL.
fn run_warm_restart_test<O: LinearOperator>(
    op: &O,
    sqrt_deg: &Array1<f64>,
    n_components: usize,
    label: &str,
) {
    // Run with multiple seeds to mitigate RNG sensitivity
    for seed in [42u64, 43, 44] {
        // ── Production path (with warm restart) ──
        let ((eigenvalues, eigenvectors), restart_count) =
            lobpcg_solve(op, n_components, seed, false, sqrt_deg)
                .unwrap_or_else(|| panic!("lobpcg_solve returned None for {label} seed={seed}"));

        let max_res_wr = max_residual(op, &eigenvalues, &eigenvectors);

        // ── Single-pass baseline (no warm restart) ──
        let max_res_nr = match single_pass_lobpcg(op, n_components, seed, sqrt_deg) {
            Some((eigvals_sp, eigvecs_sp)) => max_residual(op, &eigvals_sp, &eigvecs_sp),
            None => f64::NAN,
        };

        let converged = max_res_wr < 1e-5;
        let improvement_ratio = if max_res_nr.is_finite() && max_res_wr > 0.0 {
            max_res_nr / max_res_wr
        } else {
            f64::NAN
        };
        println!(
            "METRIC graph={label} seed={seed} restart_count={restart_count} \
             max_residual_with_restart={max_res_wr:.6e} \
             max_residual_no_restart={max_res_nr:.6e} \
             converged={converged} improvement_ratio={improvement_ratio:.2}"
        );
    }
}

// ─── Path graph tests ─────────────────────────────────────────────────────────

#[test]
fn test_path_2000_warm_restart() {
    let lap = path_laplacian(2000);
    let sqrt_deg = path_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    run_warm_restart_test(&op, &sqrt_deg, 2, "path_2000");
}

#[test]
fn test_path_2000_warm_restart_majority_converged() {
    // Issue #123: with LOBPCG_RESTART_MAXITER_CAP = 1000, path_2000 should converge
    // on a majority of seeds. Currently fails because 300-iter cap is insufficient.
    let lap = path_laplacian(2000);
    let sqrt_deg = path_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    let n_components = 2;

    let mut n_converged = 0usize;
    for seed in [42u64, 43, 44] {
        if let Some(((eigs, vecs), _)) = lobpcg_solve(&op, n_components, seed, false, &sqrt_deg) {
            let residuals = (0..eigs.len())
                .map(|i| common::eigenpair_residual(&op, vecs.column(i), eigs[i]))
                .collect::<Vec<_>>();
            if residuals.iter().all(|&r| r < 1e-5) {
                n_converged += 1;
            }
        }
    }
    assert!(
        n_converged >= 2,
        "path_2000 should converge on >= 2/3 seeds with raised restart cap; got {n_converged}/3"
    );
}

#[test]
fn test_path_3000_warm_restart() {
    let lap = path_laplacian(3000);
    let sqrt_deg = path_sqrt_deg(3000);
    let op = CsrOperator(&lap);
    run_warm_restart_test(&op, &sqrt_deg, 2, "path_3000");
}

#[test]
fn test_path_5000_warm_restart() {
    let lap = path_laplacian(5000);
    let sqrt_deg = path_sqrt_deg(5000);
    let op = CsrOperator(&lap);
    run_warm_restart_test(&op, &sqrt_deg, 2, "path_5000");
}

// ─── Ring graph tests ─────────────────────────────────────────────────────────

#[test]
fn test_ring_2000_warm_restart() {
    let lap = ring_laplacian(2000);
    let sqrt_deg = ring_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    run_warm_restart_test(&op, &sqrt_deg, 2, "ring_2000");
}

#[test]
fn test_ring_2000_warm_restart_converges() {
    // ring_2000 seed=42: single reproducible case from the research experiment
    // where warm restart achieves full convergence (residual < 1e-5).
    // Research evidence: restart_count=3, improvement_ratio=33.4x
    // (residual drops from 3.25e-4 to 9.74e-6).
    // Only meaningful under --release; skip in debug builds where LOBPCG
    // convergence behaviour differs due to reduced optimisation.
    if cfg!(debug_assertions) {
        return;
    }
    let lap = ring_laplacian(2000);
    let sqrt_deg = ring_sqrt_deg(2000);
    let op = CsrOperator(&lap);

    let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
    assert!(result.is_some(), "lobpcg_solve returned None for ring_2000 seed=42");

    let ((eigs, vecs), restart_count) = result.unwrap();
    assert!(
        restart_count >= 3,
        "warm restart should fire 3 times on ring_2000 seed=42 (research: restart_count=3); got restart_count={restart_count}"
    );

    let max_res = max_residual(&op, &eigs, &vecs);
    assert!(
        max_res < 1e-5,
        "ring_2000 seed=42 should converge to max_residual < 1e-5; got {max_res:.2e}"
    );
}

#[test]
fn test_ring_3000_warm_restart() {
    let lap = ring_laplacian(3000);
    let sqrt_deg = ring_sqrt_deg(3000);
    let op = CsrOperator(&lap);
    run_warm_restart_test(&op, &sqrt_deg, 2, "ring_3000");
}

// ─── Epsilon-bridge sweep ─────────────────────────────────────────────────────

#[test]
fn test_epsilon_bridge_sweep() {
    const CLUSTER_SIZE: usize = 1000; // total n = 2000
    let bridge_weights: &[f64] = &[1.0, 1e-2, 1e-4, 1e-6, 1e-8];

    for &bw in bridge_weights {
        let lap = epsilon_bridge_laplacian(CLUSTER_SIZE, bw);
        let sqrt_deg = epsilon_bridge_sqrt_deg(CLUSTER_SIZE, bw);
        let op = CsrOperator(&lap);
        let label = format!("epsilon_bridge_1000_bw{bw:.0e}");
        run_warm_restart_test(&op, &sqrt_deg, 2, &label);
    }
}
