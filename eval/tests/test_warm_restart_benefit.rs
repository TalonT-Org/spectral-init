//! Warm-restart benefit of the LOBPCG solver on adversarial graph types.
//!
//! Measures the warm-restart count and the eigenvector residuals to determine whether
//! the warm-restart loop in lobpcg_solve is reachable and beneficial for UMAP-style
//! Laplacians.
//!
//! The no-restart baseline is production's own first pass (`lobpcg_solve_with_restarts`
//! with no warm restarts, Rayleigh–Ritz refinement included), so `improvement_ratio` is
//! exactly 1.00 whenever the first pass converges. A first pass that ends in a numerical
//! error has no baseline (production restarts there) and prints NaN. When no pass yields
//! a result, `lobpcg_solve` returns `None` (production escalates to Level 2) and the row
//! prints `restart_count=none` with NaN residuals; whether that happens on the
//! near-degenerate epsilon-bridge spectra varies with the platform. The values differ
//! from `research/2026-03-22-lobpcg-warm-restart-benefit/`, whose baseline was a hand copy
//! of the start-up code without the refinement.
//!
//! Run with:
//!   cargo test -p spectral-init-eval --release --test test_warm_restart_benefit -- --nocapture

use ndarray::Array1;
use spectral_init::__internal::{
    CsrOperator, LinearOperator, lobpcg_solve, lobpcg_solve_with_restarts,
};
use spectral_init_eval::fixtures::{
    epsilon_bridge_laplacian, epsilon_bridge_sqrt_deg, path_laplacian, path_sqrt_deg,
    ring_laplacian, ring_sqrt_deg,
};
use spectral_init_eval::internals::max_operator_residual;

// ─── Core measurement function ────────────────────────────────────────────────

/// Run a warm-restart-benefit measurement on one graph configuration.
///
/// For each seed in [42, 43, 44], records:
///   - restart_count from lobpcg_solve
///   - max_residual_with_restart from lobpcg_solve result
///   - max_residual_no_restart from the same solver's first pass alone
///
/// Prints METRIC lines for result collection.
fn run_warm_restart_test<O: LinearOperator>(
    op: &O,
    sqrt_deg: &Array1<f64>,
    n_components: usize,
    label: &str,
) {
    // Run with multiple seeds to mitigate RNG sensitivity
    for seed in [42u64, 43, 44] {
        // ── Production path (with warm restart) ──
        let Some(((eigenvalues, eigenvectors), restart_count)) =
            lobpcg_solve(op, n_components, seed, false, sqrt_deg)
        else {
            // No pass produced a result; production escalates to Level 2 here.
            println!(
                "METRIC graph={label} seed={seed} restart_count=none \
                 max_residual_with_restart=NaN max_residual_no_restart=NaN \
                 converged=false improvement_ratio=NaN"
            );
            continue;
        };

        let max_res_wr = max_operator_residual(op, &eigenvalues, &eigenvectors);

        // ── First-pass baseline (no warm restart) ──
        let max_res_nr = lobpcg_solve_with_restarts(op, n_components, seed, false, sqrt_deg, 0)
            .map_or(f64::NAN, |((eigvals_sp, eigvecs_sp), _)| {
                max_operator_residual(op, &eigvals_sp, &eigvecs_sp)
            });

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
