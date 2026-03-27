// Solver invariant tests for RQ2 mutation detection.
// Each test validates a per-solver property that must hold on clean source
// and is designed to fail when its targeted mutation (B1–B8) is applied.

#[path = "../common/mod.rs"]
mod common;

use spectral_init::operator::CsrOperator;
use spectral_init::solvers::lobpcg::lobpcg_solve;
use spectral_init::lobpcg_sinv_solve;
use spectral_init::rsvd_solve;
use spectral_init::solve_eigenproblem_pub;
use spectral_init::{spectral_init, SpectralInitConfig};

// ─── Test 1: eigenvalue ascending order (targets B1) ─────────────────────────

#[test]
fn test_inv_eigenvalue_ascending_order() {
    let lap_path = common::fixture_path("blobs_50", "comp_b_laplacian.npz");
    let lap = common::load_sparse_csr(&lap_path);
    let ((eigvals, _eigvecs), _level) = solve_eigenproblem_pub(&lap, 2, 42);
    for i in 0..eigvals.len().saturating_sub(1) {
        assert!(
            eigvals[i] <= eigvals[i + 1] + 1e-12,
            "eigenvalues not ascending: λ[{i}]={} > λ[{}]={}",
            eigvals[i],
            i + 1,
            eigvals[i + 1]
        );
    }
}

// ─── Test 2: escalation routing for large n (targets B2) ─────────────────────

#[test]
fn test_inv_escalation_routing_large_n() {
    let lap = common::ring_laplacian(100);

    // Clean path: n=100 < DENSE_N_THRESHOLD=2000 → level 0 (dense EVD).
    let (_, level) = solve_eigenproblem_pub(&lap, 2, 42);
    assert_eq!(
        level, 0,
        "expected dense EVD (level 0) for n=100, got level={level}"
    );

    // Forced path: threshold=0 forces all n away from dense → level ≥ 1.
    // Safe because nextest runs each test in a dedicated process.
    unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "0"); }
    let (_, level_forced) = solve_eigenproblem_pub(&lap, 2, 42);
    unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }
    assert!(
        level_forced >= 1,
        "expected level ≥ 1 with SPECTRAL_DENSE_N_THRESHOLD=0, got {level_forced}"
    );
}

// ─── Test 3: sign convention (targets B3) ────────────────────────────────────

#[test]
fn test_inv_sign_convention() {
    // blobs_50 (n=50) → dense EVD; blobs_connected_2000 (n=2000) → LOBPCG.
    // Both have clear cluster structure → unambiguous argmax per eigenvector column.
    // normalize_signs makes the argmax-absolute-value element positive per column.
    // B3 negates all signs in scaling.rs → argmax element becomes negative.
    // Ring graphs are NOT used here: their degenerate ±1 Fiedler eigenvectors cause
    // the argmax to shift after noise, making the check unreliable.
    for dataset in &["blobs_50", "blobs_connected_2000"] {
        let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
        let graph = common::load_sparse_csr_f32_u32(&graph_path);
        let result = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
            .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));
        let (nrows, ncols) = (result.shape()[0], result.shape()[1]);
        for col in 0..ncols {
            // Check the argmax-absolute-value element is positive (the sign convention).
            // After noisy scaling (noise=0.0001, max=10.0), the argmax is stable for
            // blob cluster data where the top element dominates by a large margin.
            let argmax_val = (0..nrows)
                .map(|row| result[[row, col]])
                .reduce(|a, b| if b.abs() > a.abs() { b } else { a })
                .unwrap_or(0.0f32);
            assert!(
                argmax_val >= 0.0f32,
                "sign convention violated for {dataset} col={col}: \
                 argmax element is {argmax_val}; B3 may have negated all signs"
            );
        }
    }
}

// ─── Test 4: multi-component completeness (targets B4) ───────────────────────

#[test]
fn test_inv_multi_component_completeness() {
    let graph_path = common::fixture_path("disconnected_200", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);
    let result = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
        .unwrap_or_else(|e| panic!("spectral_init failed on disconnected_200: {e}"));
    assert_eq!(result.shape()[0], 200, "expected 200 rows");
    for row in 0..200 {
        let row_norm: f64 = result
            .row(row)
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            row_norm > 1e-9,
            "row {row} is effectively zero (norm={row_norm:.2e}); \
             component likely skipped"
        );
    }
}

// ─── Test 5: eigenvalue non-negative (targets B5) ────────────────────────────

#[test]
fn test_inv_eigenvalue_non_negative() {
    let lap = common::ring_laplacian(500);
    let sqrt_deg = common::ring_sqrt_deg(500);
    let op = CsrOperator(&lap);
    let ((eigvals, _eigvecs), _) =
        lobpcg_solve(&op, 2, 42, false, &sqrt_deg)
            .expect("lobpcg_solve returned None on ring C_500");
    for (i, &lambda) in eigvals.iter().enumerate() {
        assert!(
            lambda >= -1e-9,
            "eigenvalue[{i}] = {lambda:.6e} is negative (< -1e-9)"
        );
    }
}

// ─── Test 6: LOBPCG warm restarts fire (targets B6) ──────────────────────────

#[test]
fn test_inv_lobpcg_convergence_ill_conditioned() {
    // ring C_2000 seed=42: 300 initial iterations are insufficient for the tiny Fiedler
    // eigenvalue; warm restarts fire on clean code (restart_count > 0).
    // B6 changes `for restart in 0..=MAX_WARM_RESTARTS` to `for restart in 0..=0`,
    // forcing restart_count=0 → this assertion fails.
    let lap = common::ring_laplacian(2000);
    let sqrt_deg = common::ring_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
    let (_, restart_count) = result.expect(
        "lobpcg_solve returned None for ring C_2000 seed=42"
    );
    assert!(
        restart_count > 0,
        "ring C_2000 seed=42 must trigger at least one warm restart on clean code; \
         got restart_count={restart_count}; B6 (restart limit=1) may be active"
    );
}

// ─── Test 7: rSVD eigenvector distinctness (targets B7) ──────────────────────

#[test]
fn test_inv_rsvd_eigenvector_distinctness() {
    let lap_path = common::fixture_path("blobs_5000", "comp_b_laplacian.npz");
    let lap = common::load_sparse_csr(&lap_path);
    let (_eigvals, eigvecs) = rsvd_solve(&lap, 2, 42);

    let v0 = eigvecs.column(0);
    let v1 = eigvecs.column(1);
    let dot: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
    let norm0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine_sim = dot.abs() / (norm0 * norm1).max(1e-300);

    assert!(
        cosine_sim < 0.99,
        "rSVD eigenvectors are collinear: |cos|={cosine_sim:.4}; \
         B7 (constant slot) may be active"
    );
}

// ─── Test 8: sinv non-zero result (targets B8) ───────────────────────────────

#[test]
fn test_inv_sinv_non_zero_result() {
    let lap = common::ring_laplacian(2500);
    let sqrt_deg = common::ring_sqrt_deg(2500);
    let (eigvals, eigvecs) = lobpcg_sinv_solve(&lap, 2, 42, &sqrt_deg)
        .expect("lobpcg_sinv_solve returned None on ring C_2500");
    for (i, _lambda) in eigvals.iter().enumerate() {
        let norm: f64 = eigvecs.column(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm > 1e-6,
            "sinv eigenvector[{i}] has near-zero norm={norm:.2e}; \
             B8 (zero result) may be active"
        );
    }
}
