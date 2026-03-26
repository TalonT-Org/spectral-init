//! RQ4: Property-based tests for spectral_init mathematical invariants.
//!
//! Uses proptest to verify invariants hold across random graph sizes n in [50, 200].
//! This test target requires `--features testing`.

#[path = "../common/mod.rs"]
mod common;

use proptest::prelude::*;
use spectral_init::{spectral_init, SpectralInitConfig};

// ─── Eigenpair invariant tests ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 64, ..Default::default() })]

    #[test]
    fn proptest_eigenvalues_sorted(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for i in 0..eigs.len() - 1 {
            prop_assert!(
                eigs[i] <= eigs[i + 1] + 1e-12,
                "eigenvalues not sorted: λ[{}]={} > λ[{}]={}",
                i, eigs[i], i + 1, eigs[i + 1]
            );
        }
    }

    #[test]
    fn proptest_eigenvalues_nonneg(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for (i, &e) in eigs.iter().enumerate() {
            prop_assert!(e >= -1e-12, "eigenvalue λ[{}]={} is negative", i, e);
        }
    }

    #[test]
    fn proptest_eigenvalues_leq_2(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for (i, &e) in eigs.iter().enumerate() {
            prop_assert!(
                e <= 2.0 + 1e-12,
                "eigenvalue λ[{}]={} exceeds spectral bound 2.0", i, e
            );
        }
    }

    #[test]
    fn proptest_trivial_null_space(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, _vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        prop_assert!(
            eigs[0].abs() < 1e-6,
            "smallest eigenvalue λ₀={} not near zero for connected ring({})",
            eigs[0], n
        );
    }

    #[test]
    fn proptest_residual_leq_1e5(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((eigs, vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        for i in 0..eigs.len() {
            let col = vecs.column(i);
            let r = common::residual_spmv(&laplacian, col, eigs[i]);
            prop_assert!(
                r < 1e-5,
                "residual for eigenpair {} is {} (threshold 1e-5), n={}",
                i, r, n
            );
        }
    }

    #[test]
    fn proptest_orthogonality(n in 50usize..=200) {
        let laplacian = common::ring_laplacian(n);
        let ((_eigs, vecs), _level) =
            spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
        // vecs is (n, 3) — compute V^T * V which should be I_3
        let vtv = vecs.t().dot(&vecs);
        let k = vtv.nrows();
        let mut frob_sq = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let target = if i == j { 1.0 } else { 0.0 };
                frob_sq += (vtv[[i, j]] - target).powi(2);
            }
        }
        let frob = frob_sq.sqrt();
        prop_assert!(
            frob < 1e-8,
            "||V^T V - I||_F = {} exceeds 1e-8, n={}", frob, n
        );
    }
}

// ─── Output invariant tests ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 64, ..Default::default() })]

    #[test]
    fn proptest_finite_output(n in 50usize..=200) {
        let graph = common::make_ring(n as u32);
        let coords = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
            .expect("spectral_init failed");
        for &v in coords.iter() {
            prop_assert!(v.is_finite(), "non-finite value {} in output, n={}", v, n);
        }
    }

    #[test]
    fn proptest_output_variance(n in 50usize..=200) {
        let graph = common::make_ring(n as u32);
        let coords = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
            .expect("spectral_init failed");
        for col_idx in 0..coords.ncols() {
            let col = coords.column(col_idx);
            let mean = col.mean().unwrap();
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / col.len() as f32;
            prop_assert!(
                variance > 1e-10,
                "column {} has degenerate variance {}, n={}", col_idx, variance, n
            );
        }
    }

    #[test]
    fn proptest_subspace_stable(n in 50usize..=200) {
        let graph = common::make_ring(n as u32);
        let out1 = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
            .expect("spectral_init run 1 failed");
        let out2 = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
            .expect("spectral_init run 2 failed");
        prop_assert!(
            out1 == out2,
            "outputs differ for same seed=42 at n={}", n
        );
    }
}
