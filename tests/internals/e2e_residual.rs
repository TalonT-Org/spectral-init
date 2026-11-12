//! End-to-end residual quality: the coordinates `spectral_init` returns, de-scaled, must
//! still be eigenvectors of the reference Laplacian.

use crate::{fixtures, internals};
use spectral_init::{SpectralInitConfig, spectral_init};

/// Load graph, run spectral_init, check residuals for a connected dataset.
fn run_e2e_residual_check(dataset: &str, n: usize) {
    let graph_path = fixtures::fixture_path(dataset, "step5a_pruned.npz");
    let graph = fixtures::load_sparse_csr_f32_u32(&graph_path);

    let result = spectral_init(
        graph.view(),
        None,
        &SpectralInitConfig::new(2).with_seed(42),
    )
    .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));
    assert_eq!(result.shape(), &[n, 2]);
    for &v in result.iter() {
        assert!(v.is_finite(), "non-finite in {dataset}: {v}");
    }

    // Load expansion factor (stored as 0-D scalar in the fixture)
    let scaling_path = fixtures::fixture_path(dataset, "comp_f_scaling.npz");
    let expansion: ndarray::Array0<f64> = fixtures::load_dense(&scaling_path, "expansion");
    let expansion_val = expansion.into_scalar();
    assert!(
        expansion_val.is_normal() && expansion_val > 0.0,
        "fixture expansion value must be positive and normal, got {expansion_val}"
    );

    // Load Laplacian and reference eigenvalues
    let laplacian =
        fixtures::load_sparse_csr(&fixtures::fixture_path(dataset, "comp_b_laplacian.npz"));
    let ref_eigenvalues: ndarray::Array1<f64> = fixtures::load_dense(
        &fixtures::fixture_path(dataset, "comp_d_eigensolver.npz"),
        "eigenvalues",
    );

    // De-scale to recover approximate eigenvectors
    let descaled = fixtures::descale_embedding(&result, expansion_val);

    // Check residual for each of the 2 selected eigenvectors.
    // Output column 0 → eigenvalue[1], column 1 → eigenvalue[2] (skip trivial index 0).
    for col in 0..2usize {
        let evec = descaled.column(col);
        let eigval = ref_eigenvalues[col + 1];
        let r = internals::residual_spmv(&laplacian, evec, eigval);
        // Tolerance is 0.005: this bound must hold across the full solver escalation
        // chain (dense EVD → LOBPCG → randomized SVD). Dense EVD residuals are ~1e-8
        // pre-noise; LOBPCG and rSVD paths yield somewhat larger residuals but remain
        // well below this threshold. Gaussian noise (sigma=0.0001) adds at most ~1e-4
        // after de-scaling by expansion (~10); f32→f64 de-scaling contributes ~1e-7.
        // 0.005 is 50× above the expected noise floor, catching genuine solver
        // failures while rejecting the misleadingly loose 0.05 bound.
        let residual_tol = 0.005_f64;
        assert!(
            r < residual_tol,
            "residual quality check failed for {dataset} column {col}: residual={r:.6e} >= {residual_tol}"
        );
    }
}

// ── residual quality tests (7 connected datasets) ────────────────────────────
// blobs_500 and blobs_5000 are excluded: fixture generation confirms they are
// disconnected (4 and 2 components respectively). Their embeddings are produced
// by embed_disconnected() and are not Laplacian eigenvectors of the full graph.

#[test]
fn test_e2e_residual_quality_blobs_connected_200() {
    run_e2e_residual_check("blobs_connected_200", 200);
}

#[test]
fn test_e2e_residual_quality_blobs_connected_2000() {
    run_e2e_residual_check("blobs_connected_2000", 2000);
}

#[test]
fn test_e2e_residual_quality_moons_200() {
    run_e2e_residual_check("moons_200", 200);
}

#[test]
fn test_e2e_residual_quality_circles_300() {
    run_e2e_residual_check("circles_300", 300);
}

#[test]
fn test_e2e_residual_quality_near_dupes_100() {
    run_e2e_residual_check("near_dupes_100", 100);
}
