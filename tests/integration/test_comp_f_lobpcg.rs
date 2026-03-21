#[path = "../common/mod.rs"]
mod common;

use spectral_init::operator::{CsrOperator, LinearOperator};
use spectral_init::solvers::lobpcg::lobpcg_solve;

/// Verify that lobpcg_solve returns eigenpairs close to Python UMAP's reference
/// eigenvalues and that all residuals ||L*v - λ*v|| / ||v|| < 1e-4.
#[test]
#[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
fn lobpcg_blobs_connected_2000_eigenvalues() {
    let fixture_dir = common::fixture_path("blobs_connected_2000", "");
    let lap_path = fixture_dir.join("comp_b_laplacian.npz");
    let ref_path = fixture_dir.join("comp_d_eigensolver.npz");

    let laplacian = common::load_sparse_csr(&lap_path);
    let op = CsrOperator(&laplacian);
    let n = op.size();

    let result = lobpcg_solve(&op, 2, 42, false);
    assert!(result.is_some(), "lobpcg_solve returned None on blobs_connected_2000 Laplacian");
    let (eigvals, eigvecs) = result.unwrap();

    // Load Python reference eigenvalues
    let ref_eigvals: ndarray::Array1<f64> =
        common::load_dense(&ref_path, "eigenvalues");

    // Compare first n_components+1 eigenvalues (up to sign is not an issue for eigenvalues)
    let k = eigvals.len().min(ref_eigvals.len());
    for i in 0..k {
        assert!(
            (eigvals[i] - ref_eigvals[i]).abs() < 1e-6,
            "eigenvalue[{i}] mismatch: got {}, expected {}",
            eigvals[i],
            ref_eigvals[i]
        );
    }

    // Check residuals ||L*v - λ*v|| / ||v|| < 1e-4 for all eigenpairs
    for i in 0..eigvals.len() {
        let col: ndarray::Array2<f64> =
            eigvecs.column(i).to_owned().insert_axis(ndarray::Axis(1));
        let mut av = ndarray::Array2::zeros((n, 1));
        op.apply(col.view(), &mut av);
        let mut sq = 0.0_f64;
        let mut norm_sq = 0.0_f64;
        let lam = eigvals[i];
        for r in 0..n {
            let diff = av[[r, 0]] - lam * col[[r, 0]];
            sq += diff * diff;
            norm_sq += col[[r, 0]] * col[[r, 0]];
        }
        let residual = sq.sqrt() / norm_sq.sqrt().max(1e-300);
        assert!(
            residual < 1e-4,
            "residual for eigenpair {i}: {residual} >= 1e-4"
        );
    }
}

/// Level 2 (regularized) should also converge on blobs_connected_2000.
#[test]
#[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
fn lobpcg_blobs_connected_2000_level2() {
    let fixture_dir = common::fixture_path("blobs_connected_2000", "");
    let lap_path = fixture_dir.join("comp_b_laplacian.npz");

    let laplacian = common::load_sparse_csr(&lap_path);
    let op = CsrOperator(&laplacian);
    let n = op.size();

    let result = lobpcg_solve(&op, 2, 42, true);
    assert!(
        result.is_some(),
        "lobpcg_solve (regularized) returned None on blobs_connected_2000"
    );
    let (eigvals, eigvecs) = result.unwrap();

    // All residuals must be < 1e-4 for the regularized operator (which shifts by eps=1e-5).
    // For the original Laplacian, the effective residual is still bounded.
    for i in 0..eigvals.len() {
        let col: ndarray::Array2<f64> =
            eigvecs.column(i).to_owned().insert_axis(ndarray::Axis(1));
        let mut av = ndarray::Array2::zeros((n, 1));
        op.apply(col.view(), &mut av);
        let mut sq = 0.0_f64;
        let mut norm_sq = 0.0_f64;
        // For level 2, eigenvalue includes eps shift; remove it for the residual check
        let eps = 1e-5_f64;
        let lam_unshifted = eigvals[i] - eps;
        for r in 0..n {
            let diff = av[[r, 0]] - lam_unshifted * col[[r, 0]];
            sq += diff * diff;
            norm_sq += col[[r, 0]] * col[[r, 0]];
        }
        let residual = sq.sqrt() / norm_sq.sqrt().max(1e-300);
        assert!(
            residual < 1e-4,
            "level2 residual for eigenpair {i}: {residual} >= 1e-4"
        );
    }
}
