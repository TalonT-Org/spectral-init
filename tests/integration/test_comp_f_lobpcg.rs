#[path = "../common/mod.rs"]
mod common;

use spectral_init::operator::CsrOperator;
use spectral_init::solvers::lobpcg::{lobpcg_solve, REGULARIZATION_EPS};

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

    let deg_path = fixture_dir.join("comp_a_degrees.npz");
    let sqrt_deg: ndarray::Array1<f64> = common::load_dense(&deg_path, "sqrt_deg");
    assert_eq!(sqrt_deg.len(), laplacian.rows(), "sqrt_deg length must match Laplacian dimension");

    let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
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
        let residual = common::eigenpair_residual(&op, eigvecs.column(i), eigvals[i]);
        assert!(residual < 1e-4, "residual for eigenpair {i}: {residual} >= 1e-4");
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

    let deg_path = fixture_dir.join("comp_a_degrees.npz");
    let sqrt_deg: ndarray::Array1<f64> = common::load_dense(&deg_path, "sqrt_deg");
    assert_eq!(sqrt_deg.len(), laplacian.rows(), "sqrt_deg length must match Laplacian dimension");

    let result = lobpcg_solve(&op, 2, 42, true, &sqrt_deg);
    assert!(
        result.is_some(),
        "lobpcg_solve (regularized) returned None on blobs_connected_2000"
    );
    let (eigvals, eigvecs) = result.unwrap();

    // All residuals must be < 1e-4 against the original (unshifted) Laplacian.
    // Level 2 eigenvalues include the REGULARIZATION_EPS shift; remove it before checking.
    for i in 0..eigvals.len() {
        let residual =
            common::eigenpair_residual(&op, eigvecs.column(i), eigvals[i] - REGULARIZATION_EPS);
        assert!(residual < 1e-4, "level2 residual for eigenpair {i}: {residual} >= 1e-4");
    }
}
