#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2};
use spectral_init::operator::{CsrOperator, LinearOperator};

const DATASET: &str = "blobs_connected_200";

#[test]
fn test_eigenpair_residual_from_fixture() {
    let laplacian = common::load_sparse_csr(&common::fixture_path(DATASET, "comp_b_laplacian.npz"));
    let n = laplacian.rows();

    let eigen_path = common::fixture_path(DATASET, "comp_d_eigensolver.npz");
    let eigenvalues: Array1<f64> = common::load_dense(&eigen_path, "eigenvalues");
    let eigenvectors: Array2<f64> = common::load_dense(&eigen_path, "eigenvectors");
    let k = eigenvalues.len();
    assert_eq!(eigenvectors.shape()[0], n);
    assert_eq!(eigenvectors.shape()[1], k);

    // For each eigenpair, assert residual ||L*v - λ*v|| / ||v|| < 1e-10.
    // These are Python reference eigenvectors computed by numpy.linalg.eigh,
    // so near-exact residuals are expected.
    for j in 0..k {
        let lambda = eigenvalues[j];
        let residual = common::residual_spmv(&laplacian, eigenvectors.column(j), lambda);
        assert!(
            residual < 1e-10,
            "eigenpair {j}: residual={residual:.3e} >= 1e-10 (lambda={lambda:.6e})"
        );
    }
}

#[test]
fn test_multi_vector_matches_sequential_from_fixture() {
    let mat = common::load_sparse_csr(&common::fixture_path(DATASET, "comp_b_laplacian.npz"));
    let op = CsrOperator(&mat);
    let n = mat.rows();

    // Load eigenvectors as input matrix
    let eigenvectors: Array2<f64> = common::load_dense(
        &common::fixture_path(DATASET, "comp_d_eigensolver.npz"),
        "eigenvectors",
    );
    let k = eigenvectors.shape()[1];

    // Apply CsrOperator to the full (n, k) matrix at once.
    let mut y_block = Array2::<f64>::zeros((n, k));
    op.apply(eigenvectors.view(), &mut y_block);

    // Apply column-by-column and compare.
    for j in 0..k {
        let x_col = eigenvectors
            .column(j)
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        let mut y_col = Array2::<f64>::zeros((n, 1));
        op.apply(x_col.view(), &mut y_col);
        for i in 0..n {
            assert!(
                (y_block[[i, j]] - y_col[[i, 0]]).abs() < 1e-14,
                "block vs sequential mismatch at row={i} col={j}: \
                 block={} seq={}",
                y_block[[i, j]],
                y_col[[i, 0]]
            );
        }
    }
}
