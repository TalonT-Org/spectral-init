use ndarray::Array2;

/// Dense eigendecomposition via faer for small n.
/// Returns (eigenvalues, eigenvectors) sorted ascending by eigenvalue.
pub(crate) fn dense_evd(
    matrix: &Array2<f64>,
    n_components: usize,
) -> (Array2<f64>, Array2<f64>) {
    todo!("dense_evd: implement faer self_adjoint_eigen path")
}
