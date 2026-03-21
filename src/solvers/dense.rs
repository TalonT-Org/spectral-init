use ndarray::Array2;
use super::EigenResult;

/// Dense eigendecomposition via faer for small n.
/// Returns (eigenvalues, eigenvectors) sorted ascending by eigenvalue.
pub(crate) fn dense_evd(
    matrix: &Array2<f64>,
    n_components: usize,
) -> EigenResult {
    todo!("dense_evd: implement faer self_adjoint_eigen path")
}
