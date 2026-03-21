use ndarray::Array2;

/// Selects `n_components` eigenvectors, skipping the trivial zero-eigenvalue
/// eigenvector(s). Input eigenvalues are sorted ascending.
pub(crate) fn select_eigenvectors(
    eigenvalues: &[f64],
    eigenvectors: &Array2<f64>,
    n_components: usize,
) -> Array2<f64> {
    todo!("select_eigenvectors: skip trivial, take n_components columns")
}
