use ndarray::Array2;
use sprs::CsMatI;

/// Randomized SVD eigensolver via the 2I − L trick (Level 3).
pub(crate) fn rsvd_solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    todo!("rsvd_solve: implement Halko-Tropp randomized SVD")
}
