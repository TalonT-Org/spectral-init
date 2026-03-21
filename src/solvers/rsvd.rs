use sprs::CsMatI;
use super::EigenResult;

/// Randomized SVD eigensolver via the 2I − L trick (Level 3).
pub(crate) fn rsvd_solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> EigenResult {
    todo!("rsvd_solve: implement Halko-Tropp randomized SVD")
}
