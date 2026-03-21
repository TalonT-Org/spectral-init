mod dense;
mod lobpcg;
mod rsvd;

// pub (not pub(crate)) so lib.rs can re-export it for integration tests.
pub use dense::dense_evd;

use ndarray::{Array1, Array2};
pub(crate) use crate::operator::LinearOperator;
use sprs::CsMatI;
use crate::SpectralError;
use crate::operator::CsrOperator;

/// Eigendecomposition result: (eigenvalues shape [k], eigenvectors shape [n, k]).
pub(crate) type EigenResult = (Array1<f64>, Array2<f64>);

/// Solver escalation chain. Tries LOBPCG → LOBPCG+reg → rSVD → dense EVD.
/// Returns `Err(SpectralError::ConvergenceFailure)` if all solvers are exhausted.
/// The `CsrOperator` for iterative solvers is constructed internally from `laplacian`.
pub(crate) fn solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> Result<EigenResult, SpectralError> {
    let op = CsrOperator(laplacian);

    // Level 1: LOBPCG without regularization
    if let Some(result) = lobpcg::lobpcg_solve(&op, n_components, seed, false) {
        return Ok(result);
    }

    // Level 2: LOBPCG with epsilon*I regularization
    if let Some(result) = lobpcg::lobpcg_solve(&op, n_components, seed, true) {
        return Ok(result);
    }

    // Level 3: rSVD (P2-11)
    todo!("Level 3: rsvd_solve — implement in P2-11")
}
