mod dense;
mod lobpcg;
mod rsvd;

use ndarray::Array2;
use crate::operator::LinearOperator;
use sprs::CsMatI;

/// Eigendecomposition result: (eigenvalues shape [k], eigenvectors shape [n, k]).
pub(crate) type EigenResult = (Array2<f64>, Array2<f64>);

/// Solver escalation chain. Tries LOBPCG → LOBPCG+reg → rSVD → dense EVD.
/// Panics only if the graph is too large for dense EVD and all iterative solvers
/// exhausted — indicating a genuine algorithmic bug, not a degraded experience.
pub(crate) fn solve<O: LinearOperator>(
    op: &O,
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> EigenResult {
    todo!("solve: implement solver escalation chain")
}
