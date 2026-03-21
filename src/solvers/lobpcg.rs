use ndarray::Array2;
use crate::operator::LinearOperator;

/// LOBPCG iterative eigensolver (Levels 1 and 2).
/// Level 1: no regularization. Level 2: adds epsilon*I shift.
pub(crate) fn lobpcg_solve<O: LinearOperator>(
    op: &O,
    n_components: usize,
    seed: u64,
    regularize: bool,
) -> Option<(Array2<f64>, Array2<f64>)> {
    todo!("lobpcg_solve: implement linfa-linalg lobpcg path")
}
