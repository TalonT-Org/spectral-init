use ndarray::{Array2, ArrayView2};

/// Abstraction over a linear operator A: R^n → R^n.
/// Allows solvers to accept sparse or dense matrices uniformly.
pub(crate) trait LinearOperator {
    fn apply(&self, x: ArrayView2<f64>) -> Array2<f64>;
    fn nrows(&self) -> usize;
}

/// LinearOperator backed by a sprs CSR matrix (f64).
pub(crate) struct CsrOperator(pub sprs::CsMatI<f64, usize>);

impl LinearOperator for CsrOperator {
    fn apply(&self, _x: ArrayView2<f64>) -> Array2<f64> {
        todo!("CsrOperator::apply: sparse-dense multiply")
    }

    fn nrows(&self) -> usize {
        self.0.rows()
    }
}
