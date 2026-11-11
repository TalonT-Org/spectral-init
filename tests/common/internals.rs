//! Test support built on `spectral_init::__internal`.

#![allow(dead_code)]

use ndarray::{Array1, Array2, ArrayView1};
use spectral_init::__internal::{
    EigenResult, LinearOperator, SolveReport, SolverParams, solve_eigenproblem,
};
use spectral_init::ComputeMode;
use sprs::CsMatI;

/// Compute the relative residual ||L·v - λ·v|| / ||v|| for an eigenpair.
///
/// Delegates to the library's `eigenpair_residual`.
pub fn residual_spmv(laplacian: &CsMatI<f64, usize>, eigvec: ArrayView1<f64>, eigval: f64) -> f64 {
    spectral_init::__internal::eigenpair_residual(laplacian, &eigvec.to_owned(), eigval)
}

/// Compute the residual norm ||A*v - λ*v|| / ||v|| for one eigenpair of a [`LinearOperator`].
///
/// Unlike [`residual_spmv`], which takes a concrete sparse matrix, this works with any
/// operator, so it can check results produced through the SIMD operator.
pub fn eigenpair_residual<O: LinearOperator>(op: &O, eigvec: ArrayView1<f64>, eigval: f64) -> f64 {
    let n = eigvec.len();
    let col: Array2<f64> = eigvec.to_owned().insert_axis(ndarray::Axis(1));
    let mut av = Array2::zeros((n, 1));
    op.apply(col.view(), &mut av);
    let mut sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for r in 0..n {
        let diff = av[[r, 0]] - eigval * col[[r, 0]];
        sq += diff * diff;
        norm_sq += col[[r, 0]] * col[[r, 0]];
    }
    sq.sqrt() / norm_sq.sqrt().max(1e-300)
}

/// Maximum of [`eigenpair_residual`] over every eigenpair.
pub fn max_operator_residual<O: LinearOperator>(
    op: &O,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> f64 {
    (0..eigenvalues.len())
        .map(|i| eigenpair_residual(op, eigenvectors.column(i), eigenvalues[i]))
        .fold(0.0_f64, f64::max)
}

/// Run the solver escalation chain on a bare Laplacian.
///
/// The trivial-eigenvector hint is the all-ones vector (`sqrt_deg = ones(n)`), which is
/// what a Laplacian loaded without its degree vector calls for.
pub fn solve_with_unit_hint(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    compute_mode: ComputeMode,
    params: &SolverParams,
) -> (EigenResult, SolveReport) {
    let sqrt_deg = Array1::ones(laplacian.rows());
    solve_eigenproblem(
        laplacian,
        n_components,
        seed,
        &sqrt_deg,
        compute_mode,
        params,
    )
    .expect("solve_eigenproblem failed")
}
