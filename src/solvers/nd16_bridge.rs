//! Bridge to `linfa-linalg`, whose LOBPCG is built on ndarray 0.16 while this crate uses
//! ndarray 0.17.
//!
//! This module is the crate's only user of the `ndarray16` and `linfa_linalg` crates: the
//! LOBPCG solvers call linfa through [`lobpcg`] and see its array types only through the
//! [`nd16`] re-export, in the operator closures they hand over.

pub(super) use linfa_linalg::lobpcg::Order;
pub(super) use ndarray16 as nd16;

use super::EigenResult;
use ndarray::{Array1, Array2};

// ─── ndarray 0.16 ↔ 0.17 conversion helpers ──────────────────────────────────
//
// Each helper rebuilds an array from exactly `rows * cols` elements read off the source,
// so the `from_shape_vec` calls cannot fail.

/// Convert an owned ndarray 0.17 `Array2<f64>` to ndarray 0.16 `Array2<f64>`.
pub(super) fn to_nd16_array2(a: Array2<f64>) -> nd16::Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    // iter() respects logical shape regardless of memory layout or slicing.
    let raw: Vec<f64> = a.iter().copied().collect();
    nd16::Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd17→nd16 Array2")
}

/// Convert an ndarray 0.16 `ArrayView2<f64>` to an owned ndarray 0.17 `Array2<f64>`.
///
/// Uses `iter()` for the same reason as `from_nd16_array1`: slice_move can leave backing
/// Vecs larger than the logical extent, and `into_raw_vec()` would return all elements.
pub(super) fn from_nd16_view2(v: nd16::ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = (v.nrows(), v.ncols());
    let raw: Vec<f64> = v.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd16→nd17 ArrayView2")
}

/// Convert an owned ndarray 0.16 `Array1<f64>` to ndarray 0.17 `Array1<f64>`.
///
/// Uses `iter()` instead of `into_raw_vec()` because linfa-linalg can return sliced arrays
/// whose backing Vec is longer than the logical length. `iter()` always respects logical size.
fn from_nd16_array1(a: nd16::Array1<f64>) -> Array1<f64> {
    Array1::from_iter(a.iter().copied())
}

// ─── LOBPCG ───────────────────────────────────────────────────────────────────

/// Outcome of one LOBPCG run, in this crate's array types.
pub(super) enum LobpcgOutcome {
    /// linfa returned `Ok`: every residual norm met the tolerance, or the iteration
    /// budget ran out. Callers must still check the residuals.
    Converged(EigenResult),
    /// The iteration stopped on a numerical error after at least one iterate. Carries
    /// the best eigenpairs reached and their residual norms, one per eigenpair.
    Partial {
        eigenpairs: EigenResult,
        rnorm: Vec<f64>,
    },
    /// The iteration failed without producing eigenpairs.
    Failed,
}

/// Runs linfa-linalg's LOBPCG without a preconditioner or constraints.
///
/// `operator` maps a block of column vectors to the operator applied to it, in ndarray
/// 0.16 types; `x_init` is the starting block; `order` selects the end of the spectrum.
pub(super) fn lobpcg<F>(
    operator: F,
    x_init: Array2<f64>,
    tol: f32,
    maxiter: usize,
    order: Order,
) -> LobpcgOutcome
where
    F: Fn(nd16::ArrayView2<f64>) -> nd16::Array2<f64>,
{
    let eigenpairs = |eigvals: nd16::Array1<f64>, eigvecs: nd16::Array2<f64>| -> EigenResult {
        (from_nd16_array1(eigvals), from_nd16_view2(eigvecs.view()))
    };
    let result = linfa_linalg::lobpcg::lobpcg(
        operator,
        to_nd16_array2(x_init),
        |_: nd16::ArrayViewMut2<f64>| {},
        None,
        tol,
        maxiter,
        order,
    );
    match result {
        Ok(r) => LobpcgOutcome::Converged(eigenpairs(r.eigvals, r.eigvecs)),
        Err((e, Some(r))) => {
            log::debug!(
                "[spectral] LOBPCG stopped on a numerical error ({e}); keeping the best iterate"
            );
            LobpcgOutcome::Partial {
                eigenpairs: eigenpairs(r.eigvals, r.eigvecs),
                rnorm: r.rnorm,
            }
        }
        Err((e, None)) => {
            log::debug!("[spectral] LOBPCG failed without an iterate ({e})");
            LobpcgOutcome::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_conversions_round_trip() {
        let a = Array2::from_shape_fn((3, 2), |(i, j)| (i * 2 + j) as f64);
        let a16 = to_nd16_array2(a.clone());
        assert_eq!(a16.dim(), (3, 2));
        assert_eq!(from_nd16_view2(a16.view()), a);
        assert_eq!(
            from_nd16_array1(nd16::Array1::from_vec(vec![1.0, 2.0])),
            Array1::from_vec(vec![1.0, 2.0])
        );
    }

    #[test]
    fn conversions_respect_the_logical_layout_of_sliced_arrays() {
        // A column-major 0.17 array and a sliced 0.16 array: both must convert by
        // logical position, not by backing storage.
        let fortran =
            Array2::from_shape_vec(ndarray::ShapeBuilder::f((2, 2)), vec![1.0, 3.0, 2.0, 4.0])
                .unwrap();
        let converted = to_nd16_array2(fortran);
        assert_eq!(converted, nd16::arr2(&[[1.0, 2.0], [3.0, 4.0]]));

        let wide = nd16::arr2(&[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]);
        let sliced = wide.slice_move(nd16::s![.., ..2]);
        assert_eq!(
            from_nd16_view2(sliced.view()),
            ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]])
        );
    }

    #[test]
    fn lobpcg_finds_the_smallest_eigenvalues_of_a_diagonal_matrix() {
        // diag(1, 2, ..., 8): the two smallest eigenvalues are 1 and 2.
        let diagonal: Vec<f64> = (1..=8).map(f64::from).collect();
        let multiply = |x: nd16::ArrayView2<f64>| {
            let mut y = x.to_owned();
            for (mut row, &d) in y.rows_mut().into_iter().zip(&diagonal) {
                row *= d;
            }
            y
        };
        let x_init = Array2::from_shape_fn((8, 2), |(i, j)| 1.0 / (1.0 + (i + 3 * j) as f64));
        match lobpcg(multiply, x_init, 1e-8, 200, Order::Smallest) {
            LobpcgOutcome::Converged((eigenvalues, eigenvectors)) => {
                assert_eq!(eigenvectors.dim(), (8, 2));
                assert!((eigenvalues[0] - 1.0).abs() < 1e-6);
                assert!((eigenvalues[1] - 2.0).abs() < 1e-6);
            }
            LobpcgOutcome::Partial { rnorm, .. } => panic!("did not converge: rnorm {rnorm:?}"),
            LobpcgOutcome::Failed => panic!("LOBPCG failed"),
        }
    }

    #[test]
    fn lobpcg_reports_failure_when_the_block_is_wider_than_the_problem() {
        // linfa rejects a starting block with more columns than rows before iterating.
        let x_init = Array2::ones((2, 3));
        assert!(matches!(
            lobpcg(|x| x.to_owned(), x_init, 1e-8, 10, Order::Smallest),
            LobpcgOutcome::Failed
        ));
    }
}
