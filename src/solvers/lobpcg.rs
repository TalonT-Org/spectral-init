// linfa-linalg 0.2 uses ndarray 0.16, while this crate uses ndarray 0.17.
// We alias ndarray 0.16 as `nd16` for use at the linfa-linalg boundary only.
// All internal types and return values use ndarray 0.17 (the `ndarray` crate).
extern crate ndarray16;
use ndarray16 as nd16;

use linfa_linalg::lobpcg::{lobpcg, Order};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use super::{EigenResult, LinearOperator};

// ─── ndarray 0.16 ↔ 0.17 conversion helpers ──────────────────────────────────

/// Convert an owned ndarray 0.17 Array2<f64> to ndarray 0.16 Array2<f64>.
fn to_nd16_array2(a: Array2<f64>) -> nd16::Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    // iter() respects logical shape regardless of memory layout or slicing.
    let raw: Vec<f64> = a.iter().copied().collect();
    nd16::Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd17→nd16 Array2")
}

/// Convert an ndarray 0.16 ArrayView2<f64> to an owned ndarray 0.17 Array2<f64>.
fn from_nd16_view2(v: nd16::ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = (v.nrows(), v.ncols());
    let raw: Vec<f64> = v.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd16→nd17 ArrayView2")
}

/// Convert an owned ndarray 0.16 Array1<f64> to ndarray 0.17 Array1<f64>.
///
/// Uses `iter()` instead of `into_raw_vec()` because linfa-linalg can return sliced arrays
/// whose backing Vec is longer than the logical length. `iter()` always respects logical size.
fn from_nd16_array1(a: nd16::Array1<f64>) -> Array1<f64> {
    Array1::from_iter(a.iter().copied())
}

/// Convert an owned ndarray 0.16 Array2<f64> to ndarray 0.17 Array2<f64>.
///
/// Uses `iter()` for the same reason as `from_nd16_array1`: slice_move can leave backing
/// Vecs larger than the logical extent, and `into_raw_vec()` would return all elements.
fn from_nd16_array2(a: nd16::Array2<f64>) -> Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let raw: Vec<f64> = a.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd16→nd17 Array2")
}

// ─── Constants ────────────────────────────────────────────────────────────────

/// Residual threshold used to accept partially-converged LOBPCG results.
const LOBPCG_ACCEPT_TOL: f64 = 1e-4;

/// Epsilon shift applied to the operator in Level 2 (regularized) LOBPCG.
pub const REGULARIZATION_EPS: f64 = 1e-5;

// ─── LOBPCG solver ────────────────────────────────────────────────────────────

/// LOBPCG iterative eigensolver (Levels 1 and 2).
/// Level 1: no regularization. Level 2: adds epsilon*I shift.
pub fn lobpcg_solve<O: LinearOperator>(
    op: &O,
    n_components: usize,
    seed: u64,
    regularize: bool,
) -> Option<EigenResult> {
    let n = op.size();
    let k = n_components + 1; // +1 to include trivial eigenvector slot

    // Validate preconditions: need at least 1 component and block size < matrix size.
    if n_components == 0 || k >= n {
        return None;
    }

    // Build random initial block [n, k] in ndarray 0.17, then convert to 0.16 for lobpcg.
    let mut rng = StdRng::seed_from_u64(seed);
    let x_init_17: Array2<f64> =
        Array2::from_shape_fn((n, k), |_| StandardNormal.sample(&mut rng));
    let x_init = to_nd16_array2(x_init_17);

    // Convergence tolerance and iteration budget
    let tol: f32 = 1e-4;
    let maxiter = n * 5;

    // The operator closure bridges ndarray 0.16 (lobpcg boundary) ↔ 0.17 (op.apply).
    let result = if regularize {
        lobpcg(
            |x: nd16::ArrayView2<f64>| -> nd16::Array2<f64> {
                let x17 = from_nd16_view2(x);
                let mut y17 = Array2::zeros((n, x17.ncols()));
                op.apply(x17.view(), &mut y17);
                y17.zip_mut_with(&x17, |yi, &xi| *yi += REGULARIZATION_EPS * xi);
                to_nd16_array2(y17)
            },
            x_init,
            |_: nd16::ArrayViewMut2<f64>| {},
            None,
            tol,
            maxiter,
            Order::Smallest,
        )
    } else {
        lobpcg(
            |x: nd16::ArrayView2<f64>| -> nd16::Array2<f64> {
                let x17 = from_nd16_view2(x);
                let mut y17 = Array2::zeros((n, x17.ncols()));
                op.apply(x17.view(), &mut y17);
                to_nd16_array2(y17)
            },
            x_init,
            |_: nd16::ArrayViewMut2<f64>| {},
            None,
            tol,
            maxiter,
            Order::Smallest,
        )
    };

    // Extract result; treat partial convergence as success if all residuals are acceptable.
    let extract = |r: linfa_linalg::lobpcg::Lobpcg<f64>| -> EigenResult {
        (from_nd16_array1(r.eigvals), from_nd16_array2(r.eigvecs))
    };

    match result {
        Ok(r) => Some(extract(r)),
        Err((_, Some(r))) => {
            if r.rnorm.iter().all(|&norm| norm < LOBPCG_ACCEPT_TOL) {
                Some(extract(r))
            } else {
                None
            }
        }
        Err((_, None)) => None,
    }
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::CsrOperator;
    use sprs::CsMatI;

    /// Build a diagonal n×n CSR matrix with the given diagonal values.
    fn diagonal_csr(values: &[f64]) -> CsMatI<f64, usize> {
        let n = values.len();
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        CsMatI::new((n, n), indptr, indices, values.to_vec())
    }

    /// Compute the residual norm ||A*v - λ*v|| / ||v|| for one eigenpair.
    fn residual<O: LinearOperator>(op: &O, eigvec: ndarray::ArrayView1<f64>, eigval: f64) -> f64 {
        let n = eigvec.len();
        let col: Array2<f64> = eigvec.to_owned().insert_axis(ndarray::Axis(1));
        let mut av = Array2::zeros((n, 1));
        op.apply(col.view(), &mut av);
        let mut sq = 0.0_f64;
        let mut norm_sq = 0.0_f64;
        for i in 0..n {
            let diff = av[[i, 0]] - eigval * eigvec[i];
            sq += diff * diff;
            norm_sq += eigvec[i] * eigvec[i];
        }
        sq.sqrt() / norm_sq.sqrt().max(1e-300)
    }

    #[test]
    fn lobpcg_solve_diagonal_level1() {
        let diag = [0.0_f64, 0.1, 0.3, 0.7, 1.2, 2.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 2, 42, false);
        assert!(result.is_some(), "Level 1 lobpcg_solve returned None on diagonal matrix");
        let (eigvals, eigvecs) = result.unwrap();

        assert!(
            eigvals[0].abs() < 1e-6,
            "eigenvalue[0] expected ≈ 0.0, got {}",
            eigvals[0]
        );
        assert!(
            (eigvals[1] - 0.1).abs() < 1e-6,
            "eigenvalue[1] expected ≈ 0.1, got {}",
            eigvals[1]
        );

        for i in 0..eigvals.len() {
            let r = residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(r < 1e-4, "residual for eigenpair {i}: {r} >= 1e-4");
        }
    }

    #[test]
    fn lobpcg_solve_level2_converges() {
        let diag = [0.0_f64, 0.1, 0.3, 0.7, 1.2, 2.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 2, 42, true);
        assert!(result.is_some(), "Level 2 lobpcg_solve returned None on diagonal matrix");
        let (eigvals, eigvecs) = result.unwrap();

        // Regularization shifts all eigenvalues up by REGULARIZATION_EPS
        assert!(
            (eigvals[0] - (diag[0] + REGULARIZATION_EPS)).abs() < 1e-5,
            "eigenvalue[0] expected ≈ {}, got {}",
            diag[0] + REGULARIZATION_EPS,
            eigvals[0]
        );
        assert!(
            (eigvals[1] - (diag[1] + REGULARIZATION_EPS)).abs() < 1e-5,
            "eigenvalue[1] expected ≈ {}, got {}",
            diag[1] + REGULARIZATION_EPS,
            eigvals[1]
        );

        // Check eigenvector quality via residuals against the original (unshifted) operator.
        // The regularized operator shifts eigenvalues by REGULARIZATION_EPS; remove that shift.
        for i in 0..eigvals.len() {
            let r = residual(&op, eigvecs.column(i), eigvals[i] - REGULARIZATION_EPS);
            assert!(r < 1e-4, "level2 residual for eigenpair {i}: {r} >= 1e-4");
        }
    }

    #[test]
    fn lobpcg_solve_returns_k_plus_one_eigenpairs() {
        // LOBPCG requires n > 5*k (see linfa-linalg's commented-out guard) to avoid
        // ill-conditioned block Gram matrices. With n_components=3 (k=4), need n > 20.
        // Use n=30 with well-separated eigenvalues for reliable convergence.
        let n = 30;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect(); // [1.0, 2.0, ..., 30.0]
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let n_components = 3;
        let result = lobpcg_solve(&op, n_components, 42, false);
        assert!(result.is_some(), "lobpcg_solve returned None for n=30 diagonal");
        let (eigvals, eigvecs) = result.unwrap();

        let k = n_components + 1;
        assert_eq!(eigvecs.nrows(), n, "eigvecs rows expected {n}, got {}", eigvecs.nrows());
        assert_eq!(eigvecs.ncols(), k, "eigvecs cols expected {k}, got {}", eigvecs.ncols());
        assert_eq!(eigvals.len(), k, "eigvals length expected {k}, got {}", eigvals.len());
    }

    #[test]
    fn lobpcg_solve_eigenvectors_orthonormal() {
        // Start at 0.25 (not 0.0) to avoid a degenerate zero direction in the initial block.
        let diag: Vec<f64> = (1..=8).map(|i| i as f64 * 0.25).collect();
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 3, 42, false);
        assert!(result.is_some(), "lobpcg_solve returned None");
        let (_eigvals, eigvecs) = result.unwrap();

        // Check V^T V ≈ I
        let k = eigvecs.ncols();
        let gram = eigvecs.t().dot(&eigvecs);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = gram[[i, j]];
                assert!(
                    (got - expected).abs() < 1e-8,
                    "Gram matrix [{i},{j}] expected {expected}, got {got}"
                );
            }
        }
    }
}
