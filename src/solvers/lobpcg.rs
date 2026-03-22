// linfa-linalg 0.2 uses ndarray 0.16, while this crate uses ndarray 0.17.
// We alias ndarray 0.16 as `nd16` for use at the linfa-linalg boundary only.
// All internal types and return values use ndarray 0.17 (the `ndarray` crate).
extern crate ndarray16;
use ndarray16 as nd16;

use faer::{Mat as FaerMat, Side};
use faer::linalg::solvers::SelfAdjointEigen;
use linfa_linalg::lobpcg::{lobpcg, Order};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use super::EigenResult;
use crate::operator::LinearOperator;

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
/// Set to 1e-5 (Issue #92 REQ-TOL-003): matches the solver's `tol` and is achievable
/// with ChFSI-filtered starting subspace for well-conditioned graphs.
const LOBPCG_ACCEPT_TOL: f64 = 1e-5;

/// Epsilon shift applied to the operator in Level 2 (regularized) LOBPCG.
pub const REGULARIZATION_EPS: f64 = 1e-5;

/// Lower bound of the unwanted eigenvalue interval for Chebyshev preconditioning.
/// Conservative choice: works for UMAP k-NN graphs where lambda_{k+1} < 0.1.
const CHEB_LOWER_BOUND: f64 = 0.1;

/// Upper bound of the unwanted eigenvalue interval for Chebyshev preconditioning.
/// Exactly 2.0: the spectral radius of any normalized Laplacian is bounded by 2.
const CHEB_UPPER_BOUND: f64 = 2.0;

/// Polynomial degree for the Chebyshev preconditioner.
/// Degree 8 provides ~8x iteration reduction for lambda_2 ≈ 0.01 (blobs_connected_2000).
const CHEB_DEGREE: usize = 8;

/// Minimum matrix size for applying the Chebyshev preconditioner.
///
/// T_d((L - cI)/e) takes negative values for eigenvalues in (a, b), making the preconditioner
/// non-positive-definite. For small synthetic matrices (n < CHEB_MIN_N) this causes LOBPCG to
/// converge incorrectly. In practice, LOBPCG is only invoked for large graphs (n >= 2000)
/// where the filter is both numerically stable and beneficial for convergence.
const CHEB_MIN_N: usize = 1000;

// ─── LOBPCG solver ────────────────────────────────────────────────────────────

/// Chebyshev polynomial preconditioner of degree `degree`.
///
/// Applies `T_degree((L - cI)/e)` to the residual block `r`, where
/// `c = (a + b) / 2` and `e = (b - a) / 2`. The three-term recurrence
/// amplifies components with eigenvalues in `[0, a)` and suppresses those
/// in `[a, b]`, accelerating LOBPCG convergence for low-eigenvalue problems.
///
/// - `l_op`: closure applying the Laplacian L (ndarray 0.17 types)
/// - `r`: residual block of shape `[n, k]`
/// - `a`: lower bound of unwanted interval (e.g. 0.1)
/// - `b`: upper bound of unwanted interval (e.g. 2.0 for normalized Laplacian)
/// - `degree`: polynomial degree; must be >= 1
fn chebyshev_precond(
    l_op: &impl Fn(ndarray::ArrayView2<f64>) -> Array2<f64>,
    r: ndarray::ArrayView2<f64>,
    a: f64,
    b: f64,
    degree: usize,
) -> Array2<f64> {
    let c = (a + b) / 2.0; // center of [a, b]
    let e = (b - a) / 2.0; // half-width of [a, b]

    // T_0 applied to r
    let mut y_prev = r.to_owned();
    // T_1 applied to r: (L·r - c·r) / e
    let lr = l_op(r);
    let mut y = (&lr - c * &r) / e;

    for _ in 2..=degree {
        let ly = l_op(y.view());
        let y_new = (2.0 / e) * (&ly - c * &y) - &y_prev;
        y_prev = y;
        y = y_new;
    }
    y
}

/// Final exact Rayleigh-Ritz refinement applied after LOBPCG convergence.
///
/// Given the LOBPCG output eigenvectors X (n × k) and the true Laplacian operator op:
/// 1. Computes AX = L * X via sparse matvec (true Laplacian, no regularization shift).
/// 2. Forms the dense Gram matrix G = X^T * AX of size (k × k).
/// 3. Solves the dense symmetric eigenproblem on G via faer SelfAdjointEigen.
/// 4. Rotates eigenvectors: X_refined = X * V (V = eigenvectors of G).
/// 5. Returns (eigenvalues_from_gram, X_refined).
///
/// Eigenvalues from the Gram solve are exact Rayleigh quotients of the true Laplacian,
/// independent of any regularization shift applied during LOBPCG iteration.
fn rayleigh_ritz_refine<O: LinearOperator>(op: &O, eigvecs: Array2<f64>) -> Option<EigenResult> {
    let n = eigvecs.nrows();
    let k = eigvecs.ncols();

    // Step 1: AX = L * X (sparse matvec with true Laplacian)
    let mut ax = Array2::zeros((n, k));
    op.apply(eigvecs.view(), &mut ax);

    // Step 2: G = X^T * AX (k×k dense Gram matrix)
    let gram = eigvecs.t().dot(&ax);

    // Step 3: Dense symmetric eigenproblem on G via faer SelfAdjointEigen.
    // Returns None instead of panicking if the Gram matrix is numerically degenerate.
    let faer_gram = FaerMat::<f64>::from_fn(k, k, |i, j| gram[[i, j]]);
    let eigen = SelfAdjointEigen::new(faer_gram.as_ref(), Side::Lower).ok()?;

    // Step 4: Extract eigenvalues (ascending) and rotation matrix V.
    // eigen.S() iterates in ascending order; use for_each matching rsvd.rs pattern.
    let mut ev_vec: Vec<f64> = Vec::with_capacity(k);
    eigen.S().for_each(|x| ev_vec.push(*x));
    let eigenvalues = Array1::from_vec(ev_vec);
    let v_rot = {
        let u = eigen.U().to_owned();
        Array2::from_shape_fn((k, k), |(i, j)| u.col_as_slice(j)[i])
    };

    // Step 5: Rotate X_refined = X * V
    let eigvecs_refined = eigvecs.dot(&v_rot);
    Some((eigenvalues, eigvecs_refined))
}

/// LOBPCG iterative eigensolver (Levels 1 and 2).
/// Level 1: no regularization. Level 2: adds epsilon*I shift.
pub fn lobpcg_solve<O: LinearOperator>(
    op: &O,
    n_components: usize,
    seed: u64,
    regularize: bool,
    sqrt_deg: &Array1<f64>,
) -> Option<EigenResult> {
    let n = op.size();
    let k = n_components + 1; // +1 to include trivial eigenvector slot

    // Validate preconditions: need at least 1 component and block size < matrix size.
    // Also require sqrt_deg to match the operator size to avoid a panic in assign().
    if n_components == 0 || k >= n || sqrt_deg.len() != n {
        return None;
    }

    // Build random initial block [n, k] in ndarray 0.17, then convert to 0.16 for lobpcg.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_init_17: Array2<f64> =
        Array2::from_shape_fn((n, k), |_| StandardNormal.sample(&mut rng));

    // Inject trivial eigenvector hint (sqrt_deg/||sqrt_deg||) into column 0, mirroring Python UMAP.
    let sqrt_deg_norm = sqrt_deg.dot(sqrt_deg).sqrt();
    if sqrt_deg_norm > 0.0 {
        x_init_17.column_mut(0).assign(&sqrt_deg.mapv(|x| x / sqrt_deg_norm));
    }

    // l_op_nd17 wraps op.apply() in ndarray 0.17 types for use by chebyshev_precond.
    let l_op_nd17 = |x: ndarray::ArrayView2<f64>| -> Array2<f64> {
        let mut y = Array2::zeros((n, x.ncols()));
        op.apply(x, &mut y);
        y
    };

    // Chebyshev-Filtered Subspace Iteration (ChFSI): for large graphs, apply a single
    // shifted Chebyshev filter (I + T_d(A)) / 2 to x_init before LOBPCG. This amplifies
    // components with eigenvalues in [0, CHEB_LOWER_BOUND) and suppresses those in
    // [CHEB_LOWER_BOUND, CHEB_UPPER_BOUND], giving LOBPCG a better starting subspace
    // without the per-iteration scale-distortion caused by an amplifying preconditioner.
    //
    // Using it as a one-time filter (not a per-iteration preconditioner) keeps LOBPCG's
    // internal eigvec norms ≈ 1, so linfa-linalg's rnorm accurately tracks actual_residual.
    // For small n, the filter's spectrum oscillates and may hurt rather than help, so it is
    // skipped; small matrices converge quickly from a random start anyway.
    if n >= CHEB_MIN_N {
        let t_d = chebyshev_precond(
            &l_op_nd17,
            x_init_17.view(),
            CHEB_LOWER_BOUND,
            CHEB_UPPER_BOUND,
            CHEB_DEGREE,
        );
        // Shifted Chebyshev: (I + T_d(A)) / 2 · x_init. Always non-negative.
        let filtered = (x_init_17 + t_d) / 2.0;
        // Normalize each column to unit norm for LOBPCG's orthonormal-basis requirement.
        x_init_17 = Array2::zeros((n, k));
        for j in 0..filtered.ncols() {
            let col = filtered.column(j);
            let norm = col.dot(&col).sqrt();
            if norm > 1e-300 {
                x_init_17.column_mut(j).assign(&col.mapv(|v| v / norm));
            } else {
                x_init_17.column_mut(j).assign(&col);
            }
        }
    }

    let x_init = to_nd16_array2(x_init_17);

    // Convergence tolerance and iteration budget.
    // Cap at 300 to prevent runaway iteration on large graphs (e.g. n=5000 → n*5=25,000
    // iterations); 300 matches Python UMAP's LOBPCG default and is sufficient for
    // well-conditioned graphs while keeping cost predictable.
    // Tol is set to 1e-5 (Issue #92): tighter tolerance, combined with the ChFSI-filtered
    // starting subspace, allows the solver to achieve residual < 1e-5 (REQ-PERF-001).
    let tol: f32 = 1e-5;
    let maxiter = (n * 5).min(300);

    // The operator closure bridges ndarray 0.16 (lobpcg boundary) ↔ 0.17 (op.apply).
    // No per-iteration preconditioner: the ChFSI filter above already improves the starting
    // subspace. A per-iteration amplifying filter distorts eigvec norms, breaking LOBPCG's
    // rnorm-to-actual-residual correspondence and causing acceptance of unconverged results.
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

    let result_opt = match result {
        Ok(r) => Some(extract(r)),
        Err((_, Some(r))) => {
            if r.rnorm.iter().all(|&norm| norm < LOBPCG_ACCEPT_TOL) {
                Some(extract(r))
            } else {
                None
            }
        }
        Err((_, None)) => None,
    };

    // Apply Rayleigh-Ritz refinement: recompute eigenvalues as exact Rayleigh quotients
    // of the true Laplacian (G = X^T * L * X), then rotate eigenvectors.
    // This replaces LOBPCG eigenvalues (including any regularization shift) with exact
    // values — no separate REGULARIZATION_EPS subtraction is needed.
    // Note: LOBPCG eigenvalues are intentionally discarded; the returned eigenvalues are
    // the Gram-solve Rayleigh quotients, which are exact for the true (unshifted) Laplacian.
    result_opt.and_then(|(_, eigvecs)| rayleigh_ritz_refine(op, eigvecs))
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
    fn lobpcg_injects_trivial_eigenvector() {
        // Diagonal Laplacian with near-trivial first eigenvalue (1e-10) and well-separated
        // remaining eigenvalues. n=20 satisfies n > 5·k for LOBPCG convergence (k = n_components+1 = 3).
        let n = 20;
        let mut diag = vec![1e-10_f64]; // near-zero first eigenvalue
        for i in 1..n {
            diag.push(i as f64 * 0.1); // 0.1, 0.2, ..., 1.9
        }
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        // sqrt_deg is derived from the diagonal entries (treating them as node degrees),
        // so the injection is mathematically consistent with the matrix being solved.
        let sqrt_deg = Array1::from_iter(diag.iter().map(|&d| d.sqrt()));

        let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
        assert!(result.is_some(), "lobpcg_solve returned None with trivial eigenvector injection");
        let (eigvals, eigvecs) = result.unwrap();

        // First eigenvalue must be near zero (the near-trivial eigenvalue is 1e-10).
        assert!(
            eigvals[0] < 1e-6,
            "eigenvalue[0] expected ≈ 0.0 (near-trivial), got {}",
            eigvals[0]
        );

        // All residuals must be below tolerance — verifies eigenvector quality.
        for i in 0..eigvals.len() {
            let r = residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(r < 1e-5, "residual for eigenpair {i}: {r} >= 1e-5");
        }
    }

    #[test]
    fn lobpcg_solve_diagonal_level1() {
        let diag = [0.0_f64, 0.1, 0.3, 0.7, 1.2, 2.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 2, 42, false, &Array1::from_vec(vec![1.0_f64; 6]));
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
            assert!(r < 1e-5, "residual for eigenpair {i}: {r} >= 1e-5");
        }
    }

    #[test]
    fn lobpcg_solve_level2_eigenvalues_corrected() {
        let diag = [0.0_f64, 0.1, 0.3, 0.7, 1.2, 2.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 2, 42, true, &Array1::from_vec(vec![1.0_f64; 6]));
        assert!(result.is_some(), "Level 2 lobpcg_solve returned None on diagonal matrix");
        let (eigvals, eigvecs) = result.unwrap();

        // Acceptance criterion: eigenvalues within 1e-6 of true Laplacian eigenvalues.
        // 1e-6 is tight enough to catch a missing REGULARIZATION_EPS subtraction (eps = 1e-5),
        // and reliably achievable for a diagonal matrix whose eigenvectors are the standard
        // basis vectors (no numerical ill-conditioning in the eigenvector solve).
        assert!(
            (eigvals[0] - diag[0]).abs() < 1e-6,
            "eigenvalue[0] expected ≈ {}, got {} (REGULARIZATION_EPS not subtracted?)",
            diag[0],
            eigvals[0]
        );
        assert!(
            (eigvals[1] - diag[1]).abs() < 1e-6,
            "eigenvalue[1] expected ≈ {}, got {} (REGULARIZATION_EPS not subtracted?)",
            diag[1],
            eigvals[1]
        );

        // Residual check against original Laplacian — no manual shift needed by the caller
        for i in 0..eigvals.len() {
            let r = residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(r < 1e-5, "level2 residual for eigenpair {i}: {r} >= 1e-5");
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
        let result = lobpcg_solve(&op, n_components, 42, false, &Array1::ones(30));
        assert!(result.is_some(), "lobpcg_solve returned None for n=30 diagonal");
        let (eigvals, eigvecs) = result.unwrap();

        let k = n_components + 1;
        assert_eq!(eigvecs.nrows(), n, "eigvecs rows expected {n}, got {}", eigvecs.nrows());
        assert_eq!(eigvecs.ncols(), k, "eigvecs cols expected {k}, got {}", eigvecs.ncols());
        assert_eq!(eigvals.len(), k, "eigvals length expected {k}, got {}", eigvals.len());
    }

    #[test]
    fn chebyshev_precond_filters_high_eigenvalues() {
        // Diagonal L = diag(0.02, 0.02, 1.0, ..., 1.0) — 2 low + 8 high eigenvalues.
        // The Chebyshev filter (a=0.1, b=2.0, degree=8) should amplify the low-eigenvalue
        // direction relative to the high-eigenvalue direction by at least 10x.
        let diag = [0.02_f64, 0.02, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);
        let l_op = |x: ndarray::ArrayView2<f64>| -> Array2<f64> {
            let mut y = Array2::zeros((mat.rows(), x.ncols()));
            op.apply(x, &mut y);
            y
        };

        // column 0 = e_0 (eigenvalue 0.02 direction), column 1 = e_2 (eigenvalue 1.0 direction)
        let mut r = Array2::zeros((10, 2));
        r[[0, 0]] = 1.0;
        r[[2, 1]] = 1.0;

        let result = chebyshev_precond(&l_op, r.view(), 0.1, 2.0, 8);

        let norm_col0 = result.column(0).dot(&result.column(0)).sqrt();
        let norm_col1 = result.column(1).dot(&result.column(1)).sqrt();

        assert!(
            norm_col0 > norm_col1 * 10.0,
            "low-eigenvalue direction (col0 norm={norm_col0}) should be >> high-eigenvalue direction (col1 norm={norm_col1})"
        );
    }

    #[test]
    fn chebyshev_precond_degree1_matches_formula() {
        // For degree=1: result = (L·r - c·r) / e where c=(a+b)/2, e=(b-a)/2.
        // Use a=0.0, b=2.0 → c=1.0, e=1.0, so result = L·r - r.
        // L = diag(0.5, 1.5, 0.5, 1.5), r = ones(4, 1)
        // Expected = [0.5-1, 1.5-1, 0.5-1, 1.5-1]ᵀ = [-0.5, 0.5, -0.5, 0.5]ᵀ
        let diag = [0.5_f64, 1.5, 0.5, 1.5];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);
        let l_op = |x: ndarray::ArrayView2<f64>| -> Array2<f64> {
            let mut y = Array2::zeros((mat.rows(), x.ncols()));
            op.apply(x, &mut y);
            y
        };

        let r = Array2::ones((4, 1));
        let result = chebyshev_precond(&l_op, r.view(), 0.0, 2.0, 1);

        let expected = [-0.5_f64, 0.5, -0.5, 0.5];
        for i in 0..4 {
            assert!(
                (result[[i, 0]] - expected[i]).abs() < 1e-14,
                "element [{i}] expected {}, got {}",
                expected[i],
                result[[i, 0]]
            );
        }
    }

    // T-RR-1: rayleigh_ritz_refine with exact eigenvectors of a diagonal matrix.
    #[test]
    fn rayleigh_ritz_refine_exact_eigenvectors_diagonal() {
        let diag = [0.0_f64, 0.1, 0.3, 0.7];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        // Identity matrix: exact eigenvectors of a diagonal matrix are standard basis vectors.
        let eigvecs = ndarray::Array2::eye(4);
        let (eigenvalues, eigvecs_out) = rayleigh_ritz_refine(&op, eigvecs);

        for (i, &expected) in diag.iter().enumerate() {
            assert!(
                (eigenvalues[i] - expected).abs() < 1e-12,
                "T-RR-1: eigenvalue[{i}] expected {expected}, got {}",
                eigenvalues[i]
            );
        }
        for i in 0..4 {
            let r = residual(&op, eigvecs_out.column(i), eigenvalues[i]);
            assert!(r < 1e-12, "T-RR-1: residual for eigenpair {i}: {r} >= 1e-12");
        }
    }

    // T-RR-2: rayleigh_ritz_refine recovers eigenvalues from a rotated basis.
    #[test]
    fn rayleigh_ritz_refine_rotated_basis_recovers_eigenvalues() {
        let diag = [1.0_f64, 4.0];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        // 45-degree rotated orthonormal basis
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let eigvecs = ndarray::array![[s, s], [s, -s]];
        let (eigenvalues, eigvecs_out) = rayleigh_ritz_refine(&op, eigvecs);

        assert!(
            (eigenvalues[0] - 1.0).abs() < 1e-12,
            "T-RR-2: eigenvalue[0] expected 1.0, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 4.0).abs() < 1e-12,
            "T-RR-2: eigenvalue[1] expected 4.0, got {}",
            eigenvalues[1]
        );
        for i in 0..2 {
            let r = residual(&op, eigvecs_out.column(i), eigenvalues[i]);
            assert!(r < 1e-12, "T-RR-2: residual for eigenpair {i}: {r} >= 1e-12");
        }
    }

    // T-RR-3: end-to-end lobpcg_solve (with RR) achieves tight residuals on a diagonal matrix.
    #[test]
    fn lobpcg_solve_rr_achieves_tight_residuals() {
        // diag = [0.25, 0.5, ..., 2.0] (8 entries, starting at 0.25, step 0.25)
        let diag: Vec<f64> = (1..=8).map(|i| i as f64 * 0.25).collect();
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 3, 42, false, &Array1::ones(8));
        assert!(result.is_some(), "T-RR-3: lobpcg_solve returned None");
        let (eigvals, eigvecs) = result.unwrap();

        for i in 0..eigvals.len() {
            let r = residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(r < 1e-8, "T-RR-3: residual for eigenpair {i}: {r} >= 1e-8");
        }
    }

    #[test]
    fn lobpcg_accept_tol_is_1e5() {
        // REQ-TOL-003: partial-convergence acceptance tolerance must be 1e-5.
        assert_eq!(LOBPCG_ACCEPT_TOL, 1e-5_f64,
            "LOBPCG_ACCEPT_TOL must be 1e-5 per Issue #92 REQ-TOL-003");
    }

    #[test]
    fn lobpcg_solve_eigenvectors_orthonormal() {
        // Start at 0.25 (not 0.0) to avoid a degenerate zero direction in the initial block.
        let diag: Vec<f64> = (1..=8).map(|i| i as f64 * 0.25).collect();
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 3, 42, false, &Array1::ones(8));
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
