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

/// Maximum number of warm-restart attempts after unconvergence detection.
/// If per-vector residuals still exceed LOBPCG_ACCEPT_TOL after this many
/// restarts, lobpcg_solve returns the best result obtained so the escalation
/// chain in mod.rs can decide to escalate.
const MAX_WARM_RESTARTS: usize = 3;

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
/// Given approximate eigenvectors X (n × k), forms the dense Gram matrix G = X^T L X
/// and diagonalises it to recover exact Rayleigh quotients and rotate X into the
/// best eigenvector basis within the span of the LOBPCG output.
/// Returns None if the Gram eigenproblem fails (numerically degenerate Gram matrix).
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

/// Compute per-vector residuals ||L·vᵢ - λᵢ·vᵢ|| / ||vᵢ|| for each eigenpair.
///
/// Uses a single batched `op.apply` for all k vectors, then accumulates
/// per-column difference norms. Returns a `Vec<f64>` of length k.
fn per_vector_residuals<O: LinearOperator>(
    op: &O,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> Vec<f64> {
    let n = eigenvectors.nrows();
    let k = eigenvectors.ncols();
    let mut ax = Array2::zeros((n, k));
    op.apply(eigenvectors.view(), &mut ax);
    eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &lambda)| {
            let col_norm = eigenvectors
                .column(i)
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(f64::EPSILON);
            let diff_norm = (0..n)
                .map(|r| (ax[[r, i]] - lambda * eigenvectors[[r, i]]).powi(2))
                .sum::<f64>()
                .sqrt();
            diff_norm / col_norm
        })
        .collect()
}

/// LOBPCG iterative eigensolver (Levels 1 and 2).
/// Level 1: no regularization. Level 2: adds epsilon*I shift.
pub fn lobpcg_solve<O: LinearOperator>(
    op: &O,
    n_components: usize,
    seed: u64,
    regularize: bool,
    sqrt_deg: &Array1<f64>,
) -> Option<(EigenResult, usize)> {
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

    let mut x_init_opt: Option<nd16::Array2<f64>> = Some(to_nd16_array2(x_init_17));
    let mut last_result: Option<EigenResult> = None;
    let mut restart_count: usize = 0;

    // Convergence tolerance and iteration budget.
    // Cap at 300 to prevent runaway iteration on large graphs (e.g. n=5000 → n*5=25,000
    // iterations); 300 matches Python UMAP's LOBPCG default and is sufficient for
    // well-conditioned graphs while keeping cost predictable.
    // Tol is set to 1e-5 (Issue #92): tighter tolerance, combined with the ChFSI-filtered
    // starting subspace, allows the solver to achieve residual < 1e-5 (REQ-PERF-001).
    let tol: f32 = 1e-5;
    let maxiter = (n * 5).min(300);

    // Defined outside the loop: captures nothing from the environment, so no need to
    // re-create it on every iteration.
    let extract = |r: linfa_linalg::lobpcg::Lobpcg<f64>| -> EigenResult {
        (from_nd16_array1(r.eigvals), from_nd16_array2(r.eigvecs))
    };

    for restart in 0..=MAX_WARM_RESTARTS {
        // Extract x_init for this iteration (moved into lobpcg; repopulated on warm restart).
        debug_assert!(x_init_opt.is_some(), "x_init_opt must be set at every loop entry");
        let x_init_nd16 = x_init_opt.take().expect("x_init_opt is always set at loop entry");

        // ── Run linfa-linalg LOBPCG ──────────────────────────────────────────
        let lobpcg_result = if regularize {
            lobpcg(
                |x: nd16::ArrayView2<f64>| -> nd16::Array2<f64> {
                    let x17 = from_nd16_view2(x);
                    let mut y17 = Array2::zeros((n, x17.ncols()));
                    op.apply(x17.view(), &mut y17);
                    y17.zip_mut_with(&x17, |yi, &xi| *yi += REGULARIZATION_EPS * xi);
                    to_nd16_array2(y17)
                },
                x_init_nd16,
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
                x_init_nd16,
                |_: nd16::ArrayViewMut2<f64>| {},
                None,
                tol,
                maxiter,
                Order::Smallest,
            )
        };

        // ── Extract eigenpairs if rnorms are acceptable ──────────────────────
        // When rnorms fail the gate but partial eigvecs exist, preserve them as a
        // warm-restart seed (comment 2971872705): a partially-converged subspace is
        // better than a cold random start.
        let (raw_opt, partial_seed): (Option<EigenResult>, Option<Array2<f64>>) =
            match lobpcg_result {
                Ok(r) => (Some(extract(r)), None),
                Err((_, Some(r))) => {
                    if r.rnorm.iter().all(|&norm| norm < LOBPCG_ACCEPT_TOL) {
                        (Some(extract(r)), None)
                    } else {
                        let seed_vecs = from_nd16_array2(r.eigvecs);
                        (None, Some(seed_vecs))
                    }
                }
                Err((_, None)) => (None, None),
            };

        // If no primary result is available, attempt a warm restart with partial eigvecs
        // before giving up (addresses the silent-discard bug in comment 2971872705).
        if raw_opt.is_none() {
            if restart < MAX_WARM_RESTARTS {
                if let Some(seed) = partial_seed {
                    log::debug!(
                        "[lobpcg] rnorm gate failed at restart {restart}/{MAX_WARM_RESTARTS}; \
                         seeding next restart with partial eigvecs"
                    );
                    x_init_opt = Some(to_nd16_array2(seed));
                    continue;
                }
            }
            // No eigvecs to seed the next restart, or restarts exhausted:
            // returning last_result is a partial fallback if any prior restart succeeded,
            // or None (true escalation) if this is the first iteration.
            log::debug!(
                "[lobpcg] no usable result at restart {restart}/{MAX_WARM_RESTARTS}; \
                 last_result is {}; breaking",
                if last_result.is_some() { "Some (partial fallback)" } else { "None (escalating)" }
            );
            break;
        }

        // ── Rayleigh-Ritz refinement ─────────────────────────────────────────
        let (_, raw_eigvecs) = raw_opt.unwrap();
        let refined_opt = rayleigh_ritz_refine(op, raw_eigvecs.clone());

        match refined_opt {
            None => {
                // RR failed (degenerate Gram matrix). Try raw eigvecs as warm-restart seed
                // before giving up (comment 2971872707): asymmetric with the unconverged path
                // which does reuse eigvecs for the next restart.
                if restart < MAX_WARM_RESTARTS {
                    log::debug!(
                        "[lobpcg] Rayleigh-Ritz failed at restart {restart}/{MAX_WARM_RESTARTS}; \
                         seeding next restart with raw eigvecs"
                    );
                    x_init_opt = Some(to_nd16_array2(raw_eigvecs));
                    continue;
                }
                // Restarts exhausted: same dual-outcome break as the rnorm-fail path above.
                log::debug!(
                    "[lobpcg] Rayleigh-Ritz failed at restart {restart}/{MAX_WARM_RESTARTS} \
                     (final); last_result is {}; breaking",
                    if last_result.is_some() { "Some (partial fallback)" } else { "None (escalating)" }
                );
                break;
            }
            Some(result) => {
                // ── Unconvergence detection (REQ-UCD-001) ──────────────────
                let residuals = per_vector_residuals(op, &result.0, &result.1);
                if residuals.iter().all(|&r| r < LOBPCG_ACCEPT_TOL) {
                    if restart > 0 {
                        log::debug!(
                            "[lobpcg] warm restart recovered convergence in {restart} round(s)"
                        );
                    }
                    return Some((result, restart_count));
                }
                let max_res = residuals
                    .iter()
                    .cloned()
                    .fold(0.0_f64, f64::max);
                restart_count += 1;
                log::debug!(
                    "[lobpcg] unconvergence detected after Rayleigh-Ritz \
                     (max_residual={max_res:.2e}); warm restart {}/{MAX_WARM_RESTARTS}",
                    restart + 1
                );
                // ── Warm restart: use refined eigvecs as next x_init (REQ-UCD-004) ──
                if restart < MAX_WARM_RESTARTS {
                    x_init_opt = Some(to_nd16_array2(result.1.clone()));
                }
                last_result = Some(result);
            }
        }
    }

    // Restarts exhausted: return best result so mod.rs quality gate decides escalation.
    last_result.map(|r| (r, restart_count))
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
        let ((eigvals, eigvecs), _) = result.unwrap();

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
        let ((eigvals, eigvecs), _) = result.unwrap();

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
        let ((eigvals, eigvecs), _) = result.unwrap();

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
        let ((eigvals, eigvecs), _) = result.unwrap();

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
        let (eigenvalues, eigvecs_out) = rayleigh_ritz_refine(&op, eigvecs)
            .expect("T-RR-1: rayleigh_ritz_refine returned None");

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

    // T-RR-1b: rayleigh_ritz_refine with perturbed (near-) eigenvectors exercises the rotation step.
    #[test]
    fn rayleigh_ritz_refine_perturbed_eigenvectors() {
        // diag = [0.1, 0.5] — well-separated so RR can distinguish them from a mixed subspace.
        let diag = [0.1_f64, 0.5];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        // Perturbed orthonormal basis: columns are not aligned with eigenvectors.
        // Uses a 30-degree rotation so the Gram matrix is non-diagonal and the rotation step
        // V != I, actually exercising the core RR step (X_refined = X * V).
        let c = (30.0_f64.to_radians()).cos();
        let s = (30.0_f64.to_radians()).sin();
        let eigvecs = ndarray::array![[c, s], [s, -c]]; // orthonormal but off-axis

        let (eigenvalues, eigvecs_out) = rayleigh_ritz_refine(&op, eigvecs)
            .expect("T-RR-1b: rayleigh_ritz_refine returned None");

        // Rayleigh-Ritz must recover the true eigenvalues from the mixed subspace.
        assert!(
            (eigenvalues[0] - 0.1).abs() < 1e-12,
            "T-RR-1b: eigenvalue[0] expected 0.1, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 0.5).abs() < 1e-12,
            "T-RR-1b: eigenvalue[1] expected 0.5, got {}",
            eigenvalues[1]
        );
        for i in 0..2 {
            let r = residual(&op, eigvecs_out.column(i), eigenvalues[i]);
            assert!(r < 1e-12, "T-RR-1b: residual for eigenpair {i}: {r} >= 1e-12");
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
        let (eigenvalues, eigvecs_out) = rayleigh_ritz_refine(&op, eigvecs)
            .expect("T-RR-2: rayleigh_ritz_refine returned None");

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

    // T-RR-3: end-to-end lobpcg_solve (with RR) achieves tight residuals on a diagonal matrix
    // and returns eigenvalues matching the smallest true Laplacian eigenvalues.
    #[test]
    fn lobpcg_solve_rr_achieves_tight_residuals() {
        // diag = [0.25, 0.5, ..., 2.0] (8 entries, starting at 0.25, step 0.25)
        let diag: Vec<f64> = (1..=8).map(|i| i as f64 * 0.25).collect();
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);

        let result = lobpcg_solve(&op, 3, 42, false, &Array1::ones(8));
        assert!(result.is_some(), "T-RR-3: lobpcg_solve returned None");
        let ((eigvals, eigvecs), _) = result.unwrap();

        // Eigenvalues must be the 4 smallest true Laplacian eigenvalues (k = n_components+1 = 4).
        let expected = [0.25_f64, 0.5, 0.75, 1.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (eigvals[i] - exp).abs() < 1e-6,
                "T-RR-3: eigvals[{i}] expected {exp}, got {} (RR eigenvalue accuracy)",
                eigvals[i]
            );
        }

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
        let ((_eigvals, eigvecs), _) = result.unwrap();

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

    #[test]
    fn per_vector_residuals_exact_near_zero() {
        // 4-node diagonal operator: eigenvalues [0.0, 0.1, 0.2, 0.3].
        // Columns of the 4×4 identity are exact eigenvectors.
        let diag = vec![0.0_f64, 0.1, 0.2, 0.3];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);
        let eigenvalues = Array1::from_vec(diag.clone());
        let eigenvectors = Array2::eye(4);
        let residuals = per_vector_residuals(&op, &eigenvalues, &eigenvectors);
        for &r in &residuals {
            assert!(r < 1e-12, "exact eigenvector residual={r:.2e} should be < 1e-12");
        }
    }

    #[test]
    fn per_vector_residuals_non_eigenvector_large() {
        // Same diagonal; [1, 1, 1, 1]/2 is not an eigenvector of eigenvalue 0.
        let diag = vec![0.0_f64, 0.1, 0.2, 0.3];
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);
        let eigenvalues = Array1::from_vec(vec![0.0_f64]);
        let v = vec![0.5_f64, 0.5, 0.5, 0.5];
        let eigenvectors = Array2::from_shape_vec((4, 1), v).unwrap();
        let residuals = per_vector_residuals(&op, &eigenvalues, &eigenvectors);
        // Exact residual: ||(diag([0,0.1,0.2,0.3])·v - 0·v)|| / ||v||
        // = ||[0, 0.05, 0.1, 0.15]|| / ||[0.5,0.5,0.5,0.5]|| ≈ 0.187.
        // Threshold 0.15 is tight enough to catch regressions while remaining below the exact value.
        assert!(residuals[0] > 0.15,
            "non-eigenvector residual={:.2e} should be large", residuals[0]);
    }

    #[test]
    fn lobpcg_solve_near_degenerate_residuals_within_tol() {
        // n=2000 diagonal Laplacian with two near-zero eigenvalues (λ₁=1e-4, λ₂=2e-4)
        // followed by well-separated spectrum [0.1, 0.2, ..., 1.99].
        // This forces LOBPCG into the near-degenerate eigengap regime.
        let n = 2000;
        let mut diag = vec![1e-4_f64, 2e-4];
        for i in 2..n {
            diag.push(0.1 + (i as f64 - 2.0) * 0.1 / (n as f64 - 2.0));
        }
        let mat = diagonal_csr(&diag);
        let op = CsrOperator(&mat);
        let sqrt_deg = Array1::ones(n);
        let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
        assert!(result.is_some(), "lobpcg_solve returned None on near-degenerate input");
        let ((eigs, vecs), _) = result.unwrap();
        let residuals = per_vector_residuals(&op, &eigs, &vecs);
        for (i, &r) in residuals.iter().enumerate() {
            assert!(r < LOBPCG_ACCEPT_TOL,
                "eigenpair {i} residual={r:.2e} exceeds LOBPCG_ACCEPT_TOL={LOBPCG_ACCEPT_TOL:.2e}");
        }
    }
}
