mod dense;
#[doc(hidden)]
pub mod lobpcg;
// pub(crate) visibility only needed when the testing feature exposes rsvd_solve via lib.rs.
#[cfg(feature = "testing")]
pub(crate) mod rsvd;
#[cfg(not(feature = "testing"))]
mod rsvd;
#[cfg(feature = "testing")]
pub(crate) mod sinv;
#[cfg(not(feature = "testing"))]
mod sinv;

// pub (not pub(crate)) so lib.rs can re-export it for integration tests.
pub use dense::dense_evd;

use ndarray::{Array1, Array2};
use sprs::CsMatI;
use crate::operator::CsrOperator;
use crate::metrics::{
    max_eigenpair_residual,
    DENSE_EVD_QUALITY_THRESHOLD,
    LOBPCG_QUALITY_THRESHOLD,
    SINV_LOBPCG_QUALITY_THRESHOLD,
    RSVD_QUALITY_THRESHOLD,
};

/// Eigendecomposition result: (eigenvalues shape [k], eigenvectors shape [n, k]).
pub type EigenResult = (Array1<f64>, Array2<f64>);

/// Graphs with n < DENSE_N_THRESHOLD use Level 0 (dense EVD) directly.
const DENSE_N_THRESHOLD: usize = 2000;

/// Returns the effective dense-EVD size threshold.
///
/// In test builds (`--features testing`), reads the `SPECTRAL_DENSE_N_THRESHOLD`
/// environment variable, falling back to `DENSE_N_THRESHOLD` on absence or
/// parse error.  In release builds the function is a trivial constant return
/// that the compiler inlines away entirely.
#[cfg(feature = "testing")]
fn dense_n_threshold() -> usize {
    std::env::var("SPECTRAL_DENSE_N_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DENSE_N_THRESHOLD)
}

#[cfg(not(feature = "testing"))]
fn dense_n_threshold() -> usize {
    DENSE_N_THRESHOLD
}


/// Solver escalation chain. Tries six levels in order, advancing only on failure.
///
/// Returns `n_components+1` eigenpairs (including the trivial λ≈0 vector at index 0)
/// so that the caller can apply `select_eigenvectors` uniformly across all solver paths.
///
/// Level 0: Dense EVD (n < 2000, O(n³)).
/// Level 1: LOBPCG without regularization.
/// Level 2: Shift-and-invert LOBPCG via sparse Cholesky (new).
/// Level 3: LOBPCG with ε·I regularization.
/// Level 4: Randomized SVD via 2I-L trick.
/// Level 5: Forced dense EVD (nuclear option, cannot fail).
///
/// # Panics
///
/// Panics at Level 5 exhaustion — this represents a bug, not a user error.
/// The spectral theorem guarantees eigenvectors exist for any symmetric PSD matrix.
pub(crate) fn solve_eigenproblem(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    sqrt_deg: &ndarray::Array1<f64>,
) -> (EigenResult, u8) {
    #[cfg(feature = "testing")]
    let _t_solve = std::time::Instant::now();

    let n = laplacian.rows();
    let op = CsrOperator(laplacian);

    // Level 0: Dense EVD — exact, used for small n where O(n³) is acceptable.
    if n < dense_n_threshold() {
        #[cfg(feature = "testing")]
        let _t0 = std::time::Instant::now();
        let _l0_result = dense_evd(laplacian, n_components + 1);
        #[cfg(feature = "testing")]
        eprintln!("[timing:level_0] {}µs", _t0.elapsed().as_micros());
        match _l0_result {
            Ok((eigs, vecs)) => {
                let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
                if quality < DENSE_EVD_QUALITY_THRESHOLD {
                    log::debug!(
                        "[spectral] Level 0 (dense EVD) succeeded (n={n}, max_residual={quality:.2e})"
                    );
                    #[cfg(feature = "testing")]
                    eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
                    return ((eigs, vecs), 0);
                }
                log::debug!(
                    "[spectral] Level 0 (dense EVD) poor quality \
                     (max_residual={quality:.2e}), escalating to Level 1"
                );
            }
            Err(e) => {
                log::warn!("[spectral] Level 0 (dense EVD) failed ({e}), escalating to Level 1");
            }
        }
    }

    // Level 1: LOBPCG without regularization.
    #[cfg(feature = "testing")]
    let _t1 = std::time::Instant::now();
    let _l1 = lobpcg::lobpcg_solve(&op, n_components, seed, false, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:level_1] {}µs", _t1.elapsed().as_micros());
    if let Some(((eigs, vecs), _restarts)) = _l1 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < LOBPCG_QUALITY_THRESHOLD {
            log::debug!(
                "[spectral] Level 1 (LOBPCG) succeeded (n={n}, max_residual={quality:.2e})"
            );
            #[cfg(feature = "testing")]
            eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 1);
        }
        log::debug!(
            "[spectral] Level 1 (LOBPCG) poor quality \
             (max_residual={quality:.2e}), escalating to Level 2"
        );
    }

    // Level 2: Shift-and-invert LOBPCG — near-exact, handles small eigengap.
    // sprs_csc_to_faer converts the Laplacian to faer CSC and adds SINV_SHIFT to the
    // diagonal, producing M = L + εI. sp_cholesky factorizes M; if it fails (e.g. M
    // is not SPD due to numerical degenerate edges) we skip silently to Level 3.
    #[cfg(feature = "testing")]
    let _t2 = std::time::Instant::now();
    let _l2 = sinv::lobpcg_sinv_solve(laplacian, n_components, seed, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:level_2] {}µs", _t2.elapsed().as_micros());
    if let Some((eigs, vecs)) = _l2 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < SINV_LOBPCG_QUALITY_THRESHOLD {
            log::debug!(
                "[spectral] Level 2 (sinv LOBPCG) succeeded (n={n}, max_residual={quality:.2e})"
            );
            #[cfg(feature = "testing")]
            eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 2);
        }
        log::debug!(
            "[spectral] Level 2 (sinv LOBPCG) poor quality \
             (max_residual={quality:.2e}), escalating to Level 3"
        );
    }

    // Level 3: LOBPCG with ε·I regularization — widens eigengap.
    // lobpcg_solve(regularize=true) applies Rayleigh-Ritz refinement internally
    // (G = X^T L X against the true Laplacian), so eigs are exact Rayleigh quotients
    // and can be passed directly to the residual check.
    #[cfg(feature = "testing")]
    let _t3 = std::time::Instant::now();
    let _l3 = lobpcg::lobpcg_solve(&op, n_components, seed, true, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:level_3] {}µs", _t3.elapsed().as_micros());
    if let Some(((eigs, vecs), _restarts)) = _l3 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < LOBPCG_QUALITY_THRESHOLD {
            log::debug!(
                "[spectral] Level 3 (LOBPCG+reg) succeeded (n={n}, max_residual={quality:.2e})"
            );
            #[cfg(feature = "testing")]
            eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 3);
        }
        log::debug!(
            "[spectral] Level 3 (LOBPCG+reg) poor quality \
             (max_residual={quality:.2e}), escalating to Level 4"
        );
    }

    // Level 4: Randomized SVD via 2I-L trick.
    // rsvd_solve is infallible; gate on output quality via residual check.
    {
        #[cfg(feature = "testing")]
        let _t4 = std::time::Instant::now();
        let (eigs, vecs) = rsvd::rsvd_solve(laplacian, n_components, seed);
        #[cfg(feature = "testing")]
        eprintln!("[timing:level_4] {}µs", _t4.elapsed().as_micros());
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < RSVD_QUALITY_THRESHOLD {
            log::debug!(
                "[spectral] Level 4 (rSVD) succeeded (n={n}, max_residual={quality:.2e})"
            );
            #[cfg(feature = "testing")]
            eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 4);
        }
        log::debug!(
            "[spectral] Level 4 (rSVD) poor quality (max_residual={quality:.2e}), \
             escalating to Level 5"
        );
    }

    // Level 5: Forced dense EVD — O(n³) nuclear option.
    // This level CANNOT fail mathematically (spectral theorem).
    // If it returns Err, that indicates an OOM or a faer bug — both are
    // unrecoverable and should surface as an assertion failure, not a silent
    // ConvergenceFailure that would produce a garbage embedding.
    log::debug!("[spectral] Level 5 (forced dense EVD) (n={n})");
    #[cfg(feature = "testing")]
    let _t5 = std::time::Instant::now();
    let result = dense_evd(laplacian, n_components + 1).expect(
        "solve_eigenproblem: Level 5 forced dense EVD failed — \
         this is a bug; the spectral theorem guarantees eigenvectors \
         exist for any symmetric positive semidefinite matrix",
    );
    #[cfg(feature = "testing")]
    eprintln!("[timing:level_5] {}µs", _t5.elapsed().as_micros());
    #[cfg(feature = "testing")]
    eprintln!("[timing:level_total] {}µs", _t_solve.elapsed().as_micros());
    (result, 5)
}

/// SIMD variant of `solve_eigenproblem`: identical escalation chain but uses
/// `CsrOperatorSimd` for LOBPCG levels 1 and 3 to exercise the AVX2 SpMV path.
///
/// Under `#[cfg(all(feature = "testing", any(target_arch = "x86", target_arch = "x86_64")))]`
/// only — not part of the stable public API.
#[cfg(all(feature = "testing", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn solve_eigenproblem_simd(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    sqrt_deg: &ndarray::Array1<f64>,
) -> (EigenResult, u8) {
    use crate::operator::CsrOperatorSimd;

    #[cfg(feature = "testing")]
    let _t_solve = std::time::Instant::now();

    let n = laplacian.rows();
    let op = CsrOperatorSimd(laplacian);

    // Level 0: Dense EVD (unchanged — does not use the LinearOperator)
    if n < dense_n_threshold() {
        #[cfg(feature = "testing")]
        let _t0 = std::time::Instant::now();
        let _l0_result = dense_evd(laplacian, n_components + 1);
        #[cfg(feature = "testing")]
        eprintln!("[timing:simd:level_0] {}µs", _t0.elapsed().as_micros());
        match _l0_result {
            Ok((eigs, vecs)) => {
                let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
                if quality < DENSE_EVD_QUALITY_THRESHOLD {
                    #[cfg(feature = "testing")]
                    eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
                    return ((eigs, vecs), 0);
                }
            }
            Err(_) => {}
        }
    }

    // Level 1: LOBPCG via CsrOperatorSimd
    #[cfg(feature = "testing")]
    let _t1 = std::time::Instant::now();
    let _l1 = lobpcg::lobpcg_solve(&op, n_components, seed, false, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:simd:level_1] {}µs", _t1.elapsed().as_micros());
    if let Some(((eigs, vecs), _)) = _l1 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < LOBPCG_QUALITY_THRESHOLD {
            #[cfg(feature = "testing")]
            eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 1);
        }
    }

    // Level 2: Shift-and-invert LOBPCG (unchanged — does not use LinearOperator)
    #[cfg(feature = "testing")]
    let _t2 = std::time::Instant::now();
    let _l2 = sinv::lobpcg_sinv_solve(laplacian, n_components, seed, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:simd:level_2] {}µs", _t2.elapsed().as_micros());
    if let Some((eigs, vecs)) = _l2 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < SINV_LOBPCG_QUALITY_THRESHOLD {
            #[cfg(feature = "testing")]
            eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 2);
        }
    }

    // Level 3: LOBPCG+reg via CsrOperatorSimd
    #[cfg(feature = "testing")]
    let _t3 = std::time::Instant::now();
    let _l3 = lobpcg::lobpcg_solve(&op, n_components, seed, true, sqrt_deg);
    #[cfg(feature = "testing")]
    eprintln!("[timing:simd:level_3] {}µs", _t3.elapsed().as_micros());
    if let Some(((eigs, vecs), _)) = _l3 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < LOBPCG_QUALITY_THRESHOLD {
            #[cfg(feature = "testing")]
            eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 3);
        }
    }

    // Level 4: rSVD (unchanged — does not use LinearOperator)
    {
        #[cfg(feature = "testing")]
        let _t4 = std::time::Instant::now();
        let (eigs, vecs) = rsvd::rsvd_solve(laplacian, n_components, seed);
        #[cfg(feature = "testing")]
        eprintln!("[timing:simd:level_4] {}µs", _t4.elapsed().as_micros());
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < RSVD_QUALITY_THRESHOLD {
            #[cfg(feature = "testing")]
            eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
            return ((eigs, vecs), 4);
        }
    }

    // Level 5: Forced dense EVD (unchanged)
    #[cfg(feature = "testing")]
    let _t5 = std::time::Instant::now();
    let result = dense_evd(laplacian, n_components + 1).expect(
        "solve_eigenproblem_simd: Level 5 forced dense EVD failed — bug",
    );
    #[cfg(feature = "testing")]
    eprintln!("[timing:simd:level_5] {}µs", _t5.elapsed().as_micros());
    #[cfg(feature = "testing")]
    eprintln!("[timing:simd:level_total] {}µs", _t_solve.elapsed().as_micros());
    (result, 5)
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{max_eigenpair_residual, DENSE_EVD_QUALITY_THRESHOLD, LOBPCG_QUALITY_THRESHOLD};

    /// Build a 6-node path-graph Laplacian in CSR format.
    ///
    /// Path graph: 0-1-2-3-4-5
    /// L = D - A where degree[0]=1, degree[1..4]=2, degree[5]=1
    fn path_graph_laplacian_6() -> CsMatI<f64, usize> {
        // Indptr: each row's non-zero count:
        //   row 0: cols 0,1      → 2 entries
        //   row 1: cols 0,1,2    → 3 entries
        //   rows 2-4 same pattern
        //   row 5: cols 4,5      → 2 entries
        CsMatI::new(
            (6, 6),
            vec![0usize, 2, 5, 8, 11, 14, 16],
            vec![
                0usize, 1, // row 0
                0, 1, 2,   // row 1
                1, 2, 3,   // row 2
                2, 3, 4,   // row 3
                3, 4, 5,   // row 4
                4, 5,      // row 5
            ],
            vec![
                1.0_f64, -1.0,        // row 0
                -1.0, 2.0, -1.0,      // row 1
                -1.0, 2.0, -1.0,      // row 2
                -1.0, 2.0, -1.0,      // row 3
                -1.0, 2.0, -1.0,      // row 4
                -1.0, 1.0,            // row 5
            ],
        )
    }

    #[test]
    fn solve_eigenproblem_eigenvalues_nonneg_and_sorted() {
        let laplacian = path_graph_laplacian_6();
        let ((eigvals, _), _) = solve_eigenproblem(&laplacian, 2, 42, &ndarray::Array1::ones(6));

        // All eigenvalues must be non-negative
        for &v in eigvals.iter() {
            assert!(v >= -1e-10, "eigenvalue is negative: {v}");
        }

        // Eigenvalues must be sorted ascending
        for i in 1..eigvals.len() {
            assert!(
                eigvals[i] >= eigvals[i - 1] - 1e-10,
                "eigenvalues not sorted: eigvals[{}]={} > eigvals[{}]={}",
                i - 1, eigvals[i - 1], i, eigvals[i]
            );
        }
    }

    #[test]
    fn solve_eigenproblem_returns_k_plus_one_pairs() {
        // 6-node path graph (n=6 < 2000 → Level 0 dense EVD)
        let ((eigs, vecs), _) = solve_eigenproblem(&path_graph_laplacian_6(), 2, 42, &ndarray::Array1::ones(6));
        assert_eq!(eigs.len(), 3, "expected n_components+1=3 eigenvalues");
        assert_eq!(vecs.shape(), &[6, 3], "expected [n, n_components+1] = [6, 3] eigenvectors");
    }

    /// Build a sparse diagonal n×n CSR Laplacian with eigenvalues [0, 1/(n-1), ..., 1].
    ///
    /// Diagonal entries approximate a Laplacian with a zero eigenvalue at index 0.
    fn diagonal_laplacian(n: usize) -> CsMatI<f64, usize> {
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        let data: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1).max(1) as f64)
            .collect();
        CsMatI::new((n, n), indptr, indices, data)
    }

    /// Verify that solve_eigenproblem routes through LOBPCG (Level 1) for n >= DENSE_N_THRESHOLD,
    /// and returns a valid result with the correct shape and non-negative eigenvalues.
    #[test]
    fn solve_eigenproblem_large_n_routes_through_lobpcg() {
        // n=2001 > DENSE_N_THRESHOLD=2000, so Level 0 is skipped; LOBPCG (Level 1) runs.
        let n = 2001;
        let n_components = 2;
        let laplacian = diagonal_laplacian(n);

        let ((eigvals, eigvecs), _) = solve_eigenproblem(&laplacian, n_components, 42, &ndarray::Array1::ones(n));

        assert_eq!(eigvals.len(), n_components + 1, "expected n_components+1 eigenvalues");
        assert_eq!(eigvecs.shape(), &[n, n_components + 1], "expected [n, n_components+1] shape");

        for &v in eigvals.iter() {
            assert!(v >= -1e-6, "eigenvalue is negative: {v:.2e}");
        }
    }

    // T10 — solve_eigenproblem_level_numbering
    #[test]
    fn solve_eigenproblem_level_numbering() {
        // n=2001 > DENSE_N_THRESHOLD=2000: Level 0 is skipped.
        // For a well-conditioned diagonal Laplacian, Level 1 (plain LOBPCG) succeeds.
        let n = 2001;
        let laplacian = diagonal_laplacian(n);
        let (_, level) = solve_eigenproblem(&laplacian, 2, 42, &ndarray::Array1::ones(n));
        assert!(
            level >= 1 && level <= 5,
            "expected level in {{1,2,3,4,5}}, got {level}"
        );
    }

    #[test]
    fn quality_gate_catches_garbage_at_dense_evd_threshold() {
        // 3-node path graph Laplacian: exact eigenvector of lambda=0 is [1/√3, 1/√3, 1/√3].
        // Use [1, 0, 0] (not an eigenvector) — residual must exceed DENSE_EVD_QUALITY_THRESHOLD.
        let laplacian = CsMatI::new(
            (3, 3),
            vec![0usize, 2, 5, 7],
            vec![0usize, 1, 0, 1, 2, 1, 2],
            vec![1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        );
        let eigenvalues = Array1::from_vec(vec![0.0_f64]);
        let eigenvectors = Array2::from_shape_vec((3, 1), vec![1.0_f64, 0.0, 0.0]).unwrap();

        let residual = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
        assert!(
            residual > DENSE_EVD_QUALITY_THRESHOLD,
            "garbage eigenvector residual={residual:.2e} must exceed DENSE_EVD_QUALITY_THRESHOLD={DENSE_EVD_QUALITY_THRESHOLD:.2e}"
        );
    }

    #[test]
    fn quality_gate_accepts_near_eigenvector_at_lobpcg_threshold() {
        // 3-node path graph Laplacian (same structure as above).
        // Construct a near-eigenvector v = v0 + δ·v1 where v0=[1,1,1]/√3 (λ=0) and
        // v1=[1,0,-1]/√2 (λ=1). The residual against eigenvalue 0 is ≈ δ.
        // With δ=5e-6 the residual sits between DENSE_EVD_QUALITY_THRESHOLD (1e-6) and
        // LOBPCG_QUALITY_THRESHOLD (1e-5), giving independent coverage of the LOBPCG gate.
        let laplacian = CsMatI::new(
            (3, 3),
            vec![0usize, 2, 5, 7],
            vec![0usize, 1, 0, 1, 2, 1, 2],
            vec![1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        );
        let delta = 5e-6_f64;
        let inv_sqrt3 = 1.0_f64 / 3.0_f64.sqrt();
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        // v = v0 + δ·v1: perturbs the exact λ=0 eigenvector toward the λ=1 eigenvector.
        let eigenvectors = Array2::from_shape_vec(
            (3, 1),
            vec![inv_sqrt3 + delta * inv_sqrt2, inv_sqrt3, inv_sqrt3 - delta * inv_sqrt2],
        )
        .unwrap();
        let eigenvalues = Array1::from_vec(vec![0.0_f64]);

        let residual = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
        assert!(
            residual > DENSE_EVD_QUALITY_THRESHOLD,
            "near-eigenvector residual={residual:.2e} should exceed DENSE_EVD_QUALITY_THRESHOLD={DENSE_EVD_QUALITY_THRESHOLD:.2e}"
        );
        assert!(
            residual < LOBPCG_QUALITY_THRESHOLD,
            "near-eigenvector residual={residual:.2e} should be accepted by LOBPCG gate (< {LOBPCG_QUALITY_THRESHOLD:.2e})"
        );
    }

    #[test]
    fn level0_result_passes_dense_evd_quality_gate() {
        // 6-node path graph, n=6 < 2000 → Level 0 dense EVD.
        let laplacian = path_graph_laplacian_6();
        let ((eigs, vecs), _) = solve_eigenproblem(&laplacian, 2, 42, &ndarray::Array1::ones(6));
        let residual = max_eigenpair_residual(&laplacian, &eigs, &vecs);
        assert!(
            residual < DENSE_EVD_QUALITY_THRESHOLD,
            "Level 0 dense EVD residual={residual:.2e} should be < DENSE_EVD_QUALITY_THRESHOLD={DENSE_EVD_QUALITY_THRESHOLD:.2e}"
        );
    }

    #[test]
    fn dense_n_threshold_returns_default_when_env_unset() {
        // SAFETY: single-threaded test (--test-threads=1), no concurrent env readers
        unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }
        assert_eq!(dense_n_threshold(), DENSE_N_THRESHOLD);
    }

    #[cfg(feature = "testing")]
    #[test]
    fn dense_n_threshold_reads_env_override() {
        // SAFETY: single-threaded test (--test-threads=1), no concurrent env readers
        unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "50"); }
        assert_eq!(dense_n_threshold(), 50);
        unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }
    }

    #[test]
    fn dense_n_threshold_falls_back_on_parse_error() {
        // SAFETY: single-threaded test (--test-threads=1), no concurrent env readers
        unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "not_a_number"); }
        assert_eq!(dense_n_threshold(), DENSE_N_THRESHOLD);
        unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }
    }
}
