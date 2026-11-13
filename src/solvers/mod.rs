//! Eigensolver escalation chain for the symmetric normalized Laplacian.
//!
//! The chain tries progressively more robust (and expensive) solvers in order,
//! accepting a result as soon as the max eigenpair residual
//! `‖Lv − λv‖ / ‖v‖` falls below the level's quality threshold.
//! Level 5 (forced dense EVD) is the last resort: it runs for any `n` that fits the
//! dense memory budget, and its result is held to the same residual gate as Level 0.
//!
//! ## Escalation Chain
//!
//! | Level | Condition | Solver | Quality Threshold |
//! |-------|-----------|--------|-------------------|
//! | 0 | n < 2000 | Dense EVD (`dense`) via faer | 1e-6 |
//! | 1 | n ≥ 2000 | LOBPCG (`lobpcg`) without regularization | 2e-5 |
//! | 2 | Level 1 quality failed | Shift-invert LOBPCG (`sinv`) via faer Cholesky | 1e-6 |
//! | 3 | Level 2 quality failed | LOBPCG (`lobpcg`) with ε·I regularization | 2e-5 |
//! | 4 | Level 3 quality failed | Randomized SVD (`rsvd`) via 2I−L trick | 1e-2 |
//! | 5 | Level 4 quality failed | Forced dense EVD | 1e-6 |
//!
//! A level "fails" when its residual exceeds the threshold, when it returns no result
//! (e.g., Cholesky factorization failure in Level 2), or when the dense matrix it needs
//! (Levels 0 and 4) would exceed the dense memory budget. Each of these escalates to the
//! next level.
//!
//! The chain never panics. It returns an error only from Level 5:
//! `DenseBudgetExceeded` when the dense `n × n` matrix does not fit the budget, and
//! `ConvergenceFailure` when the dense result is not finite or misses its residual gate.

mod dense;
pub(crate) mod lobpcg;
mod nd16_bridge;
pub(crate) mod quality;
pub(crate) mod rsvd;
pub(crate) mod sinv;

pub use dense::dense_evd;

use crate::SpectralError;
use crate::config::{ComputeMode, SpectralInitConfig};
use crate::operator::CsrOperator;
use ndarray::{Array1, Array2};
use quality::{
    DENSE_EVD_QUALITY_THRESHOLD, LOBPCG_QUALITY_THRESHOLD, RSVD_QUALITY_THRESHOLD,
    SINV_LOBPCG_QUALITY_THRESHOLD, max_eigenpair_residual,
};
use sprs::CsMatI;
use std::time::{Duration, Instant};

/// Eigendecomposition result: (eigenvalues of shape `[k]`, eigenvectors of shape `[n, k]`).
pub type EigenResult = (Array1<f64>, Array2<f64>);

/// Graphs with n < DENSE_N_THRESHOLD use Level 0 (dense EVD) directly.
const DENSE_N_THRESHOLD: usize = 2000;

/// Level of the solver escalation chain.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SolverLevel {
    /// Level 0: dense EVD, tried first for small problems.
    DenseEvd = 0,
    /// Level 1: LOBPCG without regularization.
    Lobpcg = 1,
    /// Level 2: shift-and-invert LOBPCG.
    SinvLobpcg = 2,
    /// Level 3: LOBPCG with ε·I regularization.
    LobpcgRegularized = 3,
    /// Level 4: randomized SVD.
    Rsvd = 4,
    /// Level 5: forced dense EVD.
    ForcedDense = 5,
}

impl SolverLevel {
    /// Number of levels in the chain. `ForcedDense` must stay the last variant.
    pub const COUNT: usize = SolverLevel::ForcedDense as usize + 1;

    /// Human-readable solver name.
    pub fn name(self) -> &'static str {
        match self {
            SolverLevel::DenseEvd => "Dense EVD",
            SolverLevel::Lobpcg => "LOBPCG",
            SolverLevel::SinvLobpcg => "SINV LOBPCG",
            SolverLevel::LobpcgRegularized => "LOBPCG+reg",
            SolverLevel::Rsvd => "rSVD",
            SolverLevel::ForcedDense => "Forced Dense EVD",
        }
    }
}

/// Residual thresholds of the escalation chain. A level's result is accepted when its
/// max eigenpair residual is strictly below the level's gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityGates {
    /// Gate for Levels 0 and 5 (dense EVD).
    pub dense: f64,
    /// Gate for Levels 1 and 3 (LOBPCG, plain and regularized).
    pub lobpcg: f64,
    /// Gate for Level 2 (shift-and-invert LOBPCG).
    pub sinv: f64,
    /// Gate for Level 4 (randomized SVD).
    pub rsvd: f64,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            dense: DENSE_EVD_QUALITY_THRESHOLD,
            lobpcg: LOBPCG_QUALITY_THRESHOLD,
            sinv: SINV_LOBPCG_QUALITY_THRESHOLD,
            rsvd: RSVD_QUALITY_THRESHOLD,
        }
    }
}

/// Tunable parameters of the solver escalation chain.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverParams {
    /// Problems with `n < dense_n_threshold` start at Level 0 (dense EVD).
    pub dense_n_threshold: usize,
    /// Oversampling of the randomized SVD subspace; `None` selects the n-dependent default.
    pub rsvd_oversampling: Option<usize>,
    /// Number of power-iteration passes of the randomized SVD.
    pub rsvd_nbiter: usize,
    /// Per-level residual acceptance thresholds.
    pub gates: QualityGates,
    /// Largest dense matrix, in bytes, a solver may allocate.
    pub dense_memory_budget: u64,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            dense_n_threshold: DENSE_N_THRESHOLD,
            rsvd_oversampling: None,
            rsvd_nbiter: rsvd::DEFAULT_NBITER,
            gates: QualityGates::default(),
            dense_memory_budget: SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET,
        }
    }
}

impl SolverParams {
    /// The parameters `spectral_init` runs the chain with for `config`.
    pub fn from_config(config: &SpectralInitConfig) -> Self {
        Self {
            dense_memory_budget: config.dense_memory_budget(),
            ..Self::default()
        }
    }
}

/// Bytes a dense `rows × cols` f64 matrix occupies, saturating at `u64::MAX`.
pub(crate) fn dense_bytes(rows: usize, cols: usize) -> u64 {
    (rows as u64)
        .checked_mul(cols as u64)
        .and_then(|cells| cells.checked_mul(8))
        .unwrap_or(u64::MAX)
}

/// Errors with `DenseBudgetExceeded` unless a dense `rows × cols` f64 matrix fits in
/// `budget_bytes`.
pub(crate) fn check_dense_budget(
    rows: usize,
    cols: usize,
    budget_bytes: u64,
) -> Result<(), SpectralError> {
    let required_bytes = dense_bytes(rows, cols);
    if required_bytes > budget_bytes {
        return Err(SpectralError::DenseBudgetExceeded {
            required_bytes,
            budget_bytes,
        });
    }
    Ok(())
}

/// What the escalation chain did to produce a result.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveReport {
    /// Level whose result was returned.
    pub level: SolverLevel,
    /// Max eigenpair residual of the returned result.
    pub residual: f64,
    /// Wall-clock time spent in each level that ran, indexed by level number.
    pub level_elapsed: [Option<Duration>; SolverLevel::COUNT],
}

impl SolveReport {
    /// Emits one `log::debug!` line describing the solve of a problem of size `n`.
    pub(crate) fn log(&self, context: &str, n: usize) {
        log::debug!(
            "[spectral] {context}: n={n}, level {} ({}), max_residual={:.2e}, elapsed per level={:?}",
            self.level as u8,
            self.level.name(),
            self.residual,
            self.level_elapsed
        );
    }
}

/// Enforce the mathematical PSD contract on solver output.
///
/// The symmetric normalized Laplacian has eigenvalues in [0, 2] by construction.
/// Floating-point arithmetic in any solver (faer dense EVD, LOBPCG Rayleigh-Ritz,
/// rSVD 2-λ cancellation, sinv 1/μ-ε recovery) can produce eigenvalues slightly
/// outside this range. This function restores the mathematical invariant at the
/// single exit point of the solver chain — callers of `solve_eigenproblem` may
/// assume λ ∈ [0.0, 2.0] exactly.
///
/// This matches Python's `np.maximum(eigenvalues, 0.0)` convention in
/// `tests/generate_fixtures.py` and extends it to the upper bound.
fn enforce_psd_contract(result: EigenResult) -> EigenResult {
    let (eigenvalues, eigenvectors) = result;
    let clamped = eigenvalues.mapv(|lambda| lambda.clamp(0.0, 2.0));
    (clamped, eigenvectors)
}

/// Packages an accepted eigenpair set with the report describing how it was obtained.
fn accepted(
    result: EigenResult,
    level: SolverLevel,
    residual: f64,
    level_elapsed: [Option<Duration>; SolverLevel::COUNT],
) -> (EigenResult, SolveReport) {
    (
        enforce_psd_contract(result),
        SolveReport {
            level,
            residual,
            level_elapsed,
        },
    )
}

/// Core solver escalation chain, parameterized over the `LinearOperator` used for LOBPCG.
///
/// Returns `n_components+1` eigenpairs (including the trivial λ≈0 vector at index 0)
/// so that the caller can apply `select_eigenvectors` uniformly across all solver paths.
///
/// Level 0: Dense EVD (n < `params.dense_n_threshold`, O(n³)).
/// Level 1: LOBPCG without regularization.
/// Level 2: Shift-and-invert LOBPCG via sparse Cholesky.
/// Level 3: LOBPCG with ε·I regularization.
/// Level 4: Randomized SVD via 2I-L trick.
/// Level 5: Forced dense EVD.
///
/// # Errors
///
/// Only Level 5 produces errors; see the module documentation.
fn solve_eigenproblem_with_op<O: crate::operator::LinearOperator>(
    laplacian: &CsMatI<f64, usize>,
    op: O,
    n_components: usize,
    seed: u64,
    sqrt_deg: &ndarray::Array1<f64>,
    params: &SolverParams,
) -> Result<(EigenResult, SolveReport), SpectralError> {
    let n = laplacian.rows();
    let gates = &params.gates;
    let mut level_elapsed = [None; SolverLevel::COUNT];

    // Level 0: Dense EVD — exact, used for small n where O(n³) is acceptable.
    if n < params.dense_n_threshold {
        let started = Instant::now();
        let l0_result = dense_evd(laplacian, n_components + 1, params.dense_memory_budget);
        level_elapsed[SolverLevel::DenseEvd as usize] = Some(started.elapsed());
        match l0_result {
            Ok((eigs, vecs)) => {
                let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
                if quality < gates.dense {
                    log::debug!(
                        "[spectral] Level 0 (dense EVD) succeeded (n={n}, max_residual={quality:.2e})"
                    );
                    return Ok(accepted(
                        (eigs, vecs),
                        SolverLevel::DenseEvd,
                        quality,
                        level_elapsed,
                    ));
                }
                log::debug!(
                    "[spectral] Level 0 (dense EVD) poor quality \
                     (max_residual={quality:.2e}), escalating to Level 1"
                );
            }
            Err(e @ SpectralError::DenseBudgetExceeded { .. }) => {
                log::debug!("[spectral] Level 0 (dense EVD) skipped ({e}), escalating to Level 1");
            }
            Err(e) => {
                log::warn!("[spectral] Level 0 (dense EVD) failed ({e}), escalating to Level 1");
            }
        }
    }

    // Level 1: LOBPCG without regularization.
    let started = Instant::now();
    let l1 = lobpcg::lobpcg_solve(&op, n_components, seed, false, sqrt_deg);
    level_elapsed[SolverLevel::Lobpcg as usize] = Some(started.elapsed());
    if let Some(((eigs, vecs), _restarts)) = l1 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < gates.lobpcg {
            log::debug!(
                "[spectral] Level 1 (LOBPCG) succeeded (n={n}, max_residual={quality:.2e})"
            );
            return Ok(accepted(
                (eigs, vecs),
                SolverLevel::Lobpcg,
                quality,
                level_elapsed,
            ));
        }
        log::debug!(
            "[spectral] Level 1 (LOBPCG) poor quality \
             (max_residual={quality:.2e}), escalating to Level 2"
        );
    } else {
        log::debug!("[spectral] Level 1 (LOBPCG) produced no result, escalating to Level 2");
    }

    // Level 2: Shift-and-invert LOBPCG — near-exact, handles small eigengap.
    // sprs_csc_to_faer converts the Laplacian to faer CSC and adds SINV_SHIFT to the
    // diagonal, producing M = L + εI. sp_cholesky factorizes M; if it fails (e.g. M
    // is not SPD due to numerical degenerate edges) we escalate to Level 3.
    let started = Instant::now();
    let l2 = sinv::lobpcg_sinv_solve(laplacian, n_components, seed, sqrt_deg);
    level_elapsed[SolverLevel::SinvLobpcg as usize] = Some(started.elapsed());
    if let Some((eigs, vecs)) = l2 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < gates.sinv {
            log::debug!(
                "[spectral] Level 2 (sinv LOBPCG) succeeded (n={n}, max_residual={quality:.2e})"
            );
            return Ok(accepted(
                (eigs, vecs),
                SolverLevel::SinvLobpcg,
                quality,
                level_elapsed,
            ));
        }
        log::debug!(
            "[spectral] Level 2 (sinv LOBPCG) poor quality \
             (max_residual={quality:.2e}), escalating to Level 3"
        );
    } else {
        log::debug!("[spectral] Level 2 (sinv LOBPCG) produced no result, escalating to Level 3");
    }

    // Level 3: LOBPCG with ε·I regularization — widens eigengap.
    // lobpcg_solve(regularize=true) applies Rayleigh-Ritz refinement internally
    // (G = X^T L X against the true Laplacian), so eigs are exact Rayleigh quotients
    // and can be passed directly to the residual check.
    let started = Instant::now();
    let l3 = lobpcg::lobpcg_solve(&op, n_components, seed, true, sqrt_deg);
    level_elapsed[SolverLevel::LobpcgRegularized as usize] = Some(started.elapsed());
    if let Some(((eigs, vecs), _restarts)) = l3 {
        let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
        if quality < gates.lobpcg {
            log::debug!(
                "[spectral] Level 3 (LOBPCG+reg) succeeded (n={n}, max_residual={quality:.2e})"
            );
            return Ok(accepted(
                (eigs, vecs),
                SolverLevel::LobpcgRegularized,
                quality,
                level_elapsed,
            ));
        }
        log::debug!(
            "[spectral] Level 3 (LOBPCG+reg) poor quality \
             (max_residual={quality:.2e}), escalating to Level 4"
        );
    } else {
        log::debug!("[spectral] Level 3 (LOBPCG+reg) produced no result, escalating to Level 4");
    }

    // Level 4: Randomized SVD via 2I-L trick. Produces no result when its dense sketch
    // would exceed the dense memory budget.
    let started = Instant::now();
    let l4 = rsvd::rsvd_solve(laplacian, n_components, seed, params);
    level_elapsed[SolverLevel::Rsvd as usize] = Some(started.elapsed());
    match l4 {
        Some((eigs, vecs)) => {
            let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
            if quality < gates.rsvd {
                log::debug!(
                    "[spectral] Level 4 (rSVD) succeeded (n={n}, max_residual={quality:.2e})"
                );
                return Ok(accepted(
                    (eigs, vecs),
                    SolverLevel::Rsvd,
                    quality,
                    level_elapsed,
                ));
            }
            log::debug!(
                "[spectral] Level 4 (rSVD) poor quality (max_residual={quality:.2e}), \
                 escalating to Level 5"
            );
        }
        None => {
            log::debug!("[spectral] Level 4 (rSVD) produced no result, escalating to Level 5");
        }
    }

    // Level 5: Forced dense EVD — O(n³) last resort. Over budget it reports
    // DenseBudgetExceeded; a result that is not finite or misses the dense gate is a
    // ConvergenceFailure rather than a silently wrong embedding.
    log::debug!("[spectral] Level 5 (forced dense EVD) (n={n})");
    let started = Instant::now();
    let (eigs, vecs) = dense_evd(laplacian, n_components + 1, params.dense_memory_budget)?;
    level_elapsed[SolverLevel::ForcedDense as usize] = Some(started.elapsed());
    let quality = max_eigenpair_residual(laplacian, &eigs, &vecs);
    let finite = eigs.iter().all(|v| v.is_finite()) && vecs.iter().all(|v| v.is_finite());
    if finite && quality < gates.dense {
        return Ok(accepted(
            (eigs, vecs),
            SolverLevel::ForcedDense,
            quality,
            level_elapsed,
        ));
    }
    log::debug!("[spectral] Level 5 (forced dense EVD) rejected (max_residual={quality:.2e})");
    Err(SpectralError::ConvergenceFailure { n })
}

/// Solver escalation chain with runtime operator dispatch.
///
/// On x86_64, `ComputeMode::RustNative` routes LOBPCG levels through `CsrOperatorSimd`
/// when the CPU supports AVX2 and FMA and the Laplacian's column indices fit 32 bits;
/// all other configurations use scalar `CsrOperator`.
/// `ComputeMode::PythonCompat` always uses the scalar path regardless of CPU features.
///
/// Returns the `n_components + 1` smallest eigenpairs together with a [`SolveReport`].
///
/// # Errors
///
/// - `DenseBudgetExceeded` — every earlier level failed and the dense `n × n` matrix of
///   the forced dense EVD does not fit `params.dense_memory_budget`.
/// - `ConvergenceFailure` — the forced dense EVD produced a result that is not finite or
///   misses its residual gate.
/// - `TooFewNodes` — the problem has fewer than `n_components + 1` rows.
pub fn solve_eigenproblem(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    sqrt_deg: &ndarray::Array1<f64>,
    compute_mode: ComputeMode,
    params: &SolverParams,
) -> Result<(EigenResult, SolveReport), SpectralError> {
    #[cfg(target_arch = "x86_64")]
    if matches!(compute_mode, ComputeMode::RustNative)
        && let Some(op) = crate::operator::CsrOperatorSimd::new(laplacian)
    {
        return solve_eigenproblem_with_op(laplacian, op, n_components, seed, sqrt_deg, params);
    }
    // Other architectures have no SIMD operator: both modes use the scalar one.
    #[cfg(not(target_arch = "x86_64"))]
    let _ = compute_mode;
    let op = CsrOperator(laplacian);
    solve_eigenproblem_with_op(laplacian, op, n_components, seed, sqrt_deg, params)
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
                0, 1, 2, // row 1
                1, 2, 3, // row 2
                2, 3, 4, // row 3
                3, 4, 5, // row 4
                4, 5, // row 5
            ],
            vec![
                1.0_f64, -1.0, // row 0
                -1.0, 2.0, -1.0, // row 1
                -1.0, 2.0, -1.0, // row 2
                -1.0, 2.0, -1.0, // row 3
                -1.0, 2.0, -1.0, // row 4
                -1.0, 1.0, // row 5
            ],
        )
    }

    #[test]
    fn solve_eigenproblem_eigenvalues_nonneg_and_sorted() {
        let laplacian = path_graph_laplacian_6();
        let ((eigvals, _), _) = solve_eigenproblem(
            &laplacian,
            2,
            42,
            &ndarray::Array1::ones(6),
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        )
        .expect("solve_eigenproblem failed");

        // All eigenvalues must be non-negative
        for &v in eigvals.iter() {
            assert!(v >= -1e-10, "eigenvalue is negative: {v}");
        }

        // Eigenvalues must be sorted ascending
        for i in 1..eigvals.len() {
            assert!(
                eigvals[i] >= eigvals[i - 1] - 1e-10,
                "eigenvalues not sorted: eigvals[{}]={} > eigvals[{}]={}",
                i - 1,
                eigvals[i - 1],
                i,
                eigvals[i]
            );
        }
    }

    #[test]
    fn solve_eigenproblem_returns_k_plus_one_pairs() {
        // 6-node path graph (n=6 < 2000 → Level 0 dense EVD)
        let ((eigs, vecs), _) = solve_eigenproblem(
            &path_graph_laplacian_6(),
            2,
            42,
            &ndarray::Array1::ones(6),
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        )
        .expect("solve_eigenproblem failed");
        assert_eq!(eigs.len(), 3, "expected n_components+1=3 eigenvalues");
        assert_eq!(
            vecs.shape(),
            &[6, 3],
            "expected [n, n_components+1] = [6, 3] eigenvectors"
        );
    }

    /// Build a sparse diagonal n×n CSR Laplacian with eigenvalues [0, 1/(n-1), ..., 1].
    ///
    /// Diagonal entries approximate a Laplacian with a zero eigenvalue at index 0.
    fn diagonal_laplacian(n: usize) -> CsMatI<f64, usize> {
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        let data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1).max(1) as f64).collect();
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

        let ((eigvals, eigvecs), _) = solve_eigenproblem(
            &laplacian,
            n_components,
            42,
            &ndarray::Array1::ones(n),
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        )
        .expect("solve_eigenproblem failed");

        assert_eq!(
            eigvals.len(),
            n_components + 1,
            "expected n_components+1 eigenvalues"
        );
        assert_eq!(
            eigvecs.shape(),
            &[n, n_components + 1],
            "expected [n, n_components+1] shape"
        );

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
        let (_, report) = solve_eigenproblem(
            &laplacian,
            2,
            42,
            &ndarray::Array1::ones(n),
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        )
        .expect("solve_eigenproblem failed");
        let level = report.level as u8;
        assert!(
            (1..=5).contains(&level),
            "expected level in {{1,2,3,4,5}}, got {level}"
        );
        assert!(
            report.level_elapsed[SolverLevel::DenseEvd as usize].is_none(),
            "Level 0 must not run above the dense threshold"
        );
        assert!(
            report.level_elapsed[level as usize].is_some(),
            "the accepted level must have a recorded duration"
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
            vec![
                inv_sqrt3 + delta * inv_sqrt2,
                inv_sqrt3,
                inv_sqrt3 - delta * inv_sqrt2,
            ],
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
        let ((eigs, vecs), report) = solve_eigenproblem(
            &laplacian,
            2,
            42,
            &ndarray::Array1::ones(6),
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        )
        .expect("solve_eigenproblem failed");
        assert_eq!(report.level, SolverLevel::DenseEvd);
        let residual = max_eigenpair_residual(&laplacian, &eigs, &vecs);
        assert!(
            residual < DENSE_EVD_QUALITY_THRESHOLD,
            "Level 0 dense EVD residual={residual:.2e} should be < DENSE_EVD_QUALITY_THRESHOLD={DENSE_EVD_QUALITY_THRESHOLD:.2e}"
        );
    }

    #[test]
    fn default_params_match_the_documented_chain() {
        let params = SolverParams::default();
        assert_eq!(params.dense_n_threshold, DENSE_N_THRESHOLD);
        assert_eq!(params.rsvd_oversampling, None);
        assert_eq!(params.rsvd_nbiter, 2);
        assert_eq!(params.gates.dense, DENSE_EVD_QUALITY_THRESHOLD);
        assert_eq!(params.gates.lobpcg, LOBPCG_QUALITY_THRESHOLD);
        assert_eq!(params.gates.sinv, SINV_LOBPCG_QUALITY_THRESHOLD);
        assert_eq!(params.gates.rsvd, RSVD_QUALITY_THRESHOLD);
        assert_eq!(
            params.dense_memory_budget,
            SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET
        );
    }

    #[test]
    fn from_config_carries_the_dense_memory_budget() {
        let config = SpectralInitConfig::new(2).with_dense_memory_budget(12_345);
        let params = SolverParams::from_config(&config);
        assert_eq!(params.dense_memory_budget, 12_345);
        assert_eq!(
            SolverParams {
                dense_memory_budget: SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET,
                ..params
            },
            SolverParams::default()
        );
    }

    #[test]
    fn dense_bytes_saturates_instead_of_overflowing() {
        assert_eq!(dense_bytes(3, 5), 120);
        assert_eq!(dense_bytes(0, 5), 0);
        assert_eq!(dense_bytes(usize::MAX, usize::MAX), u64::MAX);
        assert_eq!(dense_bytes(usize::MAX, 1), u64::MAX);
    }

    #[test]
    fn check_dense_budget_reports_both_sizes() {
        assert_eq!(check_dense_budget(10, 10, 800), Ok(()));
        assert_eq!(
            check_dense_budget(10, 10, 799),
            Err(SpectralError::DenseBudgetExceeded {
                required_bytes: 800,
                budget_bytes: 799
            })
        );
    }

    #[test]
    fn dense_n_threshold_zero_skips_level_0() {
        // n=6 would use Level 0 by default; a zero threshold must route it to Level 1+.
        let params = SolverParams {
            dense_n_threshold: 0,
            ..SolverParams::default()
        };
        let (_, report) = solve_eigenproblem(
            &path_graph_laplacian_6(),
            2,
            42,
            &ndarray::Array1::ones(6),
            ComputeMode::PythonCompat,
            &params,
        )
        .expect("solve_eigenproblem failed");
        assert!(
            report.level >= SolverLevel::Lobpcg,
            "expected Level 1 or later, got {:?}",
            report.level
        );
        assert!(report.level_elapsed[SolverLevel::DenseEvd as usize].is_none());
    }

    #[test]
    fn solver_level_discriminants_and_names() {
        let levels = [
            (SolverLevel::DenseEvd, "Dense EVD"),
            (SolverLevel::Lobpcg, "LOBPCG"),
            (SolverLevel::SinvLobpcg, "SINV LOBPCG"),
            (SolverLevel::LobpcgRegularized, "LOBPCG+reg"),
            (SolverLevel::Rsvd, "rSVD"),
            (SolverLevel::ForcedDense, "Forced Dense EVD"),
        ];
        assert_eq!(levels.len(), SolverLevel::COUNT);
        for (expected, (level, name)) in levels.iter().enumerate() {
            assert_eq!(*level as usize, expected);
            assert_eq!(level.name(), *name);
        }
        assert!(SolverLevel::DenseEvd < SolverLevel::ForcedDense);
    }

    #[test]
    fn test_enforce_psd_contract_clamps_negative() {
        // Simulates dense EVD noise: smallest eigenvalue slightly negative
        let raw_eigenvalues = ndarray::arr1(&[-1.657e-10_f64, 2.668e-3, 7.235e-3]);
        let dummy_vecs = ndarray::Array2::zeros((10, 3));
        let contracted = enforce_psd_contract((raw_eigenvalues, dummy_vecs.clone()));
        assert_eq!(
            contracted.0[0], 0.0,
            "smallest eigenvalue must be clamped from -1.657e-10 to 0.0"
        );
        assert!(
            contracted.0[1] > 0.0 && contracted.0[2] > 0.0,
            "non-trivial eigenvalues must be unchanged"
        );
        assert_eq!(
            contracted.1, dummy_vecs,
            "eigenvectors must pass through unmodified"
        );
    }

    #[test]
    fn test_enforce_psd_contract_clamps_above_2() {
        // Simulates rSVD upper-bound noise near the bipartite limit
        let raw_eigenvalues = ndarray::arr1(&[0.0_f64, 1.5, 2.0 + 1.3e-15]);
        let dummy_vecs = ndarray::Array2::zeros((10, 3));
        let contracted = enforce_psd_contract((raw_eigenvalues, dummy_vecs.clone()));
        assert_eq!(
            contracted.0[2], 2.0,
            "eigenvalue above 2.0 must be clamped to exactly 2.0"
        );
        assert_eq!(
            contracted.1, dummy_vecs,
            "eigenvectors must pass through unmodified"
        );
    }

    #[test]
    fn test_enforce_psd_contract_passes_clean_eigenvalues_unchanged() {
        // Eigenvalues already in range must not be modified
        let raw_eigenvalues = ndarray::arr1(&[0.0_f64, 0.5, 1.0]);
        let dummy_vecs = ndarray::Array2::zeros((10, 3));
        let contracted = enforce_psd_contract((raw_eigenvalues.clone(), dummy_vecs));
        assert_eq!(
            contracted.0, raw_eigenvalues,
            "clean eigenvalues must pass through unmodified"
        );
    }

    #[test]
    fn test_enforce_psd_contract_clamps_rsvd_cancellation() {
        // Simulates rSVD 2 - λ_M cancellation producing ~-1e-8
        let raw_eigenvalues = ndarray::arr1(&[-8.2e-9_f64, 1.1e-3, 3.4e-3]);
        let dummy_vecs = ndarray::Array2::zeros((10, 3));
        let contracted = enforce_psd_contract((raw_eigenvalues, dummy_vecs));
        assert_eq!(
            contracted.0[0], 0.0,
            "rSVD cancellation artifact must be clamped to 0.0"
        );
    }
}
