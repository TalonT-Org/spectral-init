//! Quality metrics, threshold constants, and assessment data structures.
//!
//! This module is the single canonical home for:
//! - Eigensolver quality thresholds (used in the solver escalation chain)
//! - Accuracy metric functions (residual, orthogonality, eigenvalue bounds)
//! - Parity metric functions (comparing Rust vs Python reference outputs)
//! - Diagnostic functions (spectral gap, condition number, tolerance margin)
//! - Data structures for structured metric reporting (behind `#[cfg(feature = "testing")]`)

use faer::{Mat as FaerMat, Side};
use faer::linalg::solvers::SelfAdjointEigen;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sprs::CsMatI;

// ─── Threshold constants ──────────────────────────────────────────────────────

/// Maximum acceptable max-residual from rSVD before falling to Level 5.
/// rSVD with 2 power iterations typically achieves 1e-4 to 1e-6 on well-conditioned
/// graphs; 1e-2 accepts all such results while correctly escalating pathological cases.
pub const RSVD_QUALITY_THRESHOLD: f64 = 1e-2;

/// Maximum acceptable max-residual from dense EVD (Levels 0 and 5).
/// Dense EVD via faer is numerically exact (machine precision); 1e-6 leaves
/// a generous margin while catching any pathological faer regression.
pub const DENSE_EVD_QUALITY_THRESHOLD: f64 = 1e-6;

/// Maximum acceptable max-residual from LOBPCG (Levels 1 and 3).
/// Set to 1e-5 (Issue #92 REQ-TOL-002): matches the solver's `tol = 1e-5` and is
/// consistently achievable with ChFSI preconditioning, while being tighter than rSVD (1e-2).
pub const LOBPCG_QUALITY_THRESHOLD: f64 = 1e-5;

/// Maximum acceptable max-residual from shift-and-invert LOBPCG (Level 2).
/// Sinv achieves near-exact results (like dense EVD); 1e-6 accepts all
/// well-converged results while correctly escalating pathological graphs.
pub const SINV_LOBPCG_QUALITY_THRESHOLD: f64 = 1e-6;

/// LOBPCG internal per-vector convergence tolerance.
/// Set to 1e-5 (Issue #92 REQ-TOL-003): matches the solver's `tol` and is achievable
/// with ChFSI-filtered starting subspace for well-conditioned graphs.
pub const LOBPCG_ACCEPT_TOL: f64 = 1e-5;

/// Shift-invert LOBPCG internal convergence tolerance.
pub const SINV_ACCEPT_TOL: f64 = 1e-6;

/// Minimum eigenvalue gap to treat eigenpairs as distinct; below this use subspace comparison.
pub const DEGENERATE_GAP_THRESHOLD: f64 = 1e-6;

/// Minimum acceptable subspace Gram determinant for quality assessment.
pub const SUBSPACE_GRAM_DET_THRESHOLD: f64 = 0.95;

// ─── Accuracy metric functions ────────────────────────────────────────────────

/// Computes the relative residual `‖L·v − λ·v‖₂ / max(‖v‖₂, ε)` for a single eigenpair.
pub fn eigenpair_residual(
    laplacian: &CsMatI<f64, usize>,
    eigenvector: &Array1<f64>,
    eigenvalue: f64,
) -> f64 {
    let lv = laplacian * eigenvector;
    let lambda_v = eigenvector.mapv(|x| eigenvalue * x);
    let diff = &lv - &lambda_v;
    let v_norm = eigenvector.dot(eigenvector).sqrt().max(f64::EPSILON);
    diff.dot(&diff).sqrt() / v_norm
}

/// Returns the maximum relative residual `‖L·v − λ·v‖₂ / ‖v‖₂` over all eigenpairs.
///
/// Delegates per-pair computation to [`eigenpair_residual`]. Returns `NaN` if any
/// individual residual is `NaN`.
pub fn max_eigenpair_residual(
    laplacian: &CsMatI<f64, usize>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> f64 {
    eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &lambda)| {
            let v = eigenvectors.column(i).to_owned();
            eigenpair_residual(laplacian, &v, lambda)
        })
        .fold(0.0_f64, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Computes `‖VᵀV − I‖_F` where `V` has columns `eigenvectors`.
///
/// Returns 0.0 for a perfectly orthonormal set of columns.
pub fn orthogonality_error(eigenvectors: &Array2<f64>) -> f64 {
    let k = eigenvectors.ncols();
    let vtv = eigenvectors.t().dot(eigenvectors);
    let mut diff = vtv;
    for i in 0..k {
        diff[[i, i]] -= 1.0;
    }
    diff.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Checks whether eigenvalues are in range and sorted ascending.
///
/// Returns `(in_range, sorted_ascending)` where:
/// - `in_range`: all `λ ∈ [−tol, 2+tol]`
/// - `sorted_ascending`: `λ[i] ≤ λ[i+1] + tol` for all `i`
pub fn check_eigenvalue_bounds(eigenvalues: &Array1<f64>, tol: f64) -> (bool, bool) {
    assert!(tol >= 0.0, "check_eigenvalue_bounds: tol must be non-negative, got {}", tol);
    let in_range = eigenvalues.iter().all(|&v| v >= -tol && v <= 2.0 + tol);
    let sorted = eigenvalues
        .windows(2)
        .into_iter()
        .all(|w| w[0] <= w[1] + tol);
    (in_range, sorted)
}

/// Computes the separation ratio `min_inter_centroid_distance / max_intra_component_spread`
/// for a disconnected-graph embedding.
///
/// `labels` assigns each row of `embedding` to a component index in `0..n_components`.
/// Returns `f64::INFINITY` when `max_intra == 0.0` (all points collapse to centroids).
pub fn separation_ratio(embedding: ArrayView2<f64>, labels: &[usize]) -> f64 {
    let n = labels.len();
    assert_eq!(embedding.nrows(), n, "embedding rows must match labels length");
    let n_dims = embedding.ncols();

    let n_components = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_components < 2 {
        return f64::INFINITY;
    }

    let mut component_members: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (i, &label) in labels.iter().enumerate() {
        component_members[label].push(i);
    }

    let centroids: Vec<Vec<f64>> = component_members
        .iter()
        .map(|members| {
            let n_c = members.len() as f64;
            let mut c = vec![0.0f64; n_dims];
            for &orig_i in members {
                for d in 0..n_dims {
                    c[d] += embedding[[orig_i, d]];
                }
            }
            c.iter_mut().for_each(|x| *x /= n_c);
            c
        })
        .collect();

    let min_inter = (0..n_components)
        .flat_map(|i| ((i + 1)..n_components).map(move |j| (i, j)))
        .map(|(i, j)| {
            (0..n_dims)
                .map(|d| (centroids[i][d] - centroids[j][d]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min);

    let max_intra = component_members
        .iter()
        .enumerate()
        .map(|(c_idx, members)| {
            members
                .iter()
                .map(|&orig_i| {
                    (0..n_dims)
                        .map(|d| (embedding[[orig_i, d]] - centroids[c_idx][d]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);

    if max_intra > 0.0 {
        min_inter / max_intra
    } else {
        f64::INFINITY
    }
}

// ─── Parity metric functions ──────────────────────────────────────────────────

/// Element-wise `|computed[i] − reference[i]|`.
///
/// # Panics
/// Panics if `computed` and `reference` have different lengths.
pub fn eigenvalue_abs_errors(computed: &Array1<f64>, reference: &Array1<f64>) -> Array1<f64> {
    assert_eq!(
        computed.len(),
        reference.len(),
        "eigenvalue_abs_errors: lengths must match ({} vs {})",
        computed.len(),
        reference.len()
    );
    (computed - reference).mapv(f64::abs)
}

/// Computes `|det(G)|` for the 2×2 cross-Gram matrix of two n×2 subspaces.
///
/// Each column is normalized by its L2 norm before building the cross-Gram matrix:
/// `G[i,j] = dot(r_i / ‖r_i‖, u_j / ‖u_j‖)`.
/// Returns `|a·d − b·c|` where `[[a,b],[c,d]]` is the normalized 2×2 Gram matrix.
pub fn subspace_gram_det(u: ArrayView2<f64>, r: ArrayView2<f64>) -> f64 {
    assert_eq!(u.ncols(), 2, "subspace_gram_det: u must have exactly 2 columns");
    assert_eq!(r.ncols(), 2, "subspace_gram_det: r must have exactly 2 columns");
    assert_eq!(u.nrows(), r.nrows(), "subspace_gram_det: u and r must have the same number of rows");

    let norm = |v: ArrayView1<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    let dot = |a: ArrayView1<f64>, b: ArrayView1<f64>| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    let v1 = u.column(0);
    let v2 = u.column(1);
    let r1 = r.column(0);
    let r2 = r.column(1);

    let n1 = norm(v1);
    let n2 = norm(v2);
    let nr1 = norm(r1);
    let nr2 = norm(r2);

    let a = dot(r1, v1) / (nr1 * n1);
    let b = dot(r1, v2) / (nr1 * n2);
    let c = dot(r2, v1) / (nr2 * n1);
    let d = dot(r2, v2) / (nr2 * n2);

    (a * d - b * c).abs()
}

/// k-dimensional generalization of [`subspace_gram_det`].
///
/// Steps:
/// 1. Normalize each column of both inputs by its L2 norm.
/// 2. Compute the k×k cross-Gram matrix `G = reference_norm^T · computed_norm`.
/// 3. Compute the SVD of `G` via the eigendecomposition of `GᵀG` (using faer).
/// 4. Return the product of singular values, which equals `|det(G)|` for square `G`.
pub fn subspace_gram_det_kd(computed: ArrayView2<f64>, reference: ArrayView2<f64>) -> f64 {
    let k = computed.ncols();
    assert_eq!(
        reference.ncols(),
        k,
        "subspace_gram_det_kd: computed and reference must have the same number of columns"
    );
    assert_eq!(
        computed.nrows(),
        reference.nrows(),
        "subspace_gram_det_kd: computed and reference must have the same number of rows"
    );

    // Normalize each column
    let normalize = |mat: ArrayView2<f64>| -> Array2<f64> {
        let mut norm_mat = Array2::<f64>::zeros(mat.dim());
        for (j, col) in mat.columns().into_iter().enumerate() {
            let norm = col.dot(&col).sqrt().max(f64::EPSILON);
            for (i, &v) in col.iter().enumerate() {
                norm_mat[[i, j]] = v / norm;
            }
        }
        norm_mat
    };

    let cn = normalize(computed);
    let rn = normalize(reference);

    // G = rn^T · cn  (k × k cross-Gram matrix)
    let g = rn.t().dot(&cn);

    // G^T G is symmetric PSD; its eigenvalues are the squares of the singular values of G.
    let gtg = g.t().dot(&g);
    let gtg_faer = FaerMat::from_fn(k, k, |i, j| gtg[[i, j]]);
    let eigen = SelfAdjointEigen::new(gtg_faer.as_ref(), Side::Lower)
        .expect("subspace_gram_det_kd: SelfAdjointEigen failed on G^T G");

    // sqrt(product of eigenvalues of G^T G) = product of singular values = |det(G)|
    let mut det_sq = 1.0f64;
    eigen.S().for_each(|&x| det_sq *= x.max(0.0));
    det_sq.sqrt()
}

/// Sign-agnostic maximum column-wise error between `computed` and `reference`.
///
/// For each column, computes `min(‖col_rust − col_ref‖∞, ‖col_rust + col_ref‖∞)`.
/// Returns the maximum such value across all columns.
///
/// Inputs are `f32`; internal arithmetic is in `f64` for precision.
pub fn sign_agnostic_max_error(computed: &Array2<f32>, reference: &Array2<f32>) -> f64 {
    assert_eq!(
        computed.ncols(),
        reference.ncols(),
        "sign_agnostic_max_error: column count mismatch"
    );
    assert_eq!(
        computed.nrows(),
        reference.nrows(),
        "sign_agnostic_max_error: row count mismatch"
    );

    let mut worst = 0.0f64;
    for col in 0..computed.ncols() {
        let r = computed.column(col);
        let rf = reference.column(col);
        let err_pos: f64 = r
            .iter()
            .zip(rf.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        let err_neg: f64 = r
            .iter()
            .zip(rf.iter())
            .map(|(&a, &b)| (a as f64 + b as f64).abs())
            .fold(0.0f64, f64::max);
        worst = worst.max(err_pos.min(err_neg));
    }
    worst
}

// ─── Diagnostic functions ─────────────────────────────────────────────────────

/// Returns `threshold / worst_value`.
///
/// Returns `f64::INFINITY` if `worst_value == 0.0`.
pub fn tolerance_margin(threshold: f64, worst_value: f64) -> f64 {
    if worst_value == 0.0 {
        f64::INFINITY
    } else {
        threshold / worst_value
    }
}

/// Returns `eigenvalues[1] − eigenvalues[0]` (λ₂ − λ₁).
///
/// # Panics
/// Panics if `eigenvalues` has fewer than 2 elements.
pub fn spectral_gap(eigenvalues: &Array1<f64>) -> f64 {
    assert!(eigenvalues.len() >= 2, "spectral_gap: need at least 2 eigenvalues");
    eigenvalues[1] - eigenvalues[0]
}

/// Returns `eigenvalues[last] / eigenvalues[1]` (λ_last / λ₂).
///
/// Returns `f64::INFINITY` if `eigenvalues[1] == 0.0`.
///
/// # Panics
/// Panics if `eigenvalues` has fewer than 2 elements.
pub fn eigenvalue_condition_number(eigenvalues: &Array1<f64>) -> f64 {
    assert!(
        eigenvalues.len() >= 2,
        "eigenvalue_condition_number: need at least 2 eigenvalues"
    );
    if eigenvalues[1] == 0.0 {
        f64::INFINITY
    } else {
        eigenvalues[eigenvalues.len() - 1] / eigenvalues[1]
    }
}

// ─── Data structures (testing feature only) ──────────────────────────────────

#[cfg(feature = "testing")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricResult {
    pub name: String,
    pub dimension: usize,
    pub value: f64,
    pub threshold: f64,
    pub passed: bool,
}

#[cfg(feature = "testing")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssessmentReport {
    pub dataset: String,
    pub n: usize,
    pub metrics: Vec<MetricResult>,
}

#[cfg(feature = "testing")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentMetrics {
    pub generated_at: String,
    pub datasets: Vec<AssessmentReport>,
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-node path Laplacian: [[1,-1,0],[-1,2,-1],[0,-1,1]]
    fn path_laplacian_3() -> CsMatI<f64, usize> {
        CsMatI::new(
            (3, 3),
            vec![0usize, 2, 5, 7],
            vec![0usize, 1, 0, 1, 2, 1, 2],
            vec![1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        )
    }

    // ── Threshold constants ───────────────────────────────────────────────────

    #[test]
    fn t_const_01_all_8_constants_have_required_values() {
        assert_eq!(DENSE_EVD_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(LOBPCG_QUALITY_THRESHOLD, 1e-5_f64);
        assert_eq!(SINV_LOBPCG_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(RSVD_QUALITY_THRESHOLD, 1e-2_f64);
        assert_eq!(LOBPCG_ACCEPT_TOL, 1e-5_f64);
        assert_eq!(SINV_ACCEPT_TOL, 1e-6_f64);
        assert_eq!(DEGENERATE_GAP_THRESHOLD, 1e-6_f64);
        assert_eq!(SUBSPACE_GRAM_DET_THRESHOLD, 0.95_f64);
    }

    #[test]
    fn t_const_02_threshold_ordering() {
        assert_eq!(DENSE_EVD_QUALITY_THRESHOLD, SINV_LOBPCG_QUALITY_THRESHOLD);
        assert!(SINV_LOBPCG_QUALITY_THRESHOLD < LOBPCG_QUALITY_THRESHOLD);
        assert!(LOBPCG_QUALITY_THRESHOLD < RSVD_QUALITY_THRESHOLD);
    }

    // ── Accuracy: eigenpair_residual ─────────────────────────────────────────

    #[test]
    fn t_eresid_01_trivial_eigenvector_near_zero() {
        let laplacian = path_laplacian_3();
        let s = 1.0_f64 / 3.0_f64.sqrt();
        let v = Array1::from_vec(vec![s, s, s]);
        let residual = eigenpair_residual(&laplacian, &v, 0.0);
        assert!(
            residual < 1e-10,
            "trivial eigenvector residual={residual:.2e}, expected < 1e-10"
        );
    }

    #[test]
    fn t_eresid_02_non_eigenvector_large_residual() {
        let laplacian = path_laplacian_3();
        let v = Array1::from_vec(vec![1.0_f64, 0.0, 0.0]);
        let residual = eigenpair_residual(&laplacian, &v, 0.0);
        assert!(
            residual >= RSVD_QUALITY_THRESHOLD,
            "non-eigenvector residual={residual:.2e}, expected >= RSVD_QUALITY_THRESHOLD={RSVD_QUALITY_THRESHOLD:.2e}"
        );
    }

    // ── Accuracy: max_eigenpair_residual ─────────────────────────────────────

    #[test]
    fn t_maxresid_01_returns_worst_across_two_eigenpairs() {
        let laplacian = path_laplacian_3();
        // Trivial eigenvector at λ=0: small residual
        let s = 1.0_f64 / 3.0_f64.sqrt();
        // Non-eigenvector at λ=0: large residual
        let eigenvalues = Array1::from_vec(vec![0.0_f64, 0.0_f64]);
        let eigenvectors = Array2::from_shape_vec(
            (3, 2),
            vec![s, 1.0_f64, s, 0.0_f64, s, 0.0_f64],
        )
        .unwrap();
        let max_res = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
        // The second column [1,0,0] has a large residual; max must reflect that.
        assert!(
            max_res >= RSVD_QUALITY_THRESHOLD,
            "max_residual={max_res:.2e}, expected >= RSVD_QUALITY_THRESHOLD"
        );
    }

    // ── Accuracy: orthogonality_error ────────────────────────────────────────

    #[test]
    fn t_ortho_01_identity_columns_zero_error() {
        let v = Array2::<f64>::eye(3);
        let err = orthogonality_error(&v);
        assert!(err < 1e-14, "identity orthogonality_error={err:.2e}, expected ≈ 0");
    }

    #[test]
    fn t_ortho_02_non_orthogonal_positive_error() {
        let v = Array2::from_shape_vec(
            (3, 2),
            vec![1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 0.0_f64, 0.0_f64],
        )
        .unwrap();
        let err = orthogonality_error(&v);
        assert!(err > 0.0, "non-orthogonal matrix should have positive error, got {err:.2e}");
    }

    // ── Accuracy: check_eigenvalue_bounds ────────────────────────────────────

    #[test]
    fn t_bounds_01_valid_range_and_sorted() {
        let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
        assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (true, true));
    }

    #[test]
    fn t_bounds_02_negative_eigenvalue_out_of_range() {
        let eigs = Array1::from_vec(vec![-0.1_f64, 0.5]);
        assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (false, true));
    }

    #[test]
    fn t_bounds_03_unsorted_eigenvalues() {
        let eigs = Array1::from_vec(vec![0.5_f64, 0.1]);
        assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (true, false));
    }

    // ── Accuracy: separation_ratio ───────────────────────────────────────────

    #[test]
    fn t_sep_01_well_separated_clusters() {
        // 4 points: cluster 0 near [0,0], cluster 1 near [10,0]
        let data = vec![0.0_f64, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0];
        let embedding = Array2::from_shape_vec((4, 2), data).unwrap();
        let labels = vec![0usize, 0, 1, 1];
        let ratio = separation_ratio(embedding.view(), &labels);
        assert!(ratio > 1.0, "well-separated clusters should have ratio > 1.0, got {ratio:.4}");
    }

    #[test]
    fn t_sep_02_overlapping_clusters() {
        // 4 points where intra-cluster spread > inter-centroid distance
        let data = vec![0.0_f64, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0];
        let embedding = Array2::from_shape_vec((4, 2), data).unwrap();
        let labels = vec![0usize, 0, 1, 1];
        let ratio = separation_ratio(embedding.view(), &labels);
        // Centroids [1.5,0] and [2.5,0]: inter=1.0, max_intra=1.5. ratio=0.667
        assert!(
            ratio <= 1.0,
            "overlapping clusters should have ratio <= 1.0, got {ratio:.4}"
        );
    }

    // ── Parity: eigenvalue_abs_errors ────────────────────────────────────────

    #[test]
    fn t_abserr_01_basic_errors() {
        let computed = Array1::from_vec(vec![1.0_f64, 2.0]);
        let reference = Array1::from_vec(vec![1.1_f64, 2.2]);
        let errors = eigenvalue_abs_errors(&computed, &reference);
        assert!(
            (errors[0] - 0.1).abs() < 1e-14,
            "errors[0]={:.6e}, expected 0.1",
            errors[0]
        );
        assert!(
            (errors[1] - 0.2).abs() < 1e-14,
            "errors[1]={:.6e}, expected 0.2",
            errors[1]
        );
    }

    // ── Parity: subspace_gram_det ────────────────────────────────────────────

    #[test]
    fn t_gram2d_01_parallel_subspace_near_one() {
        // u = r = [[1,0],[0,1],[0,0]] (orthonormal columns)
        let data = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let u = Array2::from_shape_vec((3, 2), data.clone()).unwrap();
        let r = Array2::from_shape_vec((3, 2), data).unwrap();
        let det = subspace_gram_det(u.view(), r.view());
        assert!(
            (det - 1.0).abs() < 1e-12,
            "parallel orthonormal subspace: det={det:.6e}, expected ≈ 1.0"
        );
    }

    #[test]
    fn t_gram2d_02_orthogonal_subspace_near_zero() {
        // u spans {e1, e2}, r spans {e3, e4}
        let u_data = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let r_data = vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let u = Array2::from_shape_vec((4, 2), u_data).unwrap();
        let r = Array2::from_shape_vec((4, 2), r_data).unwrap();
        let det = subspace_gram_det(u.view(), r.view());
        assert!(det < 1e-12, "orthogonal subspaces: det={det:.6e}, expected ≈ 0.0");
    }

    // ── Parity: subspace_gram_det_kd ─────────────────────────────────────────

    #[test]
    fn t_gramkd_01_same_subspace_near_one() {
        // k=3 identity columns in R^3
        let eye = Array2::<f64>::eye(3);
        let det = subspace_gram_det_kd(eye.view(), eye.view());
        assert!(
            (det - 1.0).abs() < 1e-10,
            "same subspace: det={det:.6e}, expected ≈ 1.0"
        );
    }

    #[test]
    fn t_gramkd_02_orthogonal_subspaces_near_zero() {
        // computed spans {e1,e2}, reference spans {e3,e4} in R^4
        let comp_data = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let ref_data = vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let comp = Array2::from_shape_vec((4, 2), comp_data).unwrap();
        let reff = Array2::from_shape_vec((4, 2), ref_data).unwrap();
        let det = subspace_gram_det_kd(comp.view(), reff.view());
        assert!(det < 1e-10, "orthogonal subspaces: det={det:.6e}, expected ≈ 0.0");
    }

    // ── Parity: sign_agnostic_max_error ──────────────────────────────────────

    #[test]
    fn t_signerr_01_identical_arrays_zero_error() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let arr = Array2::from_shape_vec((4, 1), data).unwrap();
        let err = sign_agnostic_max_error(&arr, &arr);
        assert_eq!(err, 0.0, "identical arrays: error={err:.2e}, expected 0.0");
    }

    #[test]
    fn t_signerr_02_negated_columns_zero_error() {
        let computed_data = vec![1.0_f32, -2.0, 3.0, -4.0];
        let reference_data = vec![-1.0_f32, 2.0, -3.0, 4.0];
        let computed = Array2::from_shape_vec((4, 1), computed_data).unwrap();
        let reference = Array2::from_shape_vec((4, 1), reference_data).unwrap();
        let err = sign_agnostic_max_error(&computed, &reference);
        assert_eq!(err, 0.0, "fully negated column: error={err:.2e}, expected 0.0");
    }

    #[test]
    fn t_signerr_03_small_perturbation_small_error() {
        let reference_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let computed_data = vec![1.01_f32, 2.01, 3.01, 4.01];
        let reference = Array2::from_shape_vec((4, 1), reference_data).unwrap();
        let computed = Array2::from_shape_vec((4, 1), computed_data).unwrap();
        let err = sign_agnostic_max_error(&computed, &reference);
        assert!(
            err > 0.0 && err < 0.1,
            "small perturbation: error={err:.4e}, expected small positive"
        );
    }

    // ── Diagnostic: tolerance_margin ─────────────────────────────────────────

    #[test]
    fn t_tolmargin_01_basic() {
        let margin = tolerance_margin(1e-5, 5e-6);
        assert!(
            (margin - 2.0).abs() < 1e-12,
            "tolerance_margin={margin:.6e}, expected 2.0"
        );
    }

    // ── Diagnostic: spectral_gap ──────────────────────────────────────────────

    #[test]
    fn t_specgap_01_basic() {
        let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
        let gap = spectral_gap(&eigs);
        assert!(
            (gap - 0.5).abs() < 1e-14,
            "spectral_gap={gap:.6e}, expected 0.5"
        );
    }

    // ── Diagnostic: eigenvalue_condition_number ───────────────────────────────

    #[test]
    fn t_condnum_01_basic() {
        let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
        let cn = eigenvalue_condition_number(&eigs);
        assert!(
            (cn - 2.0).abs() < 1e-14,
            "eigenvalue_condition_number={cn:.6e}, expected 2.0"
        );
    }

    // ── Data structures: serde round-trip ────────────────────────────────────

    #[cfg(feature = "testing")]
    #[test]
    fn t_serde_01_round_trip_metric_result() {
        let m = MetricResult {
            name: "test_metric".to_string(),
            dimension: 2,
            value: 0.001,
            threshold: 0.01,
            passed: true,
        };
        let json = serde_json::to_string(&m).expect("serialize MetricResult");
        let m2: MetricResult = serde_json::from_str(&json).expect("deserialize MetricResult");
        assert_eq!(m2.name, m.name);
        assert_eq!(m2.dimension, m.dimension);
        assert_eq!(m2.value, m.value);
        assert_eq!(m2.threshold, m.threshold);
        assert_eq!(m2.passed, m.passed);
    }

    #[cfg(feature = "testing")]
    #[test]
    fn t_serde_02_round_trip_assessment_report() {
        let r = AssessmentReport {
            dataset: "test_dataset".to_string(),
            n: 100,
            metrics: vec![MetricResult {
                name: "residual".to_string(),
                dimension: 2,
                value: 1e-7,
                threshold: 1e-6,
                passed: true,
            }],
        };
        let json = serde_json::to_string(&r).expect("serialize AssessmentReport");
        let r2: AssessmentReport =
            serde_json::from_str(&json).expect("deserialize AssessmentReport");
        assert_eq!(r2.dataset, r.dataset);
        assert_eq!(r2.n, r.n);
        assert_eq!(r2.metrics.len(), 1);
        assert_eq!(r2.metrics[0].name, "residual");
    }

    #[cfg(feature = "testing")]
    #[test]
    fn t_serde_03_round_trip_experiment_metrics() {
        let e = ExperimentMetrics {
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            datasets: vec![AssessmentReport {
                dataset: "iris".to_string(),
                n: 150,
                metrics: vec![],
            }],
        };
        let json = serde_json::to_string(&e).expect("serialize ExperimentMetrics");
        let e2: ExperimentMetrics =
            serde_json::from_str(&json).expect("deserialize ExperimentMetrics");
        assert_eq!(e2.generated_at, e.generated_at);
        assert_eq!(e2.datasets.len(), 1);
        assert_eq!(e2.datasets[0].dataset, "iris");
    }
}
