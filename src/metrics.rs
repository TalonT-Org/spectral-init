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
use rayon::prelude::*;
use sprs::CsMatI;
use std::collections::HashSet;

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
/// Set to 2e-5: provides ≥2× safety margin for the tightest observed fixture
/// (blobs_connected_2000 at 9.097e-6, margin = 2.20×). Tighter than rSVD (1e-2).
pub const LOBPCG_QUALITY_THRESHOLD: f64 = 2e-5;

/// Maximum acceptable max-residual from shift-and-invert LOBPCG (Level 2).
/// Sinv achieves near-exact results (like dense EVD); 1e-6 accepts all
/// well-converged results while correctly escalating pathological graphs.
pub const SINV_LOBPCG_QUALITY_THRESHOLD: f64 = 1e-6;

/// LOBPCG internal per-vector convergence tolerance (linfa-linalg `tol` argument).
/// Remains at 1e-5 — this is independent of `LOBPCG_QUALITY_THRESHOLD`: the former
/// controls the iterative solver's per-step stopping criterion; the latter is the
/// post-solve acceptance gate applied to the returned residuals.
pub const LOBPCG_ACCEPT_TOL: f64 = 1e-5;

/// Linfa-linalg internal convergence tolerance for shift-invert LOBPCG.
/// Tighter than SINV_LOBPCG_QUALITY_THRESHOLD (1e-6) because this controls the iterative
/// solver's per-step residual; the outer acceptance guard (SINV_LOBPCG_QUALITY_THRESHOLD)
/// then decides whether the converged result is good enough to use.
pub const SINV_LINFA_TOL: f64 = 1e-8;

/// Minimum eigenvalue gap to treat eigenpairs as distinct; below this use subspace comparison.
pub const DEGENERATE_GAP_THRESHOLD: f64 = 1e-6;

/// Minimum acceptable subspace Gram determinant for quality assessment.
pub const SUBSPACE_GRAM_DET_THRESHOLD: f64 = 0.95;

/// Empirical noise floor for eigenvalue deviation below 0.0 in faer dense EVD
/// (the tightest solver; rSVD 2-λ cancellation can reach ~1e-8, which exceeds this constant).
///
/// After `enforce_psd_contract` is applied at the solver chain boundary, callers of
/// `solve_eigenproblem` receive eigenvalues already clamped to [0, 2] and do not need
/// to account for this floor. This constant is retained for internal diagnostic use only.
pub const PSD_NOISE_FLOOR: f64 = 1e-9;

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
    let v_norm = eigenvector.dot(eigenvector).sqrt().max(1e-300);
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

// ─── AVX2+FMA squared-distance kernel ────────────────────────────────────────

/// Squared Euclidean distance using AVX2+FMA intrinsics.
///
/// # Safety
/// Both slices must have at least 10 elements (enforced by the `d_x >= 10` guard at the
/// call site). Two 4-wide loads cover elements 0..8; a scalar tail handles 8..n.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2(xi: &[f64], xj: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = xi.len().min(xj.len());
    unsafe {
        let a0 = _mm256_loadu_pd(xi.as_ptr());
        let b0 = _mm256_loadu_pd(xj.as_ptr());
        let d0 = _mm256_sub_pd(a0, b0);
        let mut acc = _mm256_mul_pd(d0, d0);
        let a1 = _mm256_loadu_pd(xi.as_ptr().add(4));
        let b1 = _mm256_loadu_pd(xj.as_ptr().add(4));
        let d1 = _mm256_sub_pd(a1, b1);
        acc = _mm256_fmadd_pd(d1, d1, acc);
        let lo = _mm256_castpd256_pd128(acc);
        let hi = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(lo, hi);
        let halved = _mm_hadd_pd(sum128, sum128);
        let mut result = _mm_cvtsd_f64(halved);
        for i in 8..n {
            let d = xi[i] - xj[i];
            result += d * d;
        }
        result
    }
}

/// Batch squared Euclidean distances from a 2D query point to `n` target points
/// stored in row-major layout (`y_flat`, stride 2). Processes two target points per
/// SIMD iteration using `_mm256_hadd_pd` for horizontal reduction.
///
/// # Safety
/// - `yi` must have exactly 2 elements.
/// - `y_flat` must have at least `n * 2` elements (row-major, d_y = 2).
/// - `out` must have at least `n` elements.
/// Called only when `d_y == 2`, `y.is_standard_layout()`, and `avx2` is detected at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_sq_2d_avx2_batch(
    yi: &[f64],
    y_flat: &[f64],
    n: usize,
    out: &mut [f64],
) {
    use std::arch::x86_64::*;
    debug_assert!(yi.len() >= 2, "yi must have at least 2 elements");
    debug_assert!(y_flat.len() >= n * 2, "y_flat must have at least n*2 elements");
    debug_assert!(out.len() >= n, "out must have at least n elements");
    // Broadcast query: [yi[1], yi[0], yi[1], yi[0]] — _mm256_set_pd args are (e3,e2,e1,e0)
    let yi_bc = _mm256_set_pd(yi[1], yi[0], yi[1], yi[0]);
    let mut j = 0usize;
    while j + 1 < n {
        // Load two target points: [yj[0], yj[1], yj+1[0], yj+1[1]]
        debug_assert!(j * 2 + 3 < y_flat.len(), "y_flat bounds exceeded at j={j}");
        let yj_pair = _mm256_loadu_pd(y_flat.as_ptr().add(j * 2));
        let diff = _mm256_sub_pd(yi_bc, yj_pair);
        let sq   = _mm256_mul_pd(diff, diff);
        // hadd: lower 128-bit → sq[0]+sq[1] = dist_j; upper 128-bit → sq[2]+sq[3] = dist_{j+1}
        let hadd = _mm256_hadd_pd(sq, sq);
        out[j]     = _mm_cvtsd_f64(_mm256_castpd256_pd128(hadd));
        out[j + 1] = _mm_cvtsd_f64(_mm256_extractf128_pd(hadd, 1));
        j += 2;
    }
    // Scalar tail for odd n
    if j < n {
        let base = j * 2;
        let d0 = yi[0] - y_flat[base];
        let d1 = yi[1] - y_flat[base + 1];
        out[j] = d0 * d0 + d1 * d1;
    }
}

// ─── Trustworthiness metric ───────────────────────────────────────────────────

/// Computes the trustworthiness metric T(k) — a measure of how well the k-nearest-neighbor
/// structure of X (high-dimensional) is preserved in Y (embedding).
///
/// Uses partial rank (introselect) for O(n)-average X-NN detection, thread-local
/// scratch buffers to eliminate per-row allocations, and AVX2+FMA SIMD dispatch
/// for X-distance computation when dimensionality >= 10.
///
/// # Formula
/// `T(k) = 1 − (2 / (n·k·(2n−3k−1))) · Σᵢ Σ_{j ∈ U_i(k)} (r(i,j) − k)`
///
/// where `r(i,j)` is the 0-indexed rank of j in the distance ordering from i in X
/// (self = rank 0), and `U_i(k)` is the set of j in k-NN(i, Y) but NOT in k-NN(i, X).
///
/// # Panics
/// Panics if `k == 0`, `k >= n / 2`, or if `x` and `y` have different row counts.
/// The `k < n/2` restriction is required by the normalization denominator
/// `G_k = n·k·(2n−3k−1)`: when `k ≥ n/2` this denominator produces T > 1.
/// The guard uses integer floor division (`n / 2`), matching sklearn's exact boundary
/// condition (`n_neighbors >= n // 2` raises `ValueError` in sklearn).
pub fn trustworthiness(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64 {
    use std::cell::RefCell;
    let n = x.nrows();
    assert_eq!(y.nrows(), n, "trustworthiness: x and y must have the same number of rows");
    assert!(k > 0, "trustworthiness: k must be > 0");
    assert!(k < n / 2,
        "trustworthiness: k must be < n/2 (got k={k}, n={n}, n/2={}); \
        this constraint is required by the normalization denominator and matches sklearn's ValueError",
        n / 2);

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;
    let d_x = x.ncols();
    let d_y = y.ncols();

    #[cfg(target_arch = "x86_64")]
    let use_avx2_y = {
        d_y == 2
            && y.is_standard_layout()
            && is_x86_feature_detected!("avx2")
    };
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2_y = false;

    // Validate contiguity once before the parallel loop: SIMD dispatch requires C-contiguous rows.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    if use_avx2 && d_x >= 10 {
        assert!(x.is_standard_layout(),
            "trustworthiness: x must be in C-contiguous (standard) layout for SIMD dispatch");
    }

    thread_local! {
        static COMB_DIST_X:  RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static COMB_INDICES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    thread_local! {
        static COMB_DIST_Y:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static COMB_INDICES_Y: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    #[cfg(feature = "profiling")]
    static X_DIST_NS:  std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static X_SORT_NS:  std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static Y_DIST_NS:  std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static PENALTY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let penalty_sum: f64 = (0..n).into_par_iter().map(|i| {
        let xi = x.row(i);
        let yi = y.row(i);

        COMB_DIST_X.with(|dist_x_cell| {
            COMB_INDICES.with(|indices_cell| {
                let mut dist_x = dist_x_cell.borrow_mut();
                let mut indices = indices_cell.borrow_mut();

                // ── x_dist step ──────────────────────────────────────────────
                #[cfg(feature = "profiling")]
                let t_x_dist = std::time::Instant::now();

                dist_x.clear();
                dist_x.resize(n, 0.0f64);
                for j in 0..n {
                    let xj = x.row(j);
                    dist_x[j] = {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                        {
                            if use_avx2 && d_x >= 10 {
                                let si = xi.as_slice().expect("x row must be contiguous");
                                let sj = xj.as_slice().expect("x row must be contiguous");
                                // SAFETY: runtime + d_x check guarantees AVX2+FMA and >= 8 elements.
                                unsafe { dist_sq_avx2(si, sj) }
                            } else {
                                xi.iter().zip(xj.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum()
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                        {
                            xi.iter().zip(xj.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum()
                        }
                    };
                }

                #[cfg(feature = "profiling")]
                X_DIST_NS.fetch_add(t_x_dist.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed);

                // ── x_sort step ──────────────────────────────────────────────
                #[cfg(feature = "profiling")]
                let t_x_sort = std::time::Instant::now();

                indices.clear();
                indices.extend(0..n);
                indices.select_nth_unstable_by(k, |&a, &b| {
                    dist_x[a].total_cmp(&dist_x[b]).then(a.cmp(&b))
                });

                let knn_x_set: HashSet<usize> =
                    indices[..=k].iter().filter(|&&m| m != i).copied().collect();

                #[cfg(feature = "profiling")]
                X_SORT_NS.fetch_add(t_x_sort.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed);

                // ── y_dist step (flat_simd, replaces BinaryHeap) ─────────────
                #[cfg(feature = "profiling")]
                let t_y_dist = std::time::Instant::now();

                COMB_DIST_Y.with(|dy_cell| {
                    COMB_INDICES_Y.with(|iy_cell| {
                        let mut dist_y    = dy_cell.borrow_mut();
                        let mut indices_y = iy_cell.borrow_mut();

                        dist_y.clear();
                        dist_y.resize(n, 0.0f64);

                        // Fill Y distances: AVX2 2D batch or scalar fallback.
                        #[cfg(target_arch = "x86_64")]
                        if use_avx2_y {
                            let y_flat = y.as_slice()
                                .expect("y must be standard layout for AVX2 dispatch");
                            let yi_slice = &y_flat[i * 2..(i + 1) * 2];
                            // SAFETY: y_flat has n*d_y elements; yi_slice has 2 elements;
                            // dist_y has n elements; use_avx2_y guarantees d_y==2,
                            // standard_layout, and runtime AVX2.
                            unsafe { dist_sq_2d_avx2_batch(yi_slice, y_flat, n, &mut dist_y); }
                        } else {
                            for j in 0..n {
                                let yj = y.row(j);
                                dist_y[j] = yi.iter().zip(yj.iter())
                                    .map(|(&a, &b)| (a - b) * (a - b)).sum();
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            for j in 0..n {
                                let yj = y.row(j);
                                dist_y[j] = yi.iter().zip(yj.iter())
                                    .map(|(&a, &b)| (a - b) * (a - b)).sum();
                            }
                        }

                        dist_y[i] = f64::INFINITY; // self-exclusion

                        indices_y.clear();
                        indices_y.extend(0..n);
                        indices_y.select_nth_unstable_by(k, |&a, &b| {
                            dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b))
                        });

                        #[cfg(feature = "profiling")]
                        Y_DIST_NS.fetch_add(t_y_dist.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed);

                        // ── penalty step ──────────────────────────────────────
                        #[cfg(feature = "profiling")]
                        let t_penalty = std::time::Instant::now();

                        let mut row_penalty = 0u64;
                        for &j in &indices_y[..k] {
                            if !knn_x_set.contains(&j) {
                                let dj = dist_x[j];
                                let rank: usize = (0..n)
                                    .filter(|&m| dist_x[m] < dj || (dist_x[m] == dj && m < j))
                                    .count();
                                row_penalty += (rank - k) as u64;
                            }
                        }

                        #[cfg(feature = "profiling")]
                        PENALTY_NS.fetch_add(t_penalty.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed);

                        row_penalty as f64
                    })
                })
            })
        })
    }).sum();

    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        eprintln!("[timing:x_dist] {}",  X_DIST_NS.load(Ordering::Relaxed));
        eprintln!("[timing:x_sort] {}",  X_SORT_NS.load(Ordering::Relaxed));
        eprintln!("[timing:y_dist] {}",  Y_DIST_NS.load(Ordering::Relaxed));
        eprintln!("[timing:penalty] {}", PENALTY_NS.load(Ordering::Relaxed));
    }

    let denom = n as f64 * k as f64 * (2 * n).saturating_sub(3 * k + 1) as f64;
    1.0 - penalty_sum * 2.0 / denom
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
    fn t_const_01_all_7_constants_have_required_values() {
        assert_eq!(DENSE_EVD_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(LOBPCG_QUALITY_THRESHOLD, 2e-5_f64);
        assert_eq!(SINV_LOBPCG_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(RSVD_QUALITY_THRESHOLD, 1e-2_f64);
        assert_eq!(LOBPCG_ACCEPT_TOL, 1e-5_f64);
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

    #[test]
    fn t_tolmargin_02_zero_worst_value_returns_infinity() {
        let margin = tolerance_margin(1e-5, 0.0);
        assert!(
            margin.is_infinite() && margin > 0.0,
            "tolerance_margin with worst_value=0.0 should be +∞, got {margin:.6e}"
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

    // ── Trustworthiness metric ────────────────────────────────────────────────

    #[test]
    fn t_tw_01_perfect_preservation() {
        // 20-point grid; each point embedded to its own location
        let x = ndarray::Array2::from_shape_fn((20, 4), |(i, d)| (i * 4 + d) as f64);
        let y = x.slice(ndarray::s![.., ..2]).to_owned();
        let t = trustworthiness(x.view(), y.view(), 5);
        assert!((t - 1.0).abs() < 1e-10, "perfect preservation: T={t}");
    }

    #[test]
    fn t_tw_02_result_in_unit_interval() {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        let x = ndarray::Array2::from_shape_fn((20, 10), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((20, 2), |_| rng.random::<f64>());
        let t = trustworthiness(x.view(), y.view(), 5);
        assert!(t >= 0.0 && t <= 1.0, "T out of [0,1]: {t}");
    }

    /// Hand-verifiable 4-point, k=1 example with exactly one violation of rank penalty 1.
    ///
    /// n=4, k=1: denominator = 4·1·(8−3−1) = 16.
    ///
    /// X = [[0,0],[1,0],[0,1.5],[0,100]]:
    ///   KNN(0,X,1)={1} (rank_x[2]=2), KNN(1,X,1)={0}, KNN(2,X,1)={0}, KNN(3,X,1)={2}
    ///
    /// Y = [[0,0],[1,0],[0,0.5],[0,100]] (point 2 moved closer to 0):
    ///   KNN(0,Y,1)={2} → violation: rank_x[2]=2, penalty = 2−1 = 1
    ///   KNN(1,Y,1)={0} → 0 ∈ KNN(1,X,1): no penalty
    ///   KNN(2,Y,1)={0} → 0 ∈ KNN(2,X,1): no penalty
    ///   KNN(3,Y,1)={2} → 2 ∈ KNN(3,X,1): no penalty
    ///
    /// T = 1 − 2·1/16 = 7/8 = 0.875
    #[test]
    fn t_tw_03_formula_hand_check() {
        let x = ndarray::Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 100.0],
        ).unwrap();
        let y = ndarray::Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 100.0],
        ).unwrap();
        let t = trustworthiness(x.view(), y.view(), 1);
        let expected = 7.0 / 8.0;
        assert!(
            (t - expected).abs() < 1e-10,
            "hand-check: T={t:.10}, expected {expected:.10} (7/8)"
        );
    }

    #[test]
    fn t_tw_04_k_less_than_half_n() {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let n = 20usize;
        let k = n / 2 - 1; // 9, just below the boundary
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        let t = trustworthiness(x.view(), y.view(), k);
        assert!(t.is_finite(), "T should be finite at k=n/2-1: {t}");
        assert!(t >= 0.0 && t <= 1.0, "T out of [0,1] at k=n/2-1: {t}");
    }

    #[test]
    #[should_panic(expected = "k must be < n/2")]
    fn t_tw_05_panics_on_k_gte_half_n() {
        let n = 20usize;
        let k = n / 2; // exactly n/2 — must panic
        let x = ndarray::Array2::zeros((n, 4));
        let y = ndarray::Array2::zeros((n, 2));
        let _ = trustworthiness(x.view(), y.view(), k);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[test]
    fn t_tw_06_avx2_kernel_matches_scalar() {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for len in [10, 16, 33, 100] {
            let a: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
            let b: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
            let scalar: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
            let avx2 = unsafe { super::dist_sq_avx2(&a, &b) };
            assert!((avx2 - scalar).abs() < 1e-10,
                "dist_sq_avx2 mismatch at len={len}: avx2={avx2}, scalar={scalar}");
        }
    }

    #[test]
    fn t_tw_07_partial_rank_matches_full_sort() {
        use rand::{SeedableRng, Rng};
        use std::collections::HashSet;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let n = 30;
        let k = 5;
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());

        for i in 0..n {
            let xi = x.row(i);
            let dist_x: Vec<f64> = (0..n)
                .map(|j| xi.iter().zip(x.row(j).iter()).map(|(&a, &b)| (a - b) * (a - b)).sum())
                .collect();

            // Full-sort baseline: knn_x_set via sort + rank scatter
            let mut sorted: Vec<(f64, usize)> = dist_x.iter().copied().zip(0..n).collect();
            sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let knn_baseline: HashSet<usize> = sorted[1..=k].iter().map(|&(_, j)| j).collect();

            // Partial-rank approach: select_nth_unstable_by
            let mut indices: Vec<usize> = (0..n).collect();
            indices.select_nth_unstable_by(k, |&a, &b| {
                dist_x[a].total_cmp(&dist_x[b]).then(a.cmp(&b))
            });
            let knn_partial: HashSet<usize> =
                indices[..=k].iter().filter(|&&m| m != i).copied().collect();

            assert_eq!(knn_baseline, knn_partial,
                "knn_x_set mismatch at row {i}");

            // Verify rank values for all non-knn points
            let mut rank_x = vec![0usize; n];
            for (rank, &(_, j)) in sorted.iter().enumerate() {
                rank_x[j] = rank;
            }
            for j in 0..n {
                if j == i || knn_baseline.contains(&j) { continue; }
                let dj = dist_x[j];
                let scan_rank: usize = (0..n)
                    .filter(|&m| dist_x[m] < dj || (dist_x[m] == dj && m < j))
                    .count();
                assert_eq!(rank_x[j], scan_rank,
                    "rank mismatch for point {j} from row {i}: sort={}, scan={scan_rank}", rank_x[j]);
            }
        }
    }

    /// Brute-force O(n²) reference: full-sort X distances, sort-based Y-kNN.
    fn trustworthiness_brute_force(x: ndarray::ArrayView2<f64>, y: ndarray::ArrayView2<f64>, k: usize) -> f64 {
        use std::collections::HashSet;
        let n = x.nrows();
        let penalty_sum: f64 = (0..n).map(|i| {
            let xi = x.row(i);
            let yi = y.row(i);

            let mut dist_x: Vec<(f64, usize)> = (0..n).map(|j| {
                let d: f64 = xi.iter().zip(x.row(j).iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
                (d, j)
            }).collect();
            dist_x.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let knn_x: HashSet<usize> = dist_x[1..=k].iter().map(|&(_, j)| j).collect();
            let mut rank_x = vec![0usize; n];
            for (rank, &(_, j)) in dist_x.iter().enumerate() {
                rank_x[j] = rank;
            }

            let mut dist_y: Vec<(f64, usize)> = (0..n).filter(|&j| j != i).map(|j| {
                let d: f64 = yi.iter().zip(y.row(j).iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
                (d, j)
            }).collect();
            dist_y.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let knn_y: HashSet<usize> = dist_y[..k].iter().map(|&(_, j)| j).collect();

            let mut row_penalty = 0u64;
            for j in &knn_y {
                if !knn_x.contains(j) {
                    row_penalty += (rank_x[*j] - k) as u64;
                }
            }
            row_penalty as f64
        }).sum();
        let denom = n as f64 * k as f64 * (2 * n).saturating_sub(3 * k + 1) as f64;
        1.0 - penalty_sum * 2.0 / denom
    }

    #[test]
    fn t_tw_08_combined_matches_baseline() {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(123);
        let n = 50;
        let x = ndarray::Array2::from_shape_fn((n, 6), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());

        for k in [3, 7] {
            let t = trustworthiness(x.view(), y.view(), k);
            let t_ref = trustworthiness_brute_force(x.view(), y.view(), k);
            assert!(t.is_finite(), "T(k={k}) must be finite, got {t}");
            assert!(t >= 0.0 && t <= 1.0, "T(k={k}) out of [0,1]: {t}");
            assert!((t - t_ref).abs() < 1e-12,
                "T(k={k}) combined={t} diverges from brute-force reference={t_ref}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn t_tw_09_avx2_2d_kernel_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // skip on CPUs without AVX2
        }
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        for n in [0usize, 1, 2, 3, 10, 20, 51] {
            let y_flat: Vec<f64> = (0..n * 2).map(|_| rng.random::<f64>()).collect();
            let yi: Vec<f64> = (0..2).map(|_| rng.random::<f64>()).collect();
            let mut out_avx2 = vec![0.0f64; n];
            // SAFETY: runtime check above guarantees AVX2; y_flat has n*2, yi has 2, out has n.
            unsafe { super::dist_sq_2d_avx2_batch(&yi, &y_flat, n, &mut out_avx2); }
            for j in 0..n {
                let scalar: f64 = yi.iter()
                    .zip(y_flat[j * 2..j * 2 + 2].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                assert!(
                    (out_avx2[j] - scalar).abs() < 1e-10,
                    "dist_sq_2d_avx2_batch mismatch at n={n}, j={j}: avx2={}, scalar={scalar}",
                    out_avx2[j]
                );
            }
        }
    }

    #[test]
    fn t_tw_10_self_exclusion_never_in_knn() {
        // Verify that the self-exclusion (dist_y[i] = INFINITY) prevents point i
        // from appearing in its own k-NN result. Catching a regression where the
        // INFINITY assignment is accidentally dropped.
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let n = 20;
        let k = 3;
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        // Run trustworthiness — internally computes k-NN of Y for each row.
        // We verify correctness by comparing against brute-force which also applies self-exclusion.
        let t = trustworthiness(x.view(), y.view(), k);
        let t_ref = trustworthiness_brute_force(x.view(), y.view(), k);
        assert!((t - t_ref).abs() < 1e-12,
            "self-exclusion regression: combined={t} diverges from brute-force={t_ref}");
        // Additionally, verify directly via brute-force that no point is its own neighbor.
        for i in 0..n {
            let yi = y.row(i);
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = yi.iter().zip(y.row(j).iter())
                        .map(|(&a, &b)| (a-b)*(a-b)).sum();
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            let knn: Vec<usize> = dists[..k].iter().map(|&(_, j)| j).collect();
            assert!(!knn.contains(&i), "point {i} appeared in its own k-NN");
        }
    }

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
