//! Diagnostic metrics for judging eigensolver output: accuracy checks, parity against a
//! reference, and conditioning figures. Shared by the integration tests and the research
//! harness.

#![allow(dead_code)]

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat as FaerMat, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Minimum eigenvalue gap to treat eigenpairs as distinct; below this use subspace comparison.
pub const DEGENERATE_GAP_THRESHOLD: f64 = 1e-6;

/// Minimum acceptable subspace Gram determinant for quality assessment.
pub const SUBSPACE_GRAM_DET_THRESHOLD: f64 = 0.95;

// ─── Accuracy metric functions ────────────────────────────────────────────────

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
    assert!(
        tol >= 0.0,
        "check_eigenvalue_bounds: tol must be non-negative, got {}",
        tol
    );
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
///
/// # Panics
/// Panics if a label in `0..n_components` has no members: its centroid would be NaN and
/// silently drop out of the min/max, so the ratio would cover only some components.
pub fn separation_ratio(embedding: ArrayView2<f64>, labels: &[usize]) -> f64 {
    let n = labels.len();
    assert_eq!(
        embedding.nrows(),
        n,
        "embedding rows must match labels length"
    );
    let n_dims = embedding.ncols();

    let n_components = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_components < 2 {
        return f64::INFINITY;
    }

    let mut component_members: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (i, &label) in labels.iter().enumerate() {
        component_members[label].push(i);
    }
    if let Some(empty) = component_members.iter().position(Vec::is_empty) {
        panic!(
            "separation_ratio: labels must be contiguous over 0..{n_components}; \
             component {empty} has no members"
        );
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
    assert_eq!(
        u.ncols(),
        2,
        "subspace_gram_det: u must have exactly 2 columns"
    );
    assert_eq!(
        r.ncols(),
        2,
        "subspace_gram_det: r must have exactly 2 columns"
    );
    assert_eq!(
        u.nrows(),
        r.nrows(),
        "subspace_gram_det: u and r must have the same number of rows"
    );

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

/// Maximum principal angle between the column subspaces of `a` and `b`, in radians:
/// `arccos(σ_min(Aᵀ·B))`. Both inputs are expected to have orthonormal columns.
pub fn max_subspace_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let m = a.t().dot(b);
    // MᵀM is symmetric PSD; its eigenvalues are the squares of the singular values of M.
    let mtm = m.t().dot(&m);
    let k = mtm.nrows();
    let mtm_faer = FaerMat::from_fn(k, k, |i, j| mtm[[i, j]]);
    let eigen = SelfAdjointEigen::new(mtm_faer.as_ref(), Side::Lower)
        .expect("max_subspace_angle: SelfAdjointEigen failed on M^T M");
    let mut min_sv_sq = f64::INFINITY;
    eigen
        .S()
        .for_each(|&x| min_sv_sq = min_sv_sq.min(x.max(0.0)));
    // Clamp to [0, 1] for arccos.
    min_sv_sq.sqrt().min(1.0).acos()
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
    assert!(
        eigenvalues.len() >= 2,
        "spectral_gap: need at least 2 eigenvalues"
    );
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
