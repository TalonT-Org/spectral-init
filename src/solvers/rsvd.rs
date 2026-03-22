use sprs::{CsMatI, TriMat};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use faer::{Mat as FaerMat, Side};
use faer::linalg::solvers::{Qr, SelfAdjointEigen};
use super::EigenResult;

/// Forms M = 2I − L by negating off-diagonal entries and mapping diagonal 1→1.
///
/// The symmetric normalized Laplacian L has eigenvalues in [0, 2]. The shifted
/// matrix M = 2I − L has eigenvalues in [0, 2] in reverse order: the k smallest
/// eigenvalues of L correspond to the k largest eigenvalues of M. Since M is
/// symmetric PSD, its singular values equal its eigenvalues, so a truncated SVD
/// of M directly yields the desired eigenpairs.
fn two_i_minus_laplacian(laplacian: &CsMatI<f64, usize>) -> CsMatI<f64, usize> {
    let n = laplacian.rows();
    let mut tri = TriMat::with_capacity((n, n), laplacian.nnz());
    for (val, (row, col)) in laplacian.iter() {
        let new_val = if row == col {
            2.0 - val   // diagonal: 2 − 1 = 1.0
        } else {
            -val        // off-diagonal: negate (L off-diag ≤ 0, M off-diag ≥ 0)
        };
        tri.add_triplet(row, col, new_val);
    }
    tri.to_csr()
}

/// Compute Y = M * X where M is sparse [n, n] and X is dense [n, ncols].
/// Performs column-wise sparse matrix-vector products using `sprs`.
fn sparse_dense_mult(m: &CsMatI<f64, usize>, x: &Array2<f64>) -> Array2<f64> {
    let (n, ncols) = x.dim();
    let mut y = Array2::<f64>::zeros((n, ncols));
    for j in 0..ncols {
        let col = x.column(j).to_owned();
        let result = m * &col;          // sprs SpMV: CsMat * Array1 → Array1
        y.column_mut(j).assign(&result);
    }
    y
}

/// Convert `ndarray::Array2<f64>` → `faer::Mat<f64>` element-by-element.
fn to_faer(a: &Array2<f64>) -> FaerMat<f64> {
    let (nrows, ncols) = a.dim();
    FaerMat::from_fn(nrows, ncols, |i, j| a[[i, j]])
}

/// Convert `faer::Mat<f64>` → `ndarray::Array2<f64>` element-by-element.
fn from_faer(a: &FaerMat<f64>) -> Array2<f64> {
    let nrows = a.nrows();
    let ncols = a.ncols();
    Array2::from_shape_fn((nrows, ncols), |(i, j)| a.col_as_slice(j)[i])
}

/// Compute the thin Q factor from the QR decomposition of Y using `faer`.
/// Returns Q with shape [nrows, min(nrows, ncols)].
fn qr_thin_q(y: &Array2<f64>) -> Array2<f64> {
    let faer_y = to_faer(y);
    let qr = Qr::new(faer_y.as_ref());
    let thin_q = qr.compute_thin_Q();
    from_faer(&thin_q)
}

/// Compute symmetric eigendecomposition of the small matrix B using `faer`.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are in ASCENDING order
/// and eigenvectors are columns of the returned Array2.
fn sym_eig(b: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let faer_b = to_faer(b);
    let eigen = SelfAdjointEigen::new(faer_b.as_ref(), Side::Lower)
        .expect("rsvd_solve: SelfAdjointEigen failed on B = Q^T M Q");
    // Eigenvalues in ASCENDING order (smallest first).
    // Use for_each since ColRef may have non-contiguous stride (no as_slice guarantee).
    let mut eigenvalues: Vec<f64> = Vec::new();
    eigen.S().for_each(|x| eigenvalues.push(*x));
    let eigenvectors = from_faer(&eigen.U().to_owned());
    (eigenvalues, eigenvectors)
}

/// Randomized SVD eigensolver via the 2I − L trick (Level 3).
///
/// Implements Algorithm 5.1 (Halko-Tropp) with power iteration:
/// - Forms M = 2I − L (eigenvalues of M = 2 − eigenvalues of L)
/// - Computes the top-(n_components+1) eigenvalues/vectors of M via randomized SVD
/// - Discards the trivial eigenvector (column 0, corresponding to λ_L ≈ 0)
/// - Maps eigenvalues back to L eigenvalues via λ_L = 2 − σ_M
pub(crate) fn rsvd_solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> EigenResult {
    let n = laplacian.rows();
    assert!(n_components > 0, "rsvd_solve: n_components must be >= 1, got 0");
    assert!(
        n_components < n,
        "rsvd_solve: n_components ({n_components}) must be < n ({n})"
    );
    // rank = n_components + 1: request one extra so we can discard the trivial vector
    let rank = n_components + 1;
    // k = total random vectors (rank + oversampling for accuracy).
    // Scale oversampling with n/10 so large graphs (e.g. n=5000 → oversampling=500)
    // get a subspace large enough to produce residuals below RSVD_QUALITY_THRESHOLD,
    // avoiding fallthrough to the O(n³) Level 4 dense EVD.
    let oversampling = (n / 10).max(rank.max(5)).min(n.saturating_sub(rank));
    let k = (rank + oversampling).min(n);
    let nbiter = 2;  // QR-stabilized subspace iterations (Halko-Tropp Algorithm 4.4)

    // ── Step A: Form M = 2I - L ──────────────────────────────────────────────
    let m = two_i_minus_laplacian(laplacian);

    // ── Step B: Generate random Gaussian sketch matrix Ω ∈ ℝ^{n × k} ────────
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut omega = Array2::from_shape_fn((n, k), |_| normal.sample(&mut rng));

    // ── Step C: Power iteration (Algorithm 4.4) to amplify signal ────────────
    // Each iteration squares the effective singular value decay of M,
    // improving accuracy for clustered eigenvalues.
    for _ in 0..nbiter {
        let y = sparse_dense_mult(&m, &omega);   // Y = M Ω [n, k]
        omega = qr_thin_q(&y);                   // Ω ← QR(Y).Q [n, k]
        let z = sparse_dense_mult(&m, &omega);   // Z = M Ω [n, k]
        omega = qr_thin_q(&z);                   // Ω ← QR(Z).Q [n, k]
    }

    // ── Step D: Final sketch and orthonormal basis ────────────────────────────
    let y = sparse_dense_mult(&m, &omega);       // Y = M Ω [n, k]
    let q = qr_thin_q(&y);                       // Q = QR(Y).Q [n, k]

    // ── Step E: Form small projected matrix B = Q^T M Q ──────────────────────
    let mq = sparse_dense_mult(&m, &q);          // M Q [n, k]
    let b = q.t().dot(&mq);                      // B = Q^T (M Q) [k, k]  symmetric PSD

    // ── Step F: Eigendecompose B (small k×k symmetric matrix) ─────────────────
    // eigenvalues ascending (smallest M eigenvalue first = largest L eigenvalue)
    let (m_eigenvals, u_b) = sym_eig(&b);       // m_eigenvals[k-1] ≈ 2 (trivial)

    // ── Step G: Recover full eigenvectors V = Q U_B ───────────────────────────
    let v = q.dot(&u_b);                         // [n, k]

    // ── Step I: Build output — trivial vector first, then n_components non-trivial ──
    //
    // M eigenvalues (ascending): index actual_n-1 is the trivial (λ_M ≈ 2.0 → λ_L ≈ 0).
    // Non-trivial L eigenvalues (ascending): indices actual_n-2, actual_n-3, ...,
    //   actual_n-1-n_components in the M ascending array.
    let actual_n = m_eigenvals.len();
    let trivial_m_idx = actual_n - 1;
    let mut eig_vals_vec: Vec<f64> = Vec::with_capacity(n_components + 1);
    let mut eig_col_indices: Vec<usize> = Vec::with_capacity(n_components + 1);

    // Trivial pair (λ_L ≈ 0)
    eig_vals_vec.push(2.0 - m_eigenvals[trivial_m_idx]);
    eig_col_indices.push(trivial_m_idx);

    // Non-trivial pairs in ascending L eigenvalue order
    for i in 0..n_components {
        let idx = actual_n.saturating_sub(2 + i);
        eig_vals_vec.push(2.0 - m_eigenvals[idx]);
        eig_col_indices.push(idx);
    }

    let mut eigenvectors = Array2::<f64>::zeros((n, n_components + 1));
    for (out_col, &in_col) in eig_col_indices.iter().enumerate() {
        eigenvectors.column_mut(out_col).assign(&v.column(in_col));
    }

    let eigenvalues = Array1::from_vec(eig_vals_vec);
    (eigenvalues, eigenvectors)
}

/// Accurate eigensolver via random subspace + direct L-projection.
///
/// Builds the same QR-stabilized power-iteration subspace as `rsvd_solve`
/// (using M = 2I − L to amplify the smallest-L eigenvectors), but then
/// projects **L** directly onto the subspace (B_L = Q^T L Q) and runs
/// a dense EVD on the small matrix.  Returning L eigenvalues via EVD of B_L
/// avoids the catastrophic cancellation `λ_L = 2 − λ_M` that occurs when
/// `λ_L ≈ 0` and `λ_M ≈ 2`.
///
/// Returns `n_components + 1` eigenpairs (trivial at index 0 followed by
/// `n_components` non-trivial pairs in ascending eigenvalue order).
///
/// `k_sub` controls the subspace size; larger values give higher accuracy at
/// higher cost.  Must satisfy `k_sub > n_components + 1`.
#[cfg(feature = "testing")]
pub(crate) fn rsvd_solve_accurate(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    k_sub: usize,
) -> (Array1<f64>, Array2<f64>) {
    let n = laplacian.rows();
    assert!(k_sub > n_components + 1, "k_sub must be > n_components + 1");
    let k = k_sub.min(n);

    // ── Step A: Build power-iteration subspace using M = 2I − L ─────────────
    let m = two_i_minus_laplacian(laplacian);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut omega = Array2::from_shape_fn((n, k), |_| normal.sample(&mut rng));

    let nbiter = 6;  // more iterations for near-degenerate eigenvalue gaps
    for _ in 0..nbiter {
        let y = sparse_dense_mult(&m, &omega);
        omega = qr_thin_q(&y);
        let z = sparse_dense_mult(&m, &omega);
        omega = qr_thin_q(&z);
    }
    let y = sparse_dense_mult(&m, &omega);
    let q = qr_thin_q(&y);  // [n, k] orthonormal subspace basis

    // ── Step B: Project L directly onto subspace: B_L = Q^T L Q ─────────────
    // This avoids the λ_L = 2 − λ_M cancellation for near-zero eigenvalues.
    let lq = sparse_dense_mult(laplacian, &q);  // L·Q [n, k]
    let b_l = q.t().dot(&lq);                   // B_L [k, k]  symmetric PSD

    // ── Step C: Dense EVD of B_L (ascending = smallest L eigenvalue first) ──
    let (l_eigenvals, u_b) = sym_eig(&b_l);

    // ── Step D: Recover full Ritz vectors V = Q U_B ──────────────────────────
    let v = q.dot(&u_b);  // [n, k]

    // ── Step E: Build output — trivial first (index 0), then non-trivials ────
    // l_eigenvals[0] ≈ 0 (trivial), l_eigenvals[1..=n_components] non-trivials
    let n_out = n_components + 1;
    let eigenvalues = Array1::from_iter(l_eigenvals[..n_out].iter().copied());
    let mut eigenvectors = Array2::<f64>::zeros((n, n_out));
    for i in 0..n_out {
        eigenvectors.column_mut(i).assign(&v.column(i));
    }
    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::build_normalized_laplacian;
    use crate::ComputeMode;

    /// Build a sparse graph (CSR f32/u32) from a list of undirected (row, col, weight) edges.
    fn make_graph(n: usize, edges: &[(usize, usize, f32)]) -> sprs::CsMatI<f32, u32, usize> {
        let mut tri: TriMat<f32> = TriMat::new((n, n));
        for &(r, c, w) in edges {
            tri.add_triplet(r, c, w);
            if r != c {
                tri.add_triplet(c, r, w);  // symmetric
            }
        }
        let csr_usize: sprs::CsMatI<f32, usize> = tri.to_csr();
        let (rows, cols) = csr_usize.shape();
        let indptr: Vec<usize> = csr_usize.indptr().to_owned().into_raw_storage();
        let indices: Vec<u32> = csr_usize.indices().iter().map(|&i| i as u32).collect();
        let data: Vec<f32> = csr_usize.data().to_vec();
        sprs::CsMatI::new((rows, cols), indptr, indices, data)
    }

    /// Test 1: two_i_minus_laplacian construction correctness.
    ///
    /// Uses a 3-node path graph 0-1-2 with uniform weight 1.0.
    #[test]
    fn test_two_i_minus_laplacian_path_3() {
        let graph = make_graph(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let (_, sqrt_deg) = crate::laplacian::compute_degrees(&graph, ComputeMode::PythonCompat);
        let inv_sqrt_deg = sqrt_deg.mapv(|x| if x == 0.0 { 0.0 } else { 1.0 / x });
        let l = build_normalized_laplacian(&graph, inv_sqrt_deg.as_slice().unwrap());
        let m = two_i_minus_laplacian(&l);

        let n = l.rows();
        assert_eq!(m.rows(), n);
        assert_eq!(m.cols(), n);
        assert_eq!(m.nnz(), l.nnz(), "sparsity pattern must match L");

        // Diagonal entries must be 1.0 (= 2 - L[i,i] = 2 - 1 = 1)
        for i in 0..n {
            let l_diag = l.get(i, i).copied().unwrap_or(0.0);
            let m_diag = m.get(i, i).copied().unwrap_or(0.0);
            assert!(
                (m_diag - 1.0).abs() < 1e-14,
                "M[{i},{i}] = {m_diag:.6}, expected 1.0 (L[{i},{i}] = {l_diag:.6})"
            );
        }

        // Off-diagonal: M[i,j] = -L[i,j] and M[i,j] >= 0
        for (l_val, (r, c)) in l.iter() {
            if r != c {
                let m_val = m.get(r, c).copied().unwrap_or(0.0);
                assert!(
                    (m_val - (-l_val)).abs() < 1e-14,
                    "M[{r},{c}] = {m_val:.6}, expected {:.6}", -l_val
                );
                assert!(m_val >= -1e-14, "M[{r},{c}] = {m_val:.6} should be ≥ 0");
            }
        }

        // Symmetry: M[i,j] == M[j,i]
        for (val, (r, c)) in m.iter() {
            let sym = m.get(c, r).copied().unwrap_or(0.0);
            assert!(
                (val - sym).abs() < 1e-14,
                "Symmetry: M[{r},{c}]={val:.6} != M[{c},{r}]={sym:.6}"
            );
        }
    }

    /// Test 2: rsvd_solve output shapes and eigenvalue range for a 5-node complete graph.
    #[test]
    fn test_rsvd_solve_shapes_complete_5() {
        let n = 5;
        let edges: Vec<(usize, usize, f32)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j, 1.0f32)))
            .collect();
        let graph = make_graph(n, &edges);
        let (_, sqrt_deg) = crate::laplacian::compute_degrees(&graph, ComputeMode::PythonCompat);
        let inv_sqrt_deg = sqrt_deg.mapv(|x| if x == 0.0 { 0.0 } else { 1.0 / x });
        let l = build_normalized_laplacian(&graph, inv_sqrt_deg.as_slice().unwrap());

        let n_components = 2;
        let (eigenvalues, eigenvectors) = rsvd_solve(&l, n_components, 42);

        assert_eq!(eigenvalues.shape(), &[n_components + 1], "eigenvalues shape");
        assert_eq!(eigenvectors.shape(), &[n, n_components + 1], "eigenvectors shape");

        // Eigenvalue range: [0, 2] for normalized Laplacian
        for &lambda in eigenvalues.iter() {
            assert!(
                lambda >= -1e-10 && lambda <= 2.0 + 1e-10,
                "eigenvalue {lambda:.6} out of [0, 2]"
            );
        }

        // Orthonormality: |V^T V - I| < 1e-10  (all n_components+1 columns)
        let ncols = n_components + 1;
        let vt_v = eigenvectors.t().dot(&eigenvectors);
        for i in 0..ncols {
            for j in 0..ncols {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (vt_v[[i, j]] - expected).abs();
                assert!(
                    diff < 1e-10,
                    "V^T V [{i},{j}] = {:.6}, expected {expected:.1} (diff {diff:.2e})",
                    vt_v[[i, j]]
                );
            }
        }
    }

    /// Test 3: rsvd_solve residual check on a 6-node ring graph with analytical eigenvalues.
    ///
    /// For a 6-node ring with uniform weight 1.0, all degrees = 2.
    /// Symmetric normalized Laplacian eigenvalues: 1 - cos(2πk/6), k=0..5
    ///   λ_0 = 0 (trivial), λ_1 = λ_2 = 0.5 (smallest non-trivial).
    #[test]
    fn test_rsvd_solve_residuals_ring_6() {
        let n = 6;
        let edges: Vec<(usize, usize, f32)> =
            (0..n).map(|i| (i, (i + 1) % n, 1.0f32)).collect();
        let graph = make_graph(n, &edges);
        let (_, sqrt_deg) = crate::laplacian::compute_degrees(&graph, ComputeMode::PythonCompat);
        let inv_sqrt_deg = sqrt_deg.mapv(|x| if x == 0.0 { 0.0 } else { 1.0 / x });
        let l = build_normalized_laplacian(&graph, inv_sqrt_deg.as_slice().unwrap());

        let n_components = 2;
        let (eigenvalues, eigenvectors) = rsvd_solve(&l, n_components, 0);

        let eig_slice = eigenvalues.as_slice().unwrap();

        // Index 0 is the trivial eigenvector (λ ≈ 0)
        assert!(
            eig_slice[0].abs() < 1e-3,
            "trivial eigenvalue={:.8}, expected ≈ 0", eig_slice[0]
        );

        // Non-trivial eigenvalues of the 6-ring are 0.5 (indices 1..=n_components)
        for i in 1..=n_components {
            let err = (eig_slice[i] - 0.5_f64).abs();
            assert!(
                err < 1e-4,
                "eigenvalue[{i}] = {:.8}, expected 0.5, err = {:.2e} (threshold 1e-4)",
                eig_slice[i], err
            );
        }

        // Residual checks for non-trivial eigenpairs (index 1..=n_components)
        for i in 1..=n_components {
            let v = eigenvectors.column(i).to_owned();
            let lambda = eig_slice[i];

            // L·v via SpMV
            let lv = &l * &v;
            let lambda_v = v.mapv(|x| lambda * x);
            let diff = &lv - &lambda_v;
            let norm_v = v.dot(&v).sqrt();
            let residual = diff.dot(&diff).sqrt() / norm_v;

            assert!(
                residual < 1e-4,
                "residual[{i}] = {residual:.2e} exceeds 1e-4 (lambda={lambda:.6})"
            );
        }
    }
}
