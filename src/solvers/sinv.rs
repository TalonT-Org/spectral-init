// linfa-linalg 0.2 uses ndarray 0.16, while this crate uses ndarray 0.17.
// We alias ndarray 0.16 as `nd16` for use at the linfa-linalg boundary only.
extern crate ndarray16;
use ndarray16 as nd16;

use linfa_linalg::lobpcg::{lobpcg, Order};
use ndarray::{Array1, Array2, ArrayView2};
use sprs::CsMatI;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use faer::prelude::Solve;

use super::EigenResult;

/// Diagonal shift ε making M = L + εI strictly SPD (REQ-SINV-002).
pub(crate) const SINV_SHIFT: f64 = 1e-4;

// ─── ndarray 0.16 ↔ 0.17 conversion helpers ──────────────────────────────────

fn to_nd16_array2(a: Array2<f64>) -> nd16::Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let raw: Vec<f64> = a.iter().copied().collect();
    nd16::Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd17→nd16 Array2")
}

fn from_nd16_view2(v: nd16::ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = (v.nrows(), v.ncols());
    let raw: Vec<f64> = v.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd16→nd17 ArrayView2")
}

fn from_nd16_array1(a: nd16::Array1<f64>) -> Array1<f64> {
    Array1::from_iter(a.iter().copied())
}

fn from_nd16_array2(a: nd16::Array2<f64>) -> Array2<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let raw: Vec<f64> = a.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), raw)
        .expect("shape/data mismatch converting nd16→nd17 Array2")
}

// ─── sprs ↔ faer sparse conversion ───────────────────────────────────────────

/// Convert a sprs CsMat<f64> (CSR or CSC) to a faer SparseColMat<usize, f64>,
/// adding `diag_shift` to every diagonal entry (used to build M = L + εI).
///
/// The conversion goes CSR/CSC → sprs CSC → raw arrays → faer CSC.
fn sprs_csc_to_faer(
    mat: &CsMatI<f64, usize>,
    diag_shift: f64,
) -> faer::sparse::SparseColMat<usize, f64> {
    let n = mat.rows();
    let csc = mat.to_csc();
    let (col_ptr, row_idx, mut values) = csc.into_raw_storage();

    // Apply diagonal shift: for each column col, find entries where row_idx == col.
    for col in 0..n {
        for k in col_ptr[col]..col_ptr[col + 1] {
            if row_idx[k] == col {
                values[k] += diag_shift;
            }
        }
    }

    let symbolic = faer::sparse::SymbolicSparseColMat::<usize>::new_checked(
        n, n, col_ptr, None, row_idx,
    );
    faer::sparse::SparseColMat::<usize, f64>::new(symbolic, values)
}

// ─── ndarray 0.17 ↔ faer dense bridge helpers ────────────────────────────────

/// Convert ndarray 0.17 ArrayView2<f64> [n, k] to a faer Mat<f64> [n, k].
/// Copies element-by-element to handle ndarray row-major vs faer column-major.
fn nd17_view_to_faer(x: ArrayView2<f64>) -> faer::Mat<f64> {
    let (nrows, ncols) = x.dim();
    faer::Mat::<f64>::from_fn(nrows, ncols, |i, j| x[[i, j]])
}

/// Convert a faer Mat<f64> [n, k] to ndarray 0.17 Array2<f64>.
fn faer_to_nd17(x: &faer::Mat<f64>) -> Array2<f64> {
    let (nrows, ncols) = (x.nrows(), x.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| x[(i, j)])
}

/// Convert a faer Mat<f64> [n, k] to ndarray 0.16 Array2<f64>.
fn faer_to_nd17_nd16(x: &faer::Mat<f64>) -> nd16::Array2<f64> {
    to_nd16_array2(faer_to_nd17(x))
}

// ─── Shift-and-Invert LOBPCG ──────────────────────────────────────────────────

/// Shift-and-invert LOBPCG eigensolver (Level 2).
///
/// Pre-factorizes M = L + εI via sparse Cholesky, runs LOBPCG with M⁻¹ as
/// operator (Order::Largest) to find the k largest eigenvalues of M⁻¹, which
/// are the k smallest eigenvalues of L. Recovers original Laplacian eigenvalues
/// via λ_k = 1/μ_k − ε.
///
/// Returns `None` if any precondition fails or if Cholesky factorization fails.
pub fn lobpcg_sinv_solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    sqrt_deg: &Array1<f64>,
) -> Option<EigenResult> {
    let n = laplacian.rows();
    let k = n_components + 1;

    // Guard conditions mirror lobpcg_solve.
    if n_components == 0 || k >= n || sqrt_deg.len() != n {
        return None;
    }

    // Build faer shifted matrix M = L + SINV_SHIFT·I.
    let m_faer = sprs_csc_to_faer(laplacian, SINV_SHIFT);

    // Sparse Cholesky factorization of M (REQ-SINV-001).
    // If Cholesky fails (M not SPD despite shift — pathological case), return None.
    let chol = m_faer.sp_cholesky(faer::Side::Lower).ok()?;

    // Build random initial block [n, k] with trivial eigenvector injected at col 0.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_init_17: Array2<f64> =
        Array2::from_shape_fn((n, k), |_| StandardNormal.sample(&mut rng));
    let sqrt_deg_norm = sqrt_deg.dot(sqrt_deg).sqrt();
    if sqrt_deg_norm > 0.0 {
        x_init_17.column_mut(0).assign(&sqrt_deg.mapv(|x| x / sqrt_deg_norm));
    }
    let x_init = to_nd16_array2(x_init_17);

    // Run LOBPCG with M⁻¹ operator — finds k largest eigenvalues of M⁻¹,
    // i.e. the k smallest eigenvalues of L (REQ-SINV-003).
    let result = lobpcg(
        |x: nd16::ArrayView2<f64>| -> nd16::Array2<f64> {
            let x17 = from_nd16_view2(x);
            let mut rhs = nd17_view_to_faer(x17.view());
            // solve_in_place: chol · y = rhs, overwrite rhs with y = M⁻¹ · rhs.
            chol.solve_in_place(&mut rhs);
            faer_to_nd17_nd16(&rhs)
        },
        x_init,
        |_: nd16::ArrayViewMut2<f64>| {},
        None,
        1e-8_f32,
        (n * 5).min(300),
        Order::Largest,
    );

    // Extract result; accept partially-converged results if all residuals are small.
    let extract = |r: linfa_linalg::lobpcg::Lobpcg<f64>| -> EigenResult {
        (from_nd16_array1(r.eigvals), from_nd16_array2(r.eigvecs))
    };

    let result_opt = match result {
        Ok(r) => Some(extract(r)),
        Err((_, Some(r))) if r.rnorm.iter().all(|&norm| norm < 1e-6) => Some(extract(r)),
        _ => None,
    };

    // Eigenvalue recovery (REQ-SINV-004): λ_k = 1/μ_k − ε
    // LOBPCG Order::Largest returns eigenvalues in some order; sort ascending.
    result_opt.map(|(mut eigvals, mut eigvecs)| {
        eigvals.mapv_inplace(|mu| 1.0 / mu - SINV_SHIFT);

        // Sort ascending by recovered eigenvalue; permute eigvec columns accordingly.
        let mut pairs: Vec<(f64, usize)> = eigvals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_vals = Array1::from_iter(pairs.iter().map(|&(v, _)| v));
        let sorted_vecs = Array2::from_shape_fn(
            (eigvecs.nrows(), eigvecs.ncols()),
            |(i, j)| eigvecs[[i, pairs[j].1]],
        );
        // Update eigvecs in-place by overwriting from sorted_vecs.
        eigvecs.assign(&sorted_vecs);
        (sorted_vals, eigvecs)
    })
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Build a diagonal n×n CSR Laplacian with the given diagonal values.
    fn diagonal_csr(values: &[f64]) -> CsMatI<f64, usize> {
        let n = values.len();
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        CsMatI::new((n, n), indptr, indices, values.to_vec())
    }

    // T1 — sprs_csc_to_faer_identity_roundtrip
    #[test]
    fn sprs_csc_to_faer_identity_roundtrip() {
        let mat = diagonal_csr(&[1.0, 1.0, 1.0, 1.0]);
        let faer_mat = sprs_csc_to_faer(&mat, 0.0);
        assert_eq!(faer_mat.nrows(), 4);
        assert_eq!(faer_mat.ncols(), 4);
        assert_eq!(faer_mat.val().len(), 4, "nnz should be 4 for 4×4 identity");
    }

    // T2 — sprs_csc_to_faer_shifted_diagonal
    #[test]
    fn sprs_csc_to_faer_shifted_diagonal() {
        let mat = diagonal_csr(&[1.0, 2.0, 3.0, 4.0]);
        let faer_mat = sprs_csc_to_faer(&mat, 0.5);
        let vals = faer_mat.val();
        // CSC of a diagonal matrix has one entry per column; entries are in column order.
        let expected = [1.5, 2.5, 3.5, 4.5];
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-15,
                "val[{i}] = {v}, expected {}", expected[i]
            );
        }
    }

    // T3 — sinv_solve_diagonal_eigenvalues_accurate
    #[test]
    fn sinv_solve_diagonal_eigenvalues_accurate() {
        // n=6, eigenvalues [1e-8, 0.1, 0.3, 0.7, 1.2, 2.0].
        let diag = [1e-8_f64, 0.1, 0.3, 0.7, 1.2, 2.0];
        let mat = diagonal_csr(&diag);
        let sqrt_deg = Array1::ones(6);

        let result = lobpcg_sinv_solve(&mat, 2, 42, &sqrt_deg);
        assert!(result.is_some(), "lobpcg_sinv_solve returned None on diagonal matrix");
        let (eigvals, eigvecs) = result.unwrap();

        // First two non-trivial eigenvalues match [0.1, 0.3] within 1e-8.
        assert!(
            (eigvals[1] - 0.1).abs() < 1e-8,
            "eigvals[1] = {}, expected ≈ 0.1", eigvals[1]
        );
        assert!(
            (eigvals[2] - 0.3).abs() < 1e-8,
            "eigvals[2] = {}, expected ≈ 0.3", eigvals[2]
        );

        // All residuals < 1e-8.
        for i in 0..eigvals.len() {
            let r = crate::metrics::eigenpair_residual(&mat, &eigvecs.column(i).to_owned(), eigvals[i]);
            assert!(r < 1e-8, "residual for eigenpair {i}: {r} >= 1e-8");
        }
    }

    // T4 — sinv_solve_returns_k_plus_one_pairs
    #[test]
    fn sinv_solve_returns_k_plus_one_pairs() {
        let n = 30;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
        let mat = diagonal_csr(&diag);
        let ones = Array1::ones(n);

        let result = lobpcg_sinv_solve(&mat, 3, 42, &ones);
        assert!(result.is_some(), "lobpcg_sinv_solve returned None for n=30");
        let (eigvals, eigvecs) = result.unwrap();

        assert_eq!(eigvals.len(), 4, "expected k+1=4 eigenvalues, got {}", eigvals.len());
        assert_eq!(
            eigvecs.shape(), &[n, 4],
            "expected [{n}, 4] eigvecs, got {:?}", eigvecs.shape()
        );
    }

    // T5 — sinv_solve_eigenvectors_orthonormal
    #[test]
    fn sinv_solve_eigenvectors_orthonormal() {
        let n = 30;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
        let mat = diagonal_csr(&diag);
        let ones = Array1::ones(n);

        let result = lobpcg_sinv_solve(&mat, 3, 42, &ones);
        assert!(result.is_some(), "lobpcg_sinv_solve returned None");
        let (_eigvals, eigvecs) = result.unwrap();

        let k = eigvecs.ncols();
        let gram = eigvecs.t().dot(&eigvecs);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = gram[[i, j]];
                assert!(
                    (got - expected).abs() < 1e-8,
                    "Gram matrix [{i},{j}] = {got}, expected {expected}"
                );
            }
        }
    }

    // T6 — eigenvalue_recovery_formula_exact
    #[test]
    fn eigenvalue_recovery_formula_exact() {
        // Verify λ = 1/μ − ε exactly in f64 (pure arithmetic, no solver call).
        let lambdas = [0.0_f64, 0.2, 0.5];
        for &lambda in &lambdas {
            let mu = 1.0 / (lambda + SINV_SHIFT);
            let recovered = 1.0 / mu - SINV_SHIFT;
            assert!(
                (recovered - lambda).abs() < 1e-15,
                "recovery error for λ={lambda}: got {recovered}, diff={}", (recovered - lambda).abs()
            );
        }
    }

    // T7 — sinv_returns_none_for_empty_or_degenerate
    #[test]
    fn sinv_returns_none_for_empty_or_degenerate() {
        let n = 6;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
        let mat = diagonal_csr(&diag);
        let ones = Array1::ones(n);

        // n_components=0 guard
        assert!(
            lobpcg_sinv_solve(&mat, 0, 42, &ones).is_none(),
            "expected None for n_components=0"
        );
        // k >= n guard: k = n_components+1 = n means k >= n
        assert!(
            lobpcg_sinv_solve(&mat, n - 1, 42, &ones).is_none(),
            "expected None when k >= n"
        );
    }
}
