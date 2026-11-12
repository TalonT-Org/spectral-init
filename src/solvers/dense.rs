//! Dense eigendecomposition via faer (Levels 0 and 5 of the solver escalation chain).
//!
//! Converts the sparse Laplacian to a dense `faer::Mat<f64>`, calls `self_adjoint_eigen`,
//! and returns the `k` smallest eigenpairs sorted ascending.
//!
//! **Chain position:** Level 0 — exact, O(n³), tried first for `n < 2000` — and Level 5,
//! the last resort for any `n`.
//!
//! **Failure modes:** Returns [`SpectralError::DenseBudgetExceeded`] when the dense
//! `n × n` matrix would exceed the dense memory budget, and
//! [`SpectralError::ConvergenceFailure`] if faer's self-adjoint EVD fails (practically
//! impossible for a valid PSD matrix). The residual quality gate (`< 1e-6`) then
//! determines whether the chain accepts the result.

use super::{EigenResult, check_dense_budget};
use crate::SpectralError;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use sprs::CsMatI;

/// Dense eigendecomposition via faer.
/// Returns (eigenvalues of shape `[k]`, eigenvectors of shape `[n, k]`) sorted ascending by
/// eigenvalue.
///
/// `k` is the number of eigenpairs, i.e. the embedding dimensionality plus one. Errors
/// with `TooFewNodes` when `k > n` and with `DenseBudgetExceeded` when the dense `n × n`
/// matrix needs more than `dense_memory_budget` bytes.
pub fn dense_evd(
    laplacian: &CsMatI<f64, usize>,
    k: usize,
    dense_memory_budget: u64,
) -> Result<EigenResult, SpectralError> {
    let n = laplacian.rows();
    if k > n {
        return Err(SpectralError::TooFewNodes {
            n,
            n_components: k.saturating_sub(1),
        });
    }
    check_dense_budget(n, n, dense_memory_budget)?;

    let mut dense = Mat::<f64>::zeros(n, n);
    for (val, (row, col)) in laplacian.iter() {
        *dense.get_mut(row, col) = *val;
    }

    let evd = dense.self_adjoint_eigen(Side::Lower).map_err(|e| {
        log::warn!("dense_evd: faer self_adjoint_eigen failed: {e:?}");
        SpectralError::ConvergenceFailure { n }
    })?;

    let s = evd.S();
    let u = evd.U();

    let eigenvalues = Array1::from_iter(s.column_vector().iter().take(k).copied());

    let mut eigenvectors = Array2::<f64>::zeros((n, k));
    for j in 0..k {
        let col = u.col(j);
        for (i, &val) in col.iter().enumerate() {
            eigenvectors[[i, j]] = val;
        }
    }

    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_LIMIT: u64 = u64::MAX;

    fn make_csr(n: usize, triplets: &[(usize, usize, f64)]) -> CsMatI<f64, usize> {
        let mut mat = sprs::TriMat::new((n, n));
        for &(r, c, v) in triplets {
            mat.add_triplet(r, c, v);
        }
        mat.to_csr()
    }

    fn residual(
        laplacian: &CsMatI<f64, usize>,
        eigvec: ndarray::ArrayView1<f64>,
        eigval: f64,
    ) -> f64 {
        let n = laplacian.rows();
        let mut lv = vec![0.0f64; n];
        for (val, (row, col)) in laplacian.iter() {
            lv[row] += val * eigvec[col];
        }
        let diff_norm: f64 = lv
            .iter()
            .zip(eigvec.iter())
            .map(|(&lvi, &vi)| (lvi - eigval * vi).powi(2))
            .sum::<f64>()
            .sqrt();
        let v_norm: f64 = eigvec.iter().map(|&vi| vi.powi(2)).sum::<f64>().sqrt();
        diff_norm / v_norm
    }

    #[test]
    fn dense_evd_trivial_2x2() {
        // L = [[1, -1], [-1, 1]]  eigenvalues: 0.0, 2.0
        let l = make_csr(2, &[(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]);
        let (eigenvalues, eigenvectors) = dense_evd(&l, 2, NO_LIMIT).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 2);
        assert!(
            (eigenvalues[0] - 0.0).abs() < 1e-14,
            "eigenvalue[0] = {}, expected 0.0",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 2.0).abs() < 1e-14,
            "eigenvalue[1] = {}, expected 2.0",
            eigenvalues[1]
        );

        for j in 0..2 {
            let r = residual(&l, eigenvectors.column(j), eigenvalues[j]);
            assert!(r < 1e-14, "residual[{j}] = {r}");
        }
    }

    #[test]
    fn dense_evd_identity_3x3() {
        // L = I_3, all eigenvalues = 1.0
        let l = make_csr(3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let (eigenvalues, eigenvectors) = dense_evd(&l, 3, NO_LIMIT).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 3);
        for i in 0..3 {
            assert!(
                (eigenvalues[i] - 1.0).abs() < 1e-14,
                "eigenvalue[{i}] = {}, expected 1.0",
                eigenvalues[i]
            );
            let r = residual(&l, eigenvectors.column(i), eigenvalues[i]);
            assert!(r < 1e-14, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn dense_evd_truncation_k_less_than_n() {
        // 3x3 path graph Laplacian: L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
        // Smallest eigenvalue must be 0.0 (PSD, not PD — path graph is connected).
        let l = make_csr(
            3,
            &[
                (0, 0, 1.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 2.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 1.0),
            ],
        );
        let (eigenvalues, eigenvectors) = dense_evd(&l, 2, NO_LIMIT).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.shape(), &[3, 2]);
        assert!(eigenvalues[0] <= eigenvalues[1], "eigenvalues not sorted");
        assert!(
            eigenvalues[0].abs() < 1e-13,
            "smallest eigenvalue = {}, expected 0.0",
            eigenvalues[0]
        );

        for j in 0..2 {
            let r = residual(&l, eigenvectors.column(j), eigenvalues[j]);
            assert!(r < 1e-13, "residual[{j}] = {r}");
        }
    }

    #[test]
    fn dense_evd_k_greater_than_n_returns_error() {
        let l = make_csr(2, &[(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]);
        assert_eq!(
            dense_evd(&l, 3, NO_LIMIT),
            Err(SpectralError::TooFewNodes {
                n: 2,
                n_components: 2
            })
        );
    }

    #[test]
    fn dense_evd_respects_the_dense_memory_budget() {
        let l = make_csr(2, &[(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]);
        // A 2 × 2 f64 matrix needs 32 bytes.
        assert!(dense_evd(&l, 2, 32).is_ok());
        assert_eq!(
            dense_evd(&l, 2, 31),
            Err(SpectralError::DenseBudgetExceeded {
                required_bytes: 32,
                budget_bytes: 31
            })
        );
    }
}
