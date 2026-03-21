use ndarray::{Array1, Array2};
use sprs::CsMatI;
use faer::{Mat, Side};
use crate::SpectralError;
use super::EigenResult;

/// Dense eigendecomposition via faer for small n.
/// Returns (eigenvalues shape [k], eigenvectors shape [n, k]) sorted ascending by eigenvalue.
pub(crate) fn dense_evd(
    laplacian: &CsMatI<f64, usize>,
    k: usize,
) -> Result<EigenResult, SpectralError> {
    let n = laplacian.rows();

    // Convert sparse Laplacian to dense faer matrix
    let mut dense = Mat::<f64>::zeros(n, n);
    for (val, (row, col)) in laplacian.iter() {
        *dense.get_mut(row, col) = *val;
    }

    // Compute full symmetric eigendecomposition (eigenvalues in nondecreasing order)
    let evd = dense
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| SpectralError::ConvergenceFailure)?;

    let s = evd.S(); // DiagRef<f64> — eigenvalues in nondecreasing order
    let u = evd.U(); // MatRef<f64> shape [n, n] — columns are eigenvectors

    // k smallest eigenvalues → Array1<f64>
    let eigenvalues = Array1::from_iter(
        s.column_vector().iter().take(k).copied()
    );

    // Corresponding eigenvectors → Array2<f64> shape [n, k]
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
        let diff_norm: f64 = lv.iter().zip(eigvec.iter())
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
        let (eigenvalues, eigenvectors) = dense_evd(&l, 2).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 2);
        assert!(
            (eigenvalues[0] - 0.0).abs() < 1e-14,
            "eigenvalue[0] = {}, expected 0.0", eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 2.0).abs() < 1e-14,
            "eigenvalue[1] = {}, expected 2.0", eigenvalues[1]
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
        let (eigenvalues, eigenvectors) = dense_evd(&l, 3).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 3);
        for i in 0..3 {
            assert!(
                (eigenvalues[i] - 1.0).abs() < 1e-14,
                "eigenvalue[{i}] = {}, expected 1.0", eigenvalues[i]
            );
            let r = residual(&l, eigenvectors.column(i), eigenvalues[i]);
            assert!(r < 1e-14, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn dense_evd_truncation_k_less_than_n() {
        // 3x3 path graph Laplacian: L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
        let l = make_csr(3, &[
            (0, 0, 1.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 1.0),
        ]);
        let (eigenvalues, eigenvectors) = dense_evd(&l, 2).expect("dense_evd failed");

        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.shape(), &[3, 2]);

        // Eigenvalues must be in ascending order
        assert!(eigenvalues[0] <= eigenvalues[1], "eigenvalues not sorted");

        for j in 0..2 {
            let r = residual(&l, eigenvectors.column(j), eigenvalues[j]);
            assert!(r < 1e-13, "residual[{j}] = {r}");
        }
    }

    #[test]
    fn dense_evd_zero_eigenvalue_path_graph() {
        // Same 3x3 path graph — smallest eigenvalue must be 0.0 (PSD, not PD)
        let l = make_csr(3, &[
            (0, 0, 1.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 1.0),
        ]);
        let (eigenvalues, _) = dense_evd(&l, 2).expect("dense_evd must not error on singular PSD");

        assert!(
            eigenvalues[0].abs() < 1e-13,
            "smallest eigenvalue = {}, expected 0.0", eigenvalues[0]
        );
    }
}
