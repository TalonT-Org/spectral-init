use ndarray::{Array2, ArrayView2};
use sprs::CsMatI;

// ─── Trait ───────────────────────────────────────────────────────────────────

pub(crate) trait LinearOperator {
    /// Apply operator: y = self * x. Both arrays have shape [n, k].
    fn apply(&self, x: ArrayView2<f64>, y: &mut Array2<f64>);
    /// Number of rows (and columns — operator is square).
    fn size(&self) -> usize;
}

// ─── Standalone SpMV (SIMD boundary) ─────────────────────────────────────────

/// CSR sparse-matrix × dense-vector product: y = A * x.
///
/// This is the Phase 3 SIMD replacement point. The raw-slice signature exposes
/// the memory layout that AVX-512 / SELL-C-sigma intrinsics require. Do not
/// change the signature; replace the body in Phase 3.
pub(crate) fn spmv_csr(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    debug_assert!(
        indptr.len() == y.len() + 1,
        "CSR invariant violated: indptr.len()={} != y.len()+1={}",
        indptr.len(),
        y.len() + 1
    );
    debug_assert!(
        indices.iter().all(|&j| j < x.len()),
        "CSR invariant violated: column index out of bounds (x.len()={})",
        x.len()
    );
    debug_assert!(
        data.len() >= indptr.last().copied().unwrap_or(0),
        "CSR invariant violated: data.len()={} < nnz={}",
        data.len(),
        indptr.last().copied().unwrap_or(0)
    );
    for i in 0..y.len() {
        let mut acc = 0.0_f64;
        for k in indptr[i]..indptr[i + 1] {
            acc += data[k] * x[indices[k]];
        }
        y[i] = acc;
    }
}

// ─── CsrOperator ─────────────────────────────────────────────────────────────

/// Wraps a borrowed CSR Laplacian and implements `LinearOperator` via sprs
/// sparse-dense multiply.
pub(crate) struct CsrOperator<'a>(pub(crate) &'a CsMatI<f64, usize>);

impl<'a> LinearOperator for CsrOperator<'a> {
    fn apply(&self, x: ArrayView2<f64>, y: &mut Array2<f64>) {
        // Zero output before accumulation: csr_mulacc_dense_rowmaj adds into out.
        y.fill(0.0);
        sprs::prod::csr_mulacc_dense_rowmaj(self.0.view(), x, y.view_mut());
    }

    fn size(&self) -> usize {
        self.0.rows()
    }
}

// Note: `spmv_csr` is not called from `apply` — `csr_mulacc_dense_rowmaj` handles
// the block case more efficiently (no per-column Vec allocations). It is exercised
// directly via unit tests.

// ─── Unit tests ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    // Build the 3×3 path-graph Laplacian [[2,-1,0],[-1,2,-1],[0,-1,2]] as CSR.
    fn laplacian_3x3() -> CsMatI<f64, usize> {
        CsMatI::new(
            (3, 3),
            vec![0usize, 2, 5, 7],
            vec![0usize, 1, 0, 1, 2, 1, 2],
            vec![2.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
    }

    #[test]
    fn test_spmv_identity_matrix() {
        // 4×4 identity: indptr=[0,1,2,3,4], indices=[0,1,2,3], data=[1,1,1,1]
        let indptr = vec![0usize, 1, 2, 3, 4];
        let indices = vec![0usize, 1, 2, 3];
        let data = vec![1.0_f64; 4];
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut y = vec![0.0_f64; 4];
        spmv_csr(&indptr, &indices, &data, &x, &mut y);
        for (yi, xi) in y.iter().zip(x.iter()) {
            assert!((yi - xi).abs() < 1e-15, "identity spmv: {yi} != {xi}");
        }
    }

    #[test]
    fn test_spmv_diagonal_matrix() {
        // 4×4 diagonal with [1,2,3,4], x=[1,1,1,1] → y=[1,2,3,4]
        let indptr = vec![0usize, 1, 2, 3, 4];
        let indices = vec![0usize, 1, 2, 3];
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = vec![1.0_f64; 4];
        let mut y = vec![0.0_f64; 4];
        spmv_csr(&indptr, &indices, &data, &x, &mut y);
        let expected = [1.0_f64, 2.0, 3.0, 4.0];
        for (yi, &ei) in y.iter().zip(expected.iter()) {
            assert!((yi - ei).abs() < 1e-15, "diagonal spmv: {yi} != {ei}");
        }
    }

    #[test]
    fn test_csr_operator_single_vector() {
        let mat = laplacian_3x3();
        let op = CsrOperator(&mat);

        // x is a 3×1 matrix with column [1, 0, 0]
        let x: Array2<f64> = array![[1.0], [0.0], [0.0]];
        let mut y: Array2<f64> = Array2::zeros((3, 1));
        op.apply(x.view(), &mut y);

        // Dense reference: A * [1,0,0]^T = [2, -1, 0]
        let expected: Array2<f64> = array![[2.0], [-1.0], [0.0]];
        for i in 0..3 {
            assert!(
                (y[[i, 0]] - expected[[i, 0]]).abs() < 1e-15,
                "single vector: y[{i}]={} expected {}",
                y[[i, 0]],
                expected[[i, 0]]
            );
        }
    }

    #[test]
    fn test_csr_operator_block_matches_sequential() {
        let mat = laplacian_3x3();
        let op = CsrOperator(&mat);

        // 3×2 block
        let x_block: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let mut y_block: Array2<f64> = Array2::zeros((3, 2));
        op.apply(x_block.view(), &mut y_block);

        // Apply to each column independently
        let x0: Array2<f64> = array![[1.0], [0.0], [0.0]];
        let x1: Array2<f64> = array![[0.0], [1.0], [0.0]];
        let mut y0: Array2<f64> = Array2::zeros((3, 1));
        let mut y1: Array2<f64> = Array2::zeros((3, 1));
        op.apply(x0.view(), &mut y0);
        op.apply(x1.view(), &mut y1);

        for i in 0..3 {
            assert!(
                (y_block[[i, 0]] - y0[[i, 0]]).abs() < 1e-15,
                "block col0 mismatch at row {i}: {} vs {}",
                y_block[[i, 0]],
                y0[[i, 0]]
            );
            assert!(
                (y_block[[i, 1]] - y1[[i, 0]]).abs() < 1e-15,
                "block col1 mismatch at row {i}: {} vs {}",
                y_block[[i, 1]],
                y1[[i, 0]]
            );
        }
    }

    #[test]
    fn test_csr_operator_residual_diagonal() {
        // 4×4 diagonal with [0.0, 0.5, 1.0, 1.5].
        // Unit vectors e_i are exact eigenvectors with eigenvalue = diagonal[i].
        let mat = CsMatI::<f64, usize>::new(
            (4, 4),
            vec![0usize, 1, 2, 3, 4],
            vec![0usize, 1, 2, 3],
            vec![0.0_f64, 0.5, 1.0, 1.5],
        );
        let op = CsrOperator(&mat);
        let eigenvalues = [0.0_f64, 0.5, 1.0, 1.5];

        for (i, &lam) in eigenvalues.iter().enumerate() {
            // e_i: unit vector with 1 at position i
            let mut x_arr = Array2::<f64>::zeros((4, 1));
            x_arr[[i, 0]] = 1.0;
            let mut lv = Array2::<f64>::zeros((4, 1));
            op.apply(x_arr.view(), &mut lv); // lv = L * e_i

            // residual = ||L*e_i - λ*e_i|| / ||e_i||
            let mut residual_sq = 0.0_f64;
            for r in 0..4 {
                let diff = lv[[r, 0]] - lam * x_arr[[r, 0]];
                residual_sq += diff * diff;
            }
            let residual = residual_sq.sqrt();
            assert!(
                residual < 1e-14,
                "residual for eigenpair {i}: {residual} >= 1e-14"
            );
        }
    }

    #[test]
    fn test_spmv_matches_dense_reference() {
        // 8×8 tridiagonal Laplacian: diag=2, off-diag=-1.
        // CSR indptr, indices, data (hardcoded for determinism).
        let indptr = vec![0usize, 2, 5, 8, 11, 14, 17, 20, 22];
        let indices = vec![
            0usize, 1, // row 0
            0, 1, 2, // row 1
            1, 2, 3, // row 2
            2, 3, 4, // row 3
            3, 4, 5, // row 4
            4, 5, 6, // row 5
            5, 6, 7, // row 6
            6, 7,   // row 7
        ];
        let data = vec![
            2.0_f64, -1.0, // row 0
            -1.0, 2.0, -1.0, // row 1
            -1.0, 2.0, -1.0, // row 2
            -1.0, 2.0, -1.0, // row 3
            -1.0, 2.0, -1.0, // row 4
            -1.0, 2.0, -1.0, // row 5
            -1.0, 2.0, -1.0, // row 6
            -1.0, 2.0,      // row 7
        ];

        // Fixed input vector (hardcoded for determinism).
        let x_vec: Vec<f64> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        // Compute sparse result.
        let mut y_sparse = vec![0.0_f64; 8];
        spmv_csr(&indptr, &indices, &data, &x_vec, &mut y_sparse);

        // Compute dense reference via ndarray.
        let mut a_dense = Array2::<f64>::zeros((8, 8));
        for row in 0..8 {
            a_dense[[row, row]] = 2.0;
            if row > 0 {
                a_dense[[row, row - 1]] = -1.0;
            }
            if row < 7 {
                a_dense[[row, row + 1]] = -1.0;
            }
        }
        let x_arr = Array1::<f64>::from_vec(x_vec.clone());
        let y_dense = a_dense.dot(&x_arr);

        // Assert max absolute difference < 1e-14.
        for (i, (ys, yd)) in y_sparse.iter().zip(y_dense.iter()).enumerate() {
            assert!(
                (ys - yd).abs() < 1e-14,
                "spmv vs dense mismatch at index {i}: sparse={ys}, dense={yd}"
            );
        }
    }
}
