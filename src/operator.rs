use ndarray::{Array2, ArrayView2};
use sprs::CsMatI;

// ─── Trait ───────────────────────────────────────────────────────────────────

#[doc(hidden)]
pub trait LinearOperator {
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
///
/// `pub` under the `testing` feature (bench/test access) with always-on
/// precondition assertions; `pub(crate)` otherwise with `debug_assert!` guards.
#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn spmv_csr(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    assert!(
        indptr.len() == y.len() + 1,
        "CSR invariant violated: indptr.len()={} != y.len()+1={}",
        indptr.len(),
        y.len() + 1
    );
    assert!(
        indices.iter().all(|&j| j < x.len()),
        "CSR invariant violated: column index out of bounds (x.len()={})",
        x.len()
    );
    assert!(
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

#[cfg(not(feature = "testing"))]
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
#[doc(hidden)]
pub struct CsrOperator<'a>(pub &'a CsMatI<f64, usize>);

impl<'a> LinearOperator for CsrOperator<'a> {
    fn apply(&self, x: ArrayView2<f64>, y: &mut Array2<f64>) {
        debug_assert_eq!(
            x.shape()[0], self.0.rows(),
            "CsrOperator::apply: x row count {} != matrix rows {}",
            x.shape()[0], self.0.rows()
        );
        debug_assert_eq!(
            y.shape()[0], self.0.rows(),
            "CsrOperator::apply: y row count {} != matrix rows {}",
            y.shape()[0], self.0.rows()
        );
        y.fill(0.0);
        let k = x.shape()[1];
        if k == 1 {
            // Single-vector path: routes through the spmv_csr raw-slice SpMV kernel.
            // Avoids column buffer allocation when the layout is contiguous; falls back
            // to a Vec for strided column views.
            let mat = self.0;
            // x: use contiguous slice directly when possible; collect otherwise.
            let x_col0 = x.column(0);
            let x_col_vec: Vec<f64>;
            let x_col: &[f64] = match x_col0.as_slice() {
                Some(s) => s,
                None => {
                    x_col_vec = x_col0.iter().copied().collect();
                    &x_col_vec
                }
            };
            // y: write into the column slice directly when possible; scatter otherwise.
            match y.column_mut(0).as_slice_mut() {
                Some(y_col) => {
                    spmv_csr(mat.indptr().raw_storage(), mat.indices(), mat.data(), x_col, y_col);
                }
                None => {
                    let mut y_col = vec![0.0_f64; mat.rows()];
                    spmv_csr(mat.indptr().raw_storage(), mat.indices(), mat.data(), x_col, &mut y_col);
                    for (i, v) in y_col.into_iter().enumerate() {
                        y[[i, 0]] = v;
                    }
                }
            }
        } else {
            // Block-vector path: csr_mulacc_dense_rowmaj handles k>1 efficiently
            // (no per-column allocations, single row-major pass over the matrix).
            sprs::prod::csr_mulacc_dense_rowmaj(self.0.view(), x, y.view_mut());
        }
    }

    fn size(&self) -> usize {
        self.0.rows()
    }
}

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
    fn test_spmv_matches_csr_operator_single_vector() {
        // Cross-validate: spmv_csr and CsrOperator::apply must agree for k=1.
        let mat = laplacian_3x3();
        let op = CsrOperator(&mat);

        let x_vec = vec![1.0_f64, -1.0, 2.0];
        let x_arr: Array2<f64> = ndarray::Array1::from(x_vec.clone())
            .insert_axis(ndarray::Axis(1));
        let mut y_op = Array2::zeros((3, 1));
        op.apply(x_arr.view(), &mut y_op);

        // Direct spmv_csr call
        let mut y_spmv = vec![0.0_f64; 3];
        spmv_csr(
            mat.indptr().raw_storage(),
            mat.indices(),
            mat.data(),
            &x_vec,
            &mut y_spmv,
        );

        for i in 0..3 {
            assert!(
                (y_op[[i, 0]] - y_spmv[i]).abs() < 1e-15,
                "spmv_csr vs CsrOperator mismatch at row {i}: op={}, spmv={}",
                y_op[[i, 0]],
                y_spmv[i]
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

    // ─── Fixture-gated tests ─────────────────────────────────────────────────

    /// Load the blobs_connected_200 Laplacian (f64 CSR with i32 indices) from a fixture NPZ.
    fn load_fixture_laplacian() -> CsMatI<f64, usize> {
        use ndarray_npy::NpzReader;
        use std::path::Path;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/blobs_connected_200/comp_b_laplacian.npz");
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("cannot open {:?}: {}", path, e));
        let mut npz = NpzReader::new(file)
            .unwrap_or_else(|e| panic!("NpzReader error for {:?}: {}", path, e));
        let data: Vec<f64> = npz
            .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("data")
            .unwrap()
            .into_iter()
            .collect();
        let indices: Vec<usize> = npz
            .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices")
            .unwrap()
            .iter()
            .map(|&x| x as usize)
            .collect();
        let indptr: Vec<usize> = npz
            .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr")
            .unwrap()
            .iter()
            .map(|&x| x as usize)
            .collect();
        let shape_arr: ndarray::Array1<i64> = npz.by_name("shape").unwrap();
        assert_eq!(shape_arr.len(), 2, "shape array must have 2 elements");
        let rows = shape_arr[0] as usize;
        let cols = shape_arr[1] as usize;
        CsMatI::try_new((rows, cols), indptr, indices, data)
            .expect("fixture Laplacian CSR invalid")
    }

    #[test]
    #[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
    fn test_eigenpair_residual_from_fixture() {
        use ndarray_npy::NpzReader;
        use std::path::Path;

        let laplacian = load_fixture_laplacian();
        let n = laplacian.rows();

        // Load eigenvalues and eigenvectors from comp_d_eigensolver.npz
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/blobs_connected_200/comp_d_eigensolver.npz");
        let file = std::fs::File::open(&path).unwrap();
        let mut npz = NpzReader::new(file).unwrap();
        let eigenvalues: ndarray::Array1<f64> = npz.by_name("eigenvalues").unwrap();
        let eigenvectors: ndarray::Array2<f64> = npz.by_name("eigenvectors").unwrap();
        let k = eigenvalues.len();
        assert_eq!(eigenvectors.shape()[0], n);
        assert_eq!(eigenvectors.shape()[1], k);

        // For each eigenpair, assert residual ||L*v - λ*v|| / ||v|| < 1e-10.
        // These are Python reference eigenvectors computed by numpy.linalg.eigh,
        // so near-exact residuals are expected.
        for j in 0..k {
            let v = eigenvectors.column(j);
            let lambda = eigenvalues[j];
            let mut lv = vec![0.0f64; n];
            for (val, (row, col)) in laplacian.iter() {
                lv[row] += val * v[col];
            }
            let diff_norm: f64 = lv
                .iter()
                .zip(v.iter())
                .map(|(&lvi, &vi)| (lvi - lambda * vi).powi(2))
                .sum::<f64>()
                .sqrt();
            let v_norm: f64 = v.iter().map(|&vi| vi.powi(2)).sum::<f64>().sqrt();
            let residual = diff_norm / v_norm.max(1e-300);
            assert!(
                residual < 1e-10,
                "eigenpair {j}: residual={residual:.3e} >= 1e-10 (lambda={lambda:.6e})"
            );
        }
    }

    #[test]
    #[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
    fn test_multi_vector_matches_sequential_from_fixture() {
        use ndarray_npy::NpzReader;
        use std::path::Path;

        let mat = load_fixture_laplacian();
        let op = CsrOperator(&mat);
        let n = mat.rows();

        // Load eigenvectors as input matrix
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/blobs_connected_200/comp_d_eigensolver.npz");
        let file = std::fs::File::open(&path).unwrap();
        let mut npz = NpzReader::new(file).unwrap();
        let eigenvectors: ndarray::Array2<f64> = npz.by_name("eigenvectors").unwrap();
        let k = eigenvectors.shape()[1];

        // Apply CsrOperator to the full (n, k) matrix at once.
        let mut y_block = ndarray::Array2::<f64>::zeros((n, k));
        op.apply(eigenvectors.view(), &mut y_block);

        // Apply column-by-column and compare.
        for j in 0..k {
            let x_col = eigenvectors
                .column(j)
                .to_owned()
                .insert_axis(ndarray::Axis(1));
            let mut y_col = ndarray::Array2::<f64>::zeros((n, 1));
            op.apply(x_col.view(), &mut y_col);
            for i in 0..n {
                assert!(
                    (y_block[[i, j]] - y_col[[i, 0]]).abs() < 1e-14,
                    "block vs sequential mismatch at row={i} col={j}: \
                     block={} seq={}",
                    y_block[[i, j]],
                    y_col[[i, 0]]
                );
            }
        }
    }
}
