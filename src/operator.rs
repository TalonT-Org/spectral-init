use ndarray::{Array2, ArrayView2};
use sprs::CsMatI;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// A square linear operator applied to blocks of column vectors.
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
/// The raw-slice signature exposes the CSR memory layout directly, so scalar and
/// SIMD kernels share one calling convention. The CSR invariants (`indptr.len() ==
/// y.len() + 1`, every column index below `x.len()`, `data` at least `nnz` long) are
/// checked with `debug_assert!`; violating them in a release build panics on the
/// out-of-bounds index.
#[doc(hidden)]
pub fn spmv_csr(indptr: &[usize], indices: &[usize], data: &[f64], x: &[f64], y: &mut [f64]) {
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

// ─── AVX2 gather kernel (x86_64 only) ────────────────────────────────────────

/// Whether the column indices of a matrix with `cols` columns fit the signed 32-bit
/// gather indices of the AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn simd_index_fits(cols: usize) -> bool {
    cols <= i32::MAX as usize
}

/// CSR sparse-matrix × dense-vector product `y = A * x` with an AVX2 gather and FMA.
///
/// # Safety
///
/// - The CPU supports AVX2 and FMA.
/// - `indptr` holds `y.len() + 1` non-decreasing entries, and `data` holds at least
///   `indptr[y.len()]` elements.
/// - Every column index in `indices[..indptr[y.len()]]` is below `x.len()` and at most
///   `i32::MAX`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn spmv_avx2_gather_inner(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    use std::arch::x86_64::*;

    let n = y.len();
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];

        // SAFETY: AVX2 and FMA are present (caller's contract). Every column index is
        // below `x.len()` and at most `i32::MAX`, so each `as i32` cast is lossless and
        // the gather reads four elements inside `x`. `base + 4 <= end <= indptr[n]` and
        // `data` holds at least `indptr[n]` elements, so the unaligned load reads
        // `data[base..base + 4]` in bounds.
        let (mut result, mut base) = unsafe {
            let mut acc = _mm256_setzero_pd();
            let mut base = start;

            while base + 4 <= end {
                let vi = _mm_set_epi32(
                    indices[base + 3] as i32,
                    indices[base + 2] as i32,
                    indices[base + 1] as i32,
                    indices[base] as i32,
                );
                let xv = _mm256_i32gather_pd(x.as_ptr(), vi, 8);
                let dv = _mm256_loadu_pd(data.as_ptr().add(base));
                acc = _mm256_fmadd_pd(dv, xv, acc);
                base += 4;
            }

            let lo = _mm256_castpd256_pd128(acc);
            let hi = _mm256_extractf128_pd(acc, 1);
            let sum128 = _mm_add_pd(lo, hi);
            let halved = _mm_hadd_pd(sum128, sum128);
            let result = _mm_cvtsd_f64(halved);

            (result, base)
        };

        while base < end {
            result += data[base] * x[indices[base]];
            base += 1;
        }

        y[i] = result;
    }
}

/// Raw-slice entry to the AVX2+FMA gather SpMV kernel, for tests.
///
/// # Safety
///
/// - The CPU supports AVX2 and FMA.
/// - `indptr` holds `y.len() + 1` non-decreasing entries, and `indices` and `data` each
///   hold at least `indptr[y.len()]` elements.
/// - Every column index in `indices[..indptr[y.len()]]` is below `x.len()` and at most
///   `i32::MAX`.
#[cfg(all(any(test, feature = "__internal"), target_arch = "x86_64"))]
#[doc(hidden)]
pub unsafe fn spmv_avx2_gather(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    // SAFETY: this function's contract is the kernel's.
    unsafe { spmv_avx2_gather_inner(indptr, indices, data, x, y) }
}

// ─── CsrOperator ─────────────────────────────────────────────────────────────

/// Wraps a borrowed CSR Laplacian and implements `LinearOperator` via sprs
/// sparse-dense multiply.
#[doc(hidden)]
pub struct CsrOperator<'a>(pub &'a CsMatI<f64, usize>);

impl<'a> LinearOperator for CsrOperator<'a> {
    fn apply(&self, x: ArrayView2<f64>, y: &mut Array2<f64>) {
        debug_assert_eq!(
            x.shape()[0],
            self.0.rows(),
            "CsrOperator::apply: x row count {} != matrix rows {}",
            x.shape()[0],
            self.0.rows()
        );
        debug_assert_eq!(
            y.shape()[0],
            self.0.rows(),
            "CsrOperator::apply: y row count {} != matrix rows {}",
            y.shape()[0],
            self.0.rows()
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
                    spmv_csr(
                        mat.indptr().raw_storage(),
                        mat.indices(),
                        mat.data(),
                        x_col,
                        y_col,
                    );
                }
                None => {
                    let mut y_col = vec![0.0_f64; mat.rows()];
                    spmv_csr(
                        mat.indptr().raw_storage(),
                        mat.indices(),
                        mat.data(),
                        x_col,
                        &mut y_col,
                    );
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

// ─── CsrOperatorSimd ─────────────────────────────────────────────────────────

/// SIMD-accelerated variant of `CsrOperator`. Routes k=1 through
/// `spmv_avx2_gather_inner` (AVX2 gather kernel); k>1 falls back to `csr_mulacc_dense_rowmaj`.
///
/// Only available on x86_64. The only way to build one is [`CsrOperatorSimd::new`], which
/// checks everything the `unsafe` SIMD kernel in `apply` relies on that can be checked
/// once per matrix.
#[cfg(target_arch = "x86_64")]
pub struct CsrOperatorSimd<'a>(&'a CsMatI<f64, usize>);

#[cfg(target_arch = "x86_64")]
impl<'a> CsrOperatorSimd<'a> {
    /// Wraps `mat` if the AVX2 kernel can run on it.
    ///
    /// Returns `None` unless `mat` is a square CSR matrix whose index pointer starts at
    /// zero, its column indices fit 32 bits ([`simd_index_fits`]), and the CPU supports
    /// AVX2 and FMA. Callers fall back to [`CsrOperator`] on `None`.
    pub fn new(mat: &'a CsMatI<f64, usize>) -> Option<Self> {
        let usable = mat.is_csr()
            && mat.rows() == mat.cols()
            && mat.indptr().is_proper()
            && simd_index_fits(mat.cols())
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");
        usable.then_some(Self(mat))
    }
}

#[cfg(target_arch = "x86_64")]
impl<'a> LinearOperator for CsrOperatorSimd<'a> {
    /// # Panics
    ///
    /// Panics if `x` does not have one row per matrix column or `y` one row per matrix
    /// row.
    fn apply(&self, x: ArrayView2<f64>, y: &mut Array2<f64>) {
        let mat = self.0;
        assert_eq!(
            x.nrows(),
            mat.cols(),
            "CsrOperatorSimd::apply: x has {} rows, the matrix has {} columns",
            x.nrows(),
            mat.cols()
        );
        assert_eq!(
            y.nrows(),
            mat.rows(),
            "CsrOperatorSimd::apply: y has {} rows, the matrix has {} rows",
            y.nrows(),
            mat.rows()
        );
        y.fill(0.0);
        let k = x.shape()[1];
        if k == 1 {
            let x_col0 = x.column(0);
            let x_col_vec: Vec<f64>;
            let x_col: &[f64] = match x_col0.as_slice() {
                Some(s) => s,
                None => {
                    x_col_vec = x_col0.iter().copied().collect();
                    &x_col_vec
                }
            };
            let kernel = |y_col: &mut [f64]| {
                // SAFETY:
                // - `new` detected AVX2 and FMA, and the private field admits no other
                //   constructor.
                // - `new` checked that `mat` is CSR with an index pointer starting at
                //   zero. With the structure invariants sprs maintains for every
                //   `CsMatI`, `indptr` then holds `rows + 1` non-decreasing entries
                //   ending at `nnz`, `indices` and `data` hold `nnz` elements, and every
                //   column index is below `cols`.
                // - `x_col.len() == cols` and `y_col.len() == rows` by the asserts above,
                //   and `cols <= i32::MAX` by `new`, so every column index is below
                //   `x_col.len()` and fits an `i32`.
                unsafe {
                    spmv_avx2_gather_inner(
                        mat.indptr().raw_storage(),
                        mat.indices(),
                        mat.data(),
                        x_col,
                        y_col,
                    );
                }
            };
            match y.column_mut(0).as_slice_mut() {
                Some(y_col) => kernel(y_col),
                None => {
                    let mut y_col = vec![0.0_f64; mat.rows()];
                    kernel(&mut y_col);
                    for (i, v) in y_col.into_iter().enumerate() {
                        y[[i, 0]] = v;
                    }
                }
            }
        } else {
            sprs::prod::csr_mulacc_dense_rowmaj(mat.view(), x, y.view_mut());
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
    use ndarray::{Array1, Array2, array};

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
        let x_arr: Array2<f64> = ndarray::Array1::from(x_vec.clone()).insert_axis(ndarray::Axis(1));
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
            6, 7, // row 7
        ];
        let data = vec![
            2.0_f64, -1.0, // row 0
            -1.0, 2.0, -1.0, // row 1
            -1.0, 2.0, -1.0, // row 2
            -1.0, 2.0, -1.0, // row 3
            -1.0, 2.0, -1.0, // row 4
            -1.0, 2.0, -1.0, // row 5
            -1.0, 2.0, -1.0, // row 6
            -1.0, 2.0, // row 7
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

    // ─── AVX2/SIMD operator tests ────────────────────────────────────────────

    /// Whether the build or the CPU provides the features the AVX2 kernel needs.
    #[cfg(target_arch = "x86_64")]
    fn avx2_fma_available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    /// A 12×12 CSR matrix whose rows hold between 1 and 12 non-zeros, so the kernel's
    /// four-wide loop runs zero to three times per row, with and without a scalar tail.
    #[cfg(target_arch = "x86_64")]
    fn wide_row_matrix() -> CsMatI<f64, usize> {
        let n = 12usize;
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            for j in 0..n {
                // Rows 0..6 hold columns 0..=i+8; rows 6..12 hold columns i..12.
                let stored = if i < 6 { j <= i + 8 } else { j >= i };
                if stored {
                    indices.push(j);
                    data.push(1.0 / (1.0 + (i * n + j) as f64));
                }
            }
            indptr.push(indices.len());
        }
        CsMatI::new((n, n), indptr, indices, data)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_spmv_avx2_gather_matches_scalar() {
        if !avx2_fma_available() {
            return; // skip on non-AVX2 machines
        }
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y_scalar = vec![0.0_f64; 3];
        let mut y_avx2 = vec![0.0_f64; 3];
        spmv_csr(&indptr, &indices, &data, &x, &mut y_scalar);
        // SAFETY: AVX2 and FMA were detected above; `indptr` has `y.len() + 1`
        // non-decreasing entries ending at 7, the length of `indices` and `data`; every
        // column index is below `x.len()` = 3.
        unsafe { spmv_avx2_gather(&indptr, &indices, &data, &x, &mut y_avx2) };
        for i in 0..3 {
            assert!(
                (y_avx2[i] - y_scalar[i]).abs() < 1e-13,
                "avx2 vs scalar mismatch at row {i}: avx2={}, scalar={}",
                y_avx2[i],
                y_scalar[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_csr_operator_simd_wide_rows_match_scalar() {
        // Skipped only where the kernel cannot run. A build with AVX2 and FMA enabled
        // at compile time never skips, so there the operator must be constructible.
        if !avx2_fma_available() {
            return;
        }
        let mat = wide_row_matrix();
        let row_nnz: Vec<usize> = mat.outer_iterator().map(|row| row.nnz()).collect();
        assert!(
            row_nnz.contains(&9),
            "need a row with two vector steps and a tail"
        );
        assert!(row_nnz.contains(&12), "need a row without a tail");
        assert!(row_nnz.contains(&3), "need a row with only a tail");

        let op_scalar = CsrOperator(&mat);
        let op_simd = CsrOperatorSimd::new(&mat).expect("AVX2 and FMA are available");
        let x = Array2::from_shape_fn((12, 1), |(i, _)| (i as f64 + 1.0) * 0.25);
        let mut y_scalar = Array2::zeros((12, 1));
        let mut y_simd = Array2::zeros((12, 1));
        op_scalar.apply(x.view(), &mut y_scalar);
        op_simd.apply(x.view(), &mut y_simd);
        for i in 0..12 {
            assert!(
                (y_simd[[i, 0]] - y_scalar[[i, 0]]).abs() < 1e-13,
                "row {i}: simd={}, scalar={}",
                y_simd[[i, 0]],
                y_scalar[[i, 0]]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_csr_operator_simd_rejects_unsupported_matrices() {
        // Non-square.
        let rect = CsMatI::<f64, usize>::new((2, 3), vec![0, 1, 2], vec![0, 2], vec![1.0, 1.0]);
        assert!(CsrOperatorSimd::new(&rect).is_none());
        // CSC storage.
        let csc = laplacian_3x3().to_csc();
        assert!(CsrOperatorSimd::new(&csc).is_none());
        // Index pointer not starting at zero: the last two rows of a 4×2 matrix.
        let tall = CsMatI::<f64, usize>::new(
            (4, 2),
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let tail = tall.slice_outer(2..4).to_owned();
        assert_eq!(tail.shape(), (2, 2));
        assert!(!tail.indptr().is_proper());
        assert!(CsrOperatorSimd::new(&tail).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_index_fits_is_bounded_by_i32_max() {
        assert!(simd_index_fits(0));
        assert!(simd_index_fits(i32::MAX as usize));
        assert!(!simd_index_fits(i32::MAX as usize + 1));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_csr_operator_simd_single_vector_matches_scalar() {
        let mat = laplacian_3x3();
        let op_scalar = CsrOperator(&mat);
        let Some(op_simd) = CsrOperatorSimd::new(&mat) else {
            return; // skip on non-AVX2 machines
        };
        let x: ndarray::Array2<f64> = ndarray::array![[1.0], [2.0], [3.0]];
        let mut y_scalar = ndarray::Array2::zeros((3, 1));
        let mut y_simd = ndarray::Array2::zeros((3, 1));
        op_scalar.apply(x.view(), &mut y_scalar);
        op_simd.apply(x.view(), &mut y_simd);
        for i in 0..3 {
            assert!(
                (y_simd[[i, 0]] - y_scalar[[i, 0]]).abs() < 1e-13,
                "CsrOperatorSimd vs CsrOperator mismatch at row {i}: simd={}, scalar={}",
                y_simd[[i, 0]],
                y_scalar[[i, 0]]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_csr_operator_simd_block_matches_sequential() {
        let mat = laplacian_3x3();
        let Some(op) = CsrOperatorSimd::new(&mat) else {
            return; // skip on non-AVX2 machines
        };
        let x_block: ndarray::Array2<f64> = ndarray::array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let mut y_block = ndarray::Array2::zeros((3, 2));
        op.apply(x_block.view(), &mut y_block);
        // Verify each column independently
        for col in 0..2usize {
            let x_col = x_block.column(col).to_owned().insert_axis(ndarray::Axis(1));
            let mut y_col = ndarray::Array2::zeros((3, 1));
            op.apply(x_col.view(), &mut y_col);
            for i in 0..3 {
                assert!(
                    (y_block[[i, col]] - y_col[[i, 0]]).abs() < 1e-13,
                    "simd block vs sequential mismatch at row={i} col={col}"
                );
            }
        }
    }
}
