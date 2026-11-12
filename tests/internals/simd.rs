//! SIMD soundness and parity: the AVX2 SpMV kernel agrees with the scalar one, the two
//! compute modes agree end to end, and the safe operator around the kernel refuses or
//! panics instead of reading out of bounds.

#![cfg(target_arch = "x86_64")]

use crate::{diagnostics, fixtures, internals};
use ndarray::Array2;
use spectral_init::__internal::{
    CsrOperator, CsrOperatorSimd, LinearOperator, SolverParams, normalize_signs,
    scale_and_add_noise, simd_index_fits, spmv_avx2_gather, spmv_csr,
};
use spectral_init::sprs::CsMatI;
use spectral_init::{ComputeMode, SpectralInitConfig, spectral_init};
use std::panic::{AssertUnwindSafe, catch_unwind};

const FIXTURES: &[&str] = &[
    "blobs_50",
    "near_dupes_100",
    "moons_200",
    "blobs_connected_200",
    "disconnected_200",
    "circles_300",
    "blobs_500",
    "blobs_connected_2000",
    "blobs_5000",
];

/// Largest |scalar − AVX2| accepted for one SpMV output element. The kernels differ in
/// summation order and in fused versus separate multiply-add; on the fixture Laplacians
/// (entries of magnitude at most 1, a few dozen per row) that amounts to a few units in
/// the last place of an O(1) sum.
const SPMV_MAX_ABS_DIFF: f64 = 1e-14;

fn avx2_fma_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

fn fixture_laplacian(fixture: &str) -> CsMatI<f64, usize> {
    fixtures::load_sparse_csr(&fixtures::fixture_path(fixture, "comp_b_laplacian.npz"))
}

/// A deterministic dense vector with entries in [-1, 1).
fn test_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7919 + 13) % 2000) as f64 / 1000.0 - 1.0)
        .collect()
}

/// `RustNative` (AVX2 SpMV) and `PythonCompat` (scalar SpMV) must produce the same scaled
/// coordinates once the per-column sign is normalized.
#[test]
fn rust_native_and_python_compat_solver_outputs_agree() {
    if !avx2_fma_available() {
        eprintln!(
            "rust_native_and_python_compat_solver_outputs_agree: skipping — AVX2/FMA not \
             available; ComputeMode::RustNative uses the scalar path, making the comparison vacuous"
        );
        return;
    }

    let params = SolverParams::default();
    for &fixture in FIXTURES {
        let laplacian = fixture_laplacian(fixture);

        let ((_, mut eigvec_scalar), _) =
            internals::solve_with_unit_hint(&laplacian, 2, 42, ComputeMode::PythonCompat, &params);
        let ((_, mut eigvec_avx2), _) =
            internals::solve_with_unit_hint(&laplacian, 2, 42, ComputeMode::RustNative, &params);

        normalize_signs(&mut eigvec_scalar);
        normalize_signs(&mut eigvec_avx2);

        let scaled_scalar =
            scale_and_add_noise(eigvec_scalar, 42).expect("scale_and_add_noise failed");
        let scaled_avx2 = scale_and_add_noise(eigvec_avx2, 42).expect("scale_and_add_noise failed");

        let f32_max_abs_diff = scaled_scalar
            .iter()
            .zip(scaled_avx2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // 4 * f32::EPSILON ≈ 4.77e-7; allows one rounding-budget above zero for
        // any residual scalar/SIMD divergence that survives normalize_signs.
        assert!(
            f32_max_abs_diff <= 4.77e-7,
            "fixture {fixture}: max_abs_diff={f32_max_abs_diff:.3e} > 4.77e-7 after normalize_signs — \
             sign flip not resolved; SIMD kernel must be gated behind ComputeMode::RustNative"
        );
    }
}

/// The AVX2 kernel and the scalar kernel compute the same product on every fixture
/// Laplacian, through the raw-slice entry and through the operator.
#[test]
fn avx2_spmv_matches_scalar_on_fixture_laplacians() {
    if !avx2_fma_available() {
        eprintln!(
            "avx2_spmv_matches_scalar_on_fixture_laplacians: skipping — AVX2/FMA not available"
        );
        return;
    }

    let mut widest_row_with_tail = 0;
    for &fixture in FIXTURES {
        let laplacian = fixture_laplacian(fixture);
        let n = laplacian.rows();
        widest_row_with_tail = laplacian
            .outer_iterator()
            .map(|row| row.nnz())
            .filter(|nnz| nnz % 4 != 0)
            .fold(widest_row_with_tail, usize::max);

        let indptr = laplacian.proper_indptr();
        let x = test_vector(n);
        let mut y_scalar = vec![0.0_f64; n];
        let mut y_avx2 = vec![0.0_f64; n];
        spmv_csr(
            &indptr,
            laplacian.indices(),
            laplacian.data(),
            &x,
            &mut y_scalar,
        );
        // SAFETY: AVX2 and FMA were detected above. `indptr` (zero-based), `indices` and
        // `data` are the storage of a valid `n × n` CSR matrix: `indptr` has `n + 1`
        // non-decreasing entries ending at the length of `indices` and `data`, and
        // every column index is below `n`, the length of `x`, far below `i32::MAX`.
        unsafe {
            spmv_avx2_gather(
                &indptr,
                laplacian.indices(),
                laplacian.data(),
                &x,
                &mut y_avx2,
            )
        };

        let x_block = Array2::from_shape_vec((n, 1), x).expect("n values");
        let mut y_operator = Array2::zeros((n, 1));
        CsrOperatorSimd::new(&laplacian)
            .expect("a square CSR Laplacian on an AVX2 machine")
            .apply(x_block.view(), &mut y_operator);

        for i in 0..n {
            assert!(
                (y_avx2[i] - y_scalar[i]).abs() <= SPMV_MAX_ABS_DIFF,
                "{fixture} row {i}: avx2={}, scalar={}",
                y_avx2[i],
                y_scalar[i]
            );
            assert_eq!(
                y_operator[[i, 0]],
                y_avx2[i],
                "{fixture} row {i}: the operator must run the same kernel"
            );
        }
    }
    // A row of nine or more non-zeros that is not a multiple of four runs the four-wide
    // loop at least twice and then the scalar tail.
    assert!(
        widest_row_with_tail >= 9,
        "no fixture row exercises both the vector loop and the scalar tail"
    );
}

/// Both compute modes span the same embedding subspace through the public entry point.
#[test]
fn rust_native_and_python_compat_agree_end_to_end() {
    for &fixture in FIXTURES {
        let graph = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(
            fixture,
            "step5a_pruned.npz",
        ));
        let run = |mode| {
            let config = SpectralInitConfig::new(2)
                .with_seed(42)
                .with_compute_mode(mode);
            spectral_init(graph.view(), None, &config)
                .unwrap_or_else(|e| panic!("{fixture} {mode:?}: {e}"))
                .mapv(f64::from)
        };
        let python_compat = run(ComputeMode::PythonCompat);
        let rust_native = run(ComputeMode::RustNative);
        let gram_det = diagnostics::subspace_gram_det(rust_native.view(), python_compat.view());
        assert!(
            gram_det > 0.999,
            "{fixture}: subspace Gram determinant {gram_det}"
        );
    }
}

#[test]
fn simd_operator_is_refused_for_non_square_or_csc_input() {
    let rect = CsMatI::<f64, usize>::new((2, 3), vec![0, 1, 2], vec![0, 2], vec![1.0, 1.0]);
    assert!(CsrOperatorSimd::new(&rect).is_none(), "non-square");

    let square = fixtures::ring_laplacian(8);
    let csc = square.to_csc();
    assert!(csc.is_csc());
    assert!(CsrOperatorSimd::new(&csc).is_none(), "CSC storage");

    assert_eq!(
        CsrOperatorSimd::new(&square).is_some(),
        avx2_fma_available(),
        "a square CSR matrix is accepted exactly when the CPU has AVX2 and FMA"
    );
}

#[test]
fn simd_index_fits_up_to_i32_max_columns() {
    assert!(simd_index_fits(i32::MAX as usize));
    assert!(!simd_index_fits(i32::MAX as usize + 1));
}

/// `apply` must panic, not read out of bounds, when a vector is shorter than the matrix.
#[test]
fn simd_operator_apply_panics_on_mismatched_lengths() {
    let laplacian = fixtures::ring_laplacian(16);
    let Some(op) = CsrOperatorSimd::new(&laplacian) else {
        eprintln!(
            "simd_operator_apply_panics_on_mismatched_lengths: skipping — AVX2/FMA not available"
        );
        return;
    };
    assert_eq!(op.size(), 16);

    let short_x = Array2::<f64>::ones((15, 1));
    let mut y = Array2::<f64>::zeros((16, 1));
    let result = catch_unwind(AssertUnwindSafe(|| op.apply(short_x.view(), &mut y)));
    assert!(result.is_err(), "apply accepted an x with too few rows");

    let x = Array2::<f64>::ones((16, 1));
    let mut short_y = Array2::<f64>::zeros((15, 1));
    let result = catch_unwind(AssertUnwindSafe(|| op.apply(x.view(), &mut short_y)));
    assert!(result.is_err(), "apply accepted a y with too few rows");

    let mut long_y = Array2::<f64>::zeros((17, 1));
    let result = catch_unwind(AssertUnwindSafe(|| op.apply(x.view(), &mut long_y)));
    assert!(result.is_err(), "apply accepted a y with too many rows");

    // Matching lengths still work, and agree with the scalar operator.
    let mut y_scalar = Array2::<f64>::zeros((16, 1));
    CsrOperator(&laplacian).apply(x.view(), &mut y_scalar);
    op.apply(x.view(), &mut y);
    for i in 0..16 {
        assert!((y[[i, 0]] - y_scalar[[i, 0]]).abs() <= SPMV_MAX_ABS_DIFF);
    }
}

/// The raw-slice kernel entry checks nothing, so it has to be an `unsafe fn`: were it ever
/// made safe, the `unsafe` block below would be unused and this test would stop compiling.
#[deny(unused_unsafe)]
#[test]
fn raw_kernel_entry_is_an_unsafe_fn() {
    if !avx2_fma_available() {
        eprintln!("raw_kernel_entry_is_an_unsafe_fn: skipping — AVX2/FMA not available");
        return;
    }
    // One row of nine non-zeros: two four-wide steps and a scalar tail of one.
    let indptr = [0usize, 9];
    let indices: Vec<usize> = (0..9).collect();
    let data = vec![1.0_f64; 9];
    let x: Vec<f64> = (1..=9).map(f64::from).collect();
    let mut y = [0.0_f64];
    // SAFETY: AVX2 and FMA were detected above; `indptr` has `y.len() + 1` non-decreasing
    // entries ending at 9, the length of `indices` and `data`; every column index is
    // below `x.len()` = 9.
    unsafe { spmv_avx2_gather(&indptr, &indices, &data, &x, &mut y) };
    assert_eq!(y[0], 45.0);
}
