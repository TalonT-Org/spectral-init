//! Parity between the scalar and the SIMD solver paths.

#![cfg(target_arch = "x86_64")]

use crate::{fixtures, internals};
use spectral_init::__internal::{SolverParams, normalize_signs, scale_and_add_noise};
use spectral_init::ComputeMode;

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

/// `RustNative` (AVX2 SpMV) and `PythonCompat` (scalar SpMV) must produce the same scaled
/// coordinates once the per-column sign is normalized.
#[test]
fn rust_native_and_python_compat_solver_outputs_agree() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!(
            "rust_native_and_python_compat_solver_outputs_agree: skipping — AVX2/FMA not \
             available; ComputeMode::RustNative uses the scalar path, making the comparison vacuous"
        );
        return;
    }

    let params = SolverParams::default();
    for &fixture in FIXTURES {
        let lap_path = fixtures::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = fixtures::load_sparse_csr(&lap_path);

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
