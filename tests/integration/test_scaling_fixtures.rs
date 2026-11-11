#[path = "../common/mod.rs"]
mod common;

use ndarray::Array2;
use spectral_init::{scale_and_add_noise_pub, scale_coords};

fn load_embedding(dataset: &str) -> Array2<f64> {
    common::load_dense(
        &common::fixture_path(dataset, "comp_e_selection.npz"),
        "embedding",
    )
}

fn load_pre_noise(dataset: &str) -> Array2<f32> {
    common::load_dense(
        &common::fixture_path(dataset, "comp_f_scaling.npz"),
        "pre_noise",
    )
}

fn check_pre_noise_exact(dataset: &str) {
    let embedding = load_embedding(dataset);
    let expected_pre_noise = load_pre_noise(dataset);

    let result = scale_coords(&embedding, 10.0).expect("scale_coords failed");

    // Verify element-wise exact equality
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            approx::assert_abs_diff_eq!(
                result[[i, j]],
                expected_pre_noise[[i, j]],
                epsilon = 0.0f32
            );
        }
    }

    // Verify max absolute value ≈ 10.0
    let max_abs = result.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
    // Tolerance is 1e-6f32: scale_coords computes expansion = 10.0 / max_abs_f64,
    // then casts each element to f32. The maximum element maps to exactly 10.0
    // modulo f32 rounding (~9.5e-7, one ULP at 10.0 = 2^(3-23)); 1e-6 is just
    // above that single-ULP bound.
    approx::assert_abs_diff_eq!(max_abs, 10.0f32, epsilon = 1e-6f32);
}

#[test]
fn test_pre_noise_exact_blobs_50() {
    check_pre_noise_exact("blobs_50");
}

#[test]
fn test_pre_noise_exact_parametrized() {
    check_pre_noise_exact("moons_200");
    check_pre_noise_exact("circles_300");
}

#[test]
fn test_noise_distribution_blobs_50() {
    let embedding = load_embedding("blobs_50");
    let n = embedding.len();

    let final_result =
        scale_and_add_noise_pub(embedding.clone(), 42).expect("scale_and_add_noise failed");
    let pre_noise = scale_coords(&embedding, 10.0).expect("scale_coords failed");

    let noise: Array2<f32> = &final_result - &pre_noise;
    let noise_flat: Vec<f32> = noise.iter().cloned().collect();

    let mean: f32 = noise_flat.iter().sum::<f32>() / n as f32;
    let variance: f32 = noise_flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std_dev = variance.sqrt();

    // 3-sigma test for mean ≈ 0
    let three_sigma = 3.0 * 0.0001_f32 / (n as f32).sqrt();
    assert!(
        mean.abs() < three_sigma,
        "noise mean {mean} exceeds 3-sigma bound {three_sigma}"
    );

    // std within 20% of 0.0001
    let relative_err = (std_dev - 0.0001f32).abs() / 0.0001f32;
    assert!(
        relative_err < 0.20,
        "noise std {std_dev} is not within 20% of 0.0001 (relative error: {relative_err})"
    );
}

#[test]
fn test_max_abs_is_10() {
    for dataset in &["blobs_50", "moons_200", "circles_300"] {
        let embedding = load_embedding(dataset);
        let pre_noise = scale_coords(&embedding, 10.0).expect("scale_coords failed");
        let max_abs = pre_noise
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0.0f32, f32::max);

        // Tolerance is 1e-6f32: deterministic f64→f32 cast; max element rounds
        // to 10.0 with at most ~9.5e-7 (f32 ULP at 10.0 = 2^(3-23)); 1e-6 gives headroom.
        approx::assert_abs_diff_eq!(max_abs, 10.0f32, epsilon = 1e-6f32);
    }
}

macro_rules! make_scale_test {
    ($name:ident, $dataset:literal) => {
        #[test]
        fn $name() {
            check_pre_noise_exact($dataset);
        }
    };
}

make_scale_test!(scale_pre_noise_matches_blobs_500, "blobs_500");
make_scale_test!(scale_pre_noise_matches_blobs_5000, "blobs_5000");
make_scale_test!(
    scale_pre_noise_matches_blobs_connected_200,
    "blobs_connected_200"
);
make_scale_test!(
    scale_pre_noise_matches_blobs_connected_2000,
    "blobs_connected_2000"
);
make_scale_test!(scale_pre_noise_matches_near_dupes_100, "near_dupes_100");

#[test]
fn scale_max_abs_is_10_disconnected_200() {
    let pre_noise = load_pre_noise("disconnected_200");
    assert_eq!(pre_noise.shape(), &[200, 2]);
    assert!(pre_noise.iter().all(|v| v.is_finite()));
    let max_abs = pre_noise.iter().copied().map(f32::abs).fold(0f32, f32::max);
    assert!((max_abs - 10.0f32).abs() < 1e-6, "max_abs={max_abs}"); // f32 ULP at 10.0 ≈ 9.5e-7
}
