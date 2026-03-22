use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Internal helper: deterministic scaling only (no noise).
/// Scales `coords` so the global max absolute value equals `max_coord`, then
/// casts to f32.
pub(crate) fn scale_coords(
    coords: &Array2<f64>,
    max_coord: f64,
) -> Result<Array2<f32>, crate::SpectralError> {
    let max_abs = coords.iter().cloned().map(f64::abs).fold(0.0_f64, f64::max);
    if max_abs <= 0.0 {
        return Err(crate::SpectralError::InvalidGraph(format!(
            "scale_coords: embedding is all-zero; cannot scale to max={max_coord}"
        )));
    }
    let expansion = max_coord / max_abs;
    Ok(coords.mapv(|x| (x * expansion) as f32))
}

/// Internal helper: scale then add per-element Gaussian noise.
pub(crate) fn noisy_scale_coords(
    coords: &Array2<f64>,
    rng: &mut impl rand::Rng,
    max_coord: f64,
    noise_scale: f64,
) -> Result<Array2<f32>, crate::SpectralError> {
    if !noise_scale.is_finite() || noise_scale <= 0.0 {
        return Err(crate::SpectralError::InvalidGraph(format!(
            "noise_scale must be finite and positive, got {noise_scale}"
        )));
    }
    let noise_scale_f32 = noise_scale as f32;
    if noise_scale_f32 <= 0.0 {
        return Err(crate::SpectralError::InvalidGraph(format!(
            "noise_scale {noise_scale} underflows to zero when cast to f32"
        )));
    }
    let mut scaled = scale_coords(coords, max_coord)?;
    let normal = Normal::<f32>::new(0.0, noise_scale_f32)
        .expect("noise_scale_f32 is validated positive and finite");
    for elem in scaled.iter_mut() {
        *elem += normal.sample(rng);
    }
    Ok(scaled)
}

/// Scales columns so max absolute value is 10.0, then adds Gaussian noise
/// with scale=0.0001. Downcasts to f32.
pub(crate) fn scale_and_add_noise(
    coords: Array2<f64>,
    seed: u64,
) -> Result<Array2<f32>, crate::SpectralError> {
    if coords.is_empty() {
        return Err(crate::SpectralError::InvalidGraph(
            "scale_and_add_noise: input embedding is empty".to_string(),
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    noisy_scale_coords(&coords, &mut rng, 10.0, 0.0001)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::{NpzReader, ReadableElement};
    use std::path::PathBuf;

    fn fixture_path(dataset: &str, file: &str) -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .join("tests")
            .join("fixtures")
            .join(dataset)
            .join(file)
    }

    fn npz_array2<T: ReadableElement>(path: &PathBuf, key: &str) -> Option<ndarray::Array2<T>> {
        let file = std::fs::File::open(path).ok()?;
        let mut reader = NpzReader::new(file).ok()?;
        reader.by_name(key).ok()
    }

    fn check_pre_noise_exact(dataset: &str) {
        let e_path = fixture_path(dataset, "comp_e_selection.npz");
        let f_path = fixture_path(dataset, "comp_f_scaling.npz");

        if !e_path.exists() || !f_path.exists() {
            panic!("check_pre_noise_exact({dataset}): fixture files absent");
        }

        let embedding = npz_array2::<f64>(&e_path, "embedding").expect("embedding key missing");
        let expected_pre_noise =
            npz_array2::<f32>(&f_path, "pre_noise").expect("pre_noise key missing");

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
        let max_abs = result
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        // Tolerance is 1e-6f32: scale_coords computes expansion = 10.0 / max_abs_f64,
        // then casts each element to f32. The maximum element maps to exactly 10.0
        // modulo f32 rounding (~9.5e-7, one ULP at 10.0 = 2^(3-23)); 1e-6 is just
        // above that single-ULP bound.
        approx::assert_abs_diff_eq!(max_abs, 10.0f32, epsilon = 1e-6f32);
    }

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_pre_noise_exact_blobs_50() {
        check_pre_noise_exact("blobs_50");
    }

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_pre_noise_exact_parametrized() {
        check_pre_noise_exact("moons_200");
        check_pre_noise_exact("circles_300");
    }

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_noise_distribution_blobs_50() {
        let e_path = fixture_path("blobs_50", "comp_e_selection.npz");
        if !e_path.exists() {
            panic!("test_noise_distribution_blobs_50: fixture absent");
        }

        let embedding = npz_array2::<f64>(&e_path, "embedding").expect("embedding key missing");
        let n = embedding.len();

        let final_result = scale_and_add_noise(embedding.clone(), 42).expect("scale_and_add_noise failed");
        let pre_noise = scale_coords(&embedding, 10.0).expect("scale_coords failed");

        let noise: ndarray::Array2<f32> = &final_result - &pre_noise;
        let noise_flat: Vec<f32> = noise.iter().cloned().collect();

        let mean: f32 = noise_flat.iter().sum::<f32>() / n as f32;
        let variance: f32 =
            noise_flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
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
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_max_abs_is_10() {
        for dataset in &["blobs_50", "moons_200", "circles_300"] {
            let e_path = fixture_path(dataset, "comp_e_selection.npz");
            if !e_path.exists() {
                panic!("test_max_abs_is_10({dataset}): fixture absent");
            }

            let embedding =
                npz_array2::<f64>(&e_path, "embedding").expect("embedding key missing");
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
            #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
            fn $name() {
                check_pre_noise_exact($dataset);
            }
        };
    }

    make_scale_test!(scale_pre_noise_matches_blobs_500,            "blobs_500");
    make_scale_test!(scale_pre_noise_matches_blobs_5000,           "blobs_5000");
    make_scale_test!(scale_pre_noise_matches_blobs_connected_200,  "blobs_connected_200");
    make_scale_test!(scale_pre_noise_matches_blobs_connected_2000, "blobs_connected_2000");
    make_scale_test!(scale_pre_noise_matches_near_dupes_100,       "near_dupes_100");

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn scale_max_abs_is_10_disconnected_200() {
        let path = fixture_path("disconnected_200", "comp_f_scaling.npz");
        let pre_noise: ndarray::Array2<f32> =
            npz_array2::<f32>(&path, "pre_noise").expect("pre_noise key missing");
        assert_eq!(pre_noise.shape(), &[200, 2]);
        assert!(pre_noise.iter().all(|v| v.is_finite()));
        let max_abs = pre_noise.iter().copied().map(f32::abs).fold(0f32, f32::max);
        assert!((max_abs - 10.0f32).abs() < 1e-6, "max_abs={max_abs}"); // f32 ULP at 10.0 ≈ 9.5e-7
    }
}
