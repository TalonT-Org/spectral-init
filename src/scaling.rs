use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Internal helper: deterministic scaling only (no noise).
/// Scales `coords` so the global max absolute value equals `max_coord`, then
/// casts to f32.
pub(crate) fn scale_coords(coords: &Array2<f64>, max_coord: f64) -> Array2<f32> {
    let max_abs = coords.iter().cloned().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        max_abs > 0.0,
        "scale_coords: embedding is all-zero; cannot scale to max={max_coord}"
    );
    let expansion = max_coord / max_abs;
    coords.mapv(|x| (x * expansion) as f32)
}

/// Internal helper: scale then add per-element Gaussian noise.
fn noisy_scale_coords(
    coords: &Array2<f64>,
    rng: &mut impl rand::Rng,
    max_coord: f64,
    noise_scale: f64,
) -> Array2<f32> {
    let mut scaled = scale_coords(coords, max_coord);
    let normal = Normal::<f32>::new(0.0, noise_scale as f32)
        .expect("noise_scale must be finite and positive");
    for elem in scaled.iter_mut() {
        *elem += normal.sample(rng);
    }
    scaled
}

/// Scales columns so max absolute value is 10.0, then adds Gaussian noise
/// with scale=0.0001. Downcasts to f32.
pub(crate) fn scale_and_add_noise(coords: Array2<f64>, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    noisy_scale_coords(&coords, &mut rng, 10.0, 0.0001)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array0;
    use ndarray_npy::NpzReader;
    use std::path::PathBuf;

    fn fixture_path(dataset: &str, file: &str) -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .join("tests")
            .join("fixtures")
            .join(dataset)
            .join(file)
    }

    fn npz_f64_array2(path: &PathBuf, key: &str) -> Option<ndarray::Array2<f64>> {
        let file = std::fs::File::open(path).ok()?;
        let mut reader = NpzReader::new(file).ok()?;
        reader.by_name(key).ok()
    }

    fn npz_f32_array2(path: &PathBuf, key: &str) -> Option<ndarray::Array2<f32>> {
        let file = std::fs::File::open(path).ok()?;
        let mut reader = NpzReader::new(file).ok()?;
        reader.by_name(key).ok()
    }

    fn npz_f64_scalar(path: &PathBuf, key: &str) -> Option<f64> {
        let file = std::fs::File::open(path).ok()?;
        let mut reader = NpzReader::new(file).ok()?;
        let arr: Array0<f64> = reader.by_name(key).ok()?;
        Some(arr.into_scalar())
    }

    fn check_pre_noise_exact(dataset: &str) {
        let e_path = fixture_path(dataset, "comp_e_selection.npz");
        let f_path = fixture_path(dataset, "comp_f_scaling.npz");

        if !e_path.exists() || !f_path.exists() {
            eprintln!(
                "check_pre_noise_exact({dataset}): fixture files absent, skipping"
            );
            return;
        }

        let embedding = npz_f64_array2(&e_path, "embedding").expect("embedding key missing");
        let expected_pre_noise =
            npz_f32_array2(&f_path, "pre_noise").expect("pre_noise key missing");
        let expansion = npz_f64_scalar(&f_path, "expansion").expect("expansion key missing");

        let result = scale_coords(&embedding, 10.0);

        // Verify element-wise exact equality
        let shape = result.shape().to_vec();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
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
        approx::assert_abs_diff_eq!(max_abs, 10.0f32, epsilon = 1e-5f32);

        // Verify expansion matches Python
        let max_abs_f64 = embedding
            .iter()
            .cloned()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let expected_expansion = 10.0 / max_abs_f64;
        approx::assert_abs_diff_eq!(expansion, expected_expansion, epsilon = 1e-12);
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
        let e_path = fixture_path("blobs_50", "comp_e_selection.npz");
        if !e_path.exists() {
            eprintln!("test_noise_distribution_blobs_50: fixture absent, skipping");
            return;
        }

        let embedding = npz_f64_array2(&e_path, "embedding").expect("embedding key missing");
        let n = embedding.len();

        let final_result = scale_and_add_noise(embedding.clone(), 42);
        let pre_noise = scale_coords(&embedding, 10.0);

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
    fn test_max_abs_is_10() {
        for dataset in &["blobs_50", "moons_200", "circles_300"] {
            let e_path = fixture_path(dataset, "comp_e_selection.npz");
            if !e_path.exists() {
                eprintln!("test_max_abs_is_10({dataset}): fixture absent, skipping");
                continue;
            }

            let embedding = npz_f64_array2(&e_path, "embedding").expect("embedding key missing");
            let pre_noise = scale_coords(&embedding, 10.0);
            let max_abs = pre_noise
                .iter()
                .cloned()
                .map(f32::abs)
                .fold(0.0f32, f32::max);

            approx::assert_abs_diff_eq!(
                max_abs,
                10.0f32,
                epsilon = 1e-5f32
            );
        }
    }
}
