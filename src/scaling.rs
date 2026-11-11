use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Internal helper: deterministic scaling only (no noise).
/// Scales `coords` so the global max absolute value equals `max_coord`, then
/// casts to f32.
pub fn scale_coords(
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

/// Enforces a deterministic sign convention (argmax convention) on each eigenvector column.
///
/// For each column of `coords`, finds the element with the largest absolute value.
/// If that element is negative, the entire column is negated.  All-zero columns are
/// left unchanged.
///
/// Uses the argmax absolute value approach (as in scikit-learn's `_deterministic_vector_sign_flip`
/// and `svd_flip`), which is more robust than a first-element approach for structured graphs
/// where node 0 may have a near-zero Fiedler vector component.
///
/// Must be called on f64 eigenvectors after `select_eigenvectors` and before
/// `scale_and_add_noise`.
pub fn normalize_signs(coords: &mut Array2<f64>) {
    for col in 0..coords.ncols() {
        let sign = coords
            .column(col)
            .iter()
            .copied()
            .reduce(|a, b| {
                if b.abs().total_cmp(&a.abs()).is_gt() {
                    b
                } else {
                    a
                }
            })
            .map(|v| v.signum())
            .unwrap_or(1.0);
        if sign < 0.0 {
            coords.column_mut(col).iter_mut().for_each(|x| *x = -*x);
        }
    }
}

/// Scales columns so max absolute value is 10.0, then adds Gaussian noise
/// with scale=0.0001. Downcasts to f32.
pub fn scale_and_add_noise(
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
