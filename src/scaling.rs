use crate::SpectralError;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};

/// Largest absolute coordinate after scaling.
const MAX_COORD: f64 = 10.0;

/// Standard deviation of the Gaussian noise added to the scaled coordinates.
const NOISE_SCALE: f32 = 0.0001;

/// Internal helper: deterministic scaling only (no noise).
/// Scales `coords` so the global max absolute value equals `max_coord`, then
/// casts to f32.
///
/// Returns [`SpectralError::DegenerateEmbedding`] when every coordinate is zero.
pub fn scale_coords(coords: &Array2<f64>, max_coord: f64) -> Result<Array2<f32>, SpectralError> {
    let max_abs = coords.iter().cloned().map(f64::abs).fold(0.0_f64, f64::max);
    if max_abs <= 0.0 {
        return Err(SpectralError::DegenerateEmbedding);
    }
    let expansion = max_coord / max_abs;
    Ok(coords.mapv(|x| (x * expansion) as f32))
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

/// Scales all coordinates by one factor so the largest absolute value is 10.0, then adds
/// Gaussian noise with standard deviation 0.0001. Downcasts to f32.
///
/// The noise is drawn in logical (row-major) element order from a generator seeded with
/// `seed`. Returns [`SpectralError::DegenerateEmbedding`] when every coordinate is zero,
/// which includes an empty input.
pub fn scale_and_add_noise(coords: Array2<f64>, seed: u64) -> Result<Array2<f32>, SpectralError> {
    let mut scaled = scale_coords(&coords, MAX_COORD)?;
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    for elem in scaled.iter_mut() {
        let z: f32 = StandardNormal.sample(&mut rng);
        // Mean 0 plus scale times a standard-normal draw, as `rand_distr::Normal` computes it.
        *elem += 0.0 + NOISE_SCALE * z;
    }
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Normal;

    #[test]
    fn scale_coords_maps_the_largest_magnitude_to_max_coord() {
        let coords = Array2::from_shape_vec((2, 2), vec![0.5_f64, -2.0, 1.0, 0.25]).unwrap();
        let scaled = scale_coords(&coords, 10.0).unwrap();
        assert_eq!(scaled, ndarray::array![[2.5_f32, -10.0], [5.0, 1.25]]);
    }

    #[test]
    fn scale_coords_rejects_all_zero_input() {
        let coords = Array2::<f64>::zeros((3, 2));
        assert_eq!(
            scale_coords(&coords, 10.0),
            Err(SpectralError::DegenerateEmbedding)
        );
    }

    #[test]
    fn scale_and_add_noise_rejects_empty_input() {
        let coords = Array2::<f64>::zeros((0, 2));
        assert_eq!(
            scale_and_add_noise(coords, 42),
            Err(SpectralError::DegenerateEmbedding)
        );
    }

    #[test]
    fn noise_equals_a_normal_distribution_with_the_documented_scale() {
        // Same seed, same draw order: the noise must be bit-identical to sampling
        // `Normal(0, NOISE_SCALE)` element by element.
        let coords = Array2::from_shape_fn((7, 3), |(i, j)| (i as f64 - 3.0) * (j as f64 + 1.0));
        let seed = 42;
        let noisy = scale_and_add_noise(coords.clone(), seed).unwrap();

        let mut expected = scale_coords(&coords, MAX_COORD).unwrap();
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let normal = Normal::<f32>::new(0.0, NOISE_SCALE).unwrap();
        for elem in expected.iter_mut() {
            *elem += normal.sample(&mut rng);
        }
        assert_eq!(noisy, expected);
    }

    #[test]
    fn normalize_signs_makes_the_largest_magnitude_positive() {
        let mut coords =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 0.5, -3.0, -0.25, 2.0, 0.1]).unwrap();
        normalize_signs(&mut coords);
        // Column 0: largest magnitude is -3.0 → flipped. Column 1: 0.5 is positive → kept.
        assert_eq!(
            coords,
            ndarray::array![[-1.0_f64, 0.5], [3.0, -0.25], [-2.0, 0.1]]
        );
    }
}
