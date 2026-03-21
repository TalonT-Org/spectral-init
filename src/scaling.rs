use ndarray::Array2;

/// Scales columns so max absolute value is 10.0, then adds Gaussian noise
/// with scale=0.0001. Downcasts to f32.
pub(crate) fn scale_and_add_noise(
    coords: Array2<f64>,
    seed: u64,
) -> Array2<f32> {
    todo!("scale_and_add_noise: max_abs=10 scaling + noise, downcast to f32")
}
