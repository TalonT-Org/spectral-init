use ndarray::Array1;
use sprs::CsMatI;

/// Builds the symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} in f64.
/// Accepts the fuzzy k-NN graph in f32/u32 format from umap-rs and upcasts.
pub(crate) fn build_normalized_laplacian(
    graph: &CsMatI<f32, u32, usize>,
) -> (CsMatI<f64, usize>, Array1<f64>) {
    todo!("build_normalized_laplacian: construct L and return degree vector D")
}
