use ndarray::Array2;
use sprs::CsMatI;
use crate::SpectralError;

/// Computes spectral embedding for disconnected graphs.
/// Each component is embedded independently; component embeddings are aligned
/// into a common coordinate frame via centroid meta-embedding.
pub(crate) fn embed_disconnected(
    graph: &CsMatI<f32, u32, usize>,
    component_labels: &[usize],
    n_conn_components: usize,
    n_embedding_dims: usize,
    seed: u64,
) -> Result<Array2<f64>, SpectralError> {
    todo!("embed_disconnected: per-component spectral + centroid alignment")
}
