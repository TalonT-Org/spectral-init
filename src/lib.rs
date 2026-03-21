// spectral-init: Spectral initialization for UMAP embeddings

mod components;
mod laplacian;
mod multi_component;
#[doc(hidden)]
pub mod operator;
mod scaling;
mod selection;
#[doc(hidden)]
pub mod solvers;

// Re-exported for component-level integration tests. These are internal pipeline
// functions, not part of the stable public API.
pub use crate::selection::select_eigenvectors;
pub use crate::laplacian::compute_degrees;
pub use crate::laplacian::build_normalized_laplacian;
/// Dense EVD solver. Exposed for component-level integration tests; not part of the
/// stable public API and may change without notice.
pub use crate::solvers::dense_evd;

use ndarray::Array2;
use sprs::CsMatI;

/// Errors returned by the spectral initialization pipeline.
#[derive(Debug, thiserror::Error)]
pub enum SpectralError {
    /// The eigensolver escalation chain exhausted all levels without converging.
    /// This should never occur in practice; if seen, it indicates a degenerate graph.
    #[error("eigensolver failed to converge after full escalation chain")]
    ConvergenceFailure,

    /// The graph adjacency matrix is malformed (non-square, negative weights, NaN/Inf).
    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    /// The graph has fewer nodes than the requested embedding dimensionality.
    #[error("graph has too few nodes ({n}) for {dims}-dimensional embedding")]
    TooFewNodes { n: usize, dims: usize },
}

/// Compute spectral initialization coordinates for a UMAP fuzzy k-NN graph.
///
/// # Arguments
/// - `graph`: Symmetric sparse adjacency matrix (CSR format, f32 weights, u32 column indices).
/// - `n_components`: Number of embedding dimensions.
/// - `seed`: Random seed for reproducible noise and solver initialization.
///
/// # Returns
/// An `(n_samples, n_components)` array of f32 initial coordinates.
pub fn spectral_init(
    graph: &CsMatI<f32, u32, usize>,
    n_components: usize,
    seed: u64,
) -> Result<Array2<f32>, SpectralError> {
    todo!("spectral_init: orchestrate components → laplacian → solve → scale")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_error_variants_exist() {
        let _a = SpectralError::ConvergenceFailure;
        let _b = SpectralError::InvalidGraph("test".into());
        let _c = SpectralError::TooFewNodes { n: 1, dims: 2 };
    }

    #[test]
    fn spectral_error_is_std_error() {
        let e: &dyn std::error::Error = &SpectralError::ConvergenceFailure;
        assert!(!e.to_string().is_empty());
    }


}
