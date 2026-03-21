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

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn rsvd_solve_pub(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    crate::solvers::rsvd::rsvd_solve(laplacian, n_components, seed)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn solve_eigenproblem_pub(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> ((ndarray::Array1<f64>, ndarray::Array2<f64>), u8) {
    let n = laplacian.rows();
    let sqrt_deg = ndarray::Array1::ones(n);
    crate::solvers::solve_eigenproblem(laplacian, n_components, seed, &sqrt_deg)
}

use ndarray::{Array2, ArrayView2};
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
/// - `data`: Optional raw input data `(n_samples, n_features)` in f32. Required only when the
///   graph is disconnected and the number of connected components exceeds `2 * n_components`
///   (needed for the spectral meta-embedding of component centroids).
///
/// # Returns
/// An `(n_samples, n_components)` array of f32 initial coordinates.
pub fn spectral_init(
    graph: &CsMatI<f32, u32, usize>,
    n_components: usize,
    seed: u64,
    data: Option<ArrayView2<'_, f32>>,
) -> Result<Array2<f32>, SpectralError> {
    // ── Input validation ──────────────────────────────────────────────────
    let n = graph.rows();
    if n == 0 || n_components == 0 {
        return Err(SpectralError::InvalidGraph(format!(
            "spectral_init requires n > 0 and n_components > 0; got n={n}, n_components={n_components}"
        )));
    }
    if n <= n_components {
        return Err(SpectralError::TooFewNodes { n, dims: n_components });
    }

    // ── Component A: connectivity check ───────────────────────────────────
    let (labels, n_conn_components) = components::find_components(graph);
    if n_conn_components > 1 {
        let coords = crate::multi_component::embed_disconnected(
            graph,
            &labels,
            n_conn_components,
            n_components,
            seed,
            data,
        )?;
        return scaling::scale_and_add_noise(coords, seed);
    }

    // ── Component B: degrees and inverse-square-root degrees ──────────────
    let (_degrees, sqrt_deg) = laplacian::compute_degrees(graph);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();

    // ── Component C: normalized Laplacian ─────────────────────────────────
    let lap = laplacian::build_normalized_laplacian(graph, &inv_sqrt_deg);

    // ── Component D: eigensolver escalation chain ─────────────────────────
    let ((eigenvalues, eigenvectors), _) = solvers::solve_eigenproblem(&lap, n_components, seed, &sqrt_deg);

    // ── Component E: eigenvector selection ────────────────────────────────
    let selected = selection::select_eigenvectors(
        eigenvalues.as_slice_memory_order().expect("eigenvalues must be contiguous"),
        &eigenvectors,
        n_components,
    );

    // ── Component F: scale and add noise ──────────────────────────────────
    scaling::scale_and_add_noise(selected, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_error_is_std_error() {
        let e: &dyn std::error::Error = &SpectralError::ConvergenceFailure;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn spectral_init_zero_components_returns_error() {
        // 3-node path graph (0-1-2): valid CSR construction
        let g = CsMatI::<f32, u32, usize>::new(
            (3, 3),
            vec![0usize, 1, 3, 4],
            vec![1u32, 0u32, 2u32, 1u32],
            vec![1.0f32; 4],
        );
        let result = spectral_init(&g, 0, 42, None);
        assert!(matches!(result, Err(SpectralError::InvalidGraph(_))));
    }

    #[test]
    fn spectral_init_empty_graph_returns_error() {
        let g = CsMatI::<f32, u32, usize>::new((0, 0), vec![0usize], vec![], vec![]);
        let result = spectral_init(&g, 2, 42, None);
        assert!(matches!(result, Err(SpectralError::InvalidGraph(_))));
    }

    #[test]
    fn spectral_init_too_few_nodes_returns_error() {
        // 2-node connected graph, asking for 3 dimensions
        let g = CsMatI::<f32, u32, usize>::new((2, 2), vec![0, 1, 2], vec![1u32, 0u32], vec![1.0f32; 2]);
        let result = spectral_init(&g, 3, 42, None);
        assert!(matches!(result, Err(SpectralError::TooFewNodes { n: 2, dims: 3 })));
    }

    #[test]
    fn spectral_init_connected_graph_succeeds_and_has_correct_shape() {
        // 4-node path graph: 0-1-2-3
        let g = CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 3, 5, 6],
            vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
            vec![1.0f32; 6],
        );
        let result = spectral_init(&g, 2, 42, None);
        let arr = result.expect("spectral_init on connected graph should succeed");
        assert_eq!(arr.shape(), &[4, 2]);
        for &v in arr.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }
}
