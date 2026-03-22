// spectral-init: Spectral initialization for UMAP embeddings

mod components;
mod config;
mod laplacian;
mod multi_component;
#[doc(hidden)]
pub mod operator;
mod scaling;
mod selection;
#[doc(hidden)]
pub mod solvers;

pub use config::{ComputeMode, SpectralInitConfig};

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
pub fn find_components(
    graph: &sprs::CsMatI<f32, u32, usize>,
) -> (Vec<usize>, usize) {
    components::find_components(graph)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn rsvd_solve(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    // Use the accurate variant: builds a large random subspace via 2I-L power
    // iteration, then projects L directly onto it (B_L = Q^T L Q) and runs
    // dense EVD.  Directly projecting L avoids the λ_L = 2 − λ_M cancellation
    // that degrades accuracy for near-zero eigenvalues.
    let n = laplacian.rows();
    // k_sub: subspace size — large enough that the top-(n_components+1) Ritz
    // values of B_L closely approximate the true smallest L eigenvalues.
    // Scale k_sub with n so that larger graphs get a proportionally larger
    // subspace.  n/4 ensures enough vectors for good Ritz approximations on
    // graphs with small spectral gaps (e.g. λ_1 ≈ 0.01 for n=2000).
    let k_sub = (n / 4).max(n_components * 100 + 60).min(n.saturating_sub(1));
    let (eigs_all, evecs_all) =
        crate::solvers::rsvd::rsvd_solve_accurate(laplacian, n_components, seed, k_sub);
    // eigs_all[0]           = trivial (~0); strip it.
    // eigs_all[1..=n_components] = n_components non-trivial eigenvalues (ascending).
    let eigs = eigs_all.slice(ndarray::s![1..]).to_owned();
    let evecs = evecs_all.slice(ndarray::s![.., 1..]).to_owned();
    (eigs, evecs)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
/// Exposes `solve_eigenproblem` for integration tests that need to assert which
/// solver level was reached.
///
/// **Test-only seam — not part of the stable public API.**  The returned `u8`
/// encodes the solver level (0 = dense EVD, 1 = LOBPCG, 2 = LOBPCG+reg,
/// 3 = rSVD, 4 = forced dense EVD) and may change if the escalation chain is
/// reordered or extended.  Test code that asserts on this value is intentionally
/// coupled to the current chain ordering.
pub fn solve_eigenproblem_pub(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> ((ndarray::Array1<f64>, ndarray::Array2<f64>), u8) {
    let n = laplacian.rows();
    let sqrt_deg = ndarray::Array1::ones(n);
    crate::solvers::solve_eigenproblem(laplacian, n_components, seed, &sqrt_deg, ComputeMode::PythonCompat)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::solvers::lobpcg::lobpcg_solve;

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
/// - `config`: Pipeline configuration. Use `SpectralInitConfig::default()` for Python-compatible
///   behavior.
///
/// # Returns
/// An `(n_samples, n_components)` array of f32 initial coordinates.
pub fn spectral_init(
    graph: &CsMatI<f32, u32, usize>,
    n_components: usize,
    seed: u64,
    data: Option<ArrayView2<'_, f32>>,
    config: SpectralInitConfig,
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
            config.compute_mode,
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
    let ((eigenvalues, eigenvectors), _) = solvers::solve_eigenproblem(&lap, n_components, seed, &sqrt_deg, config.compute_mode);

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
        let result = spectral_init(&g, 0, 42, None, SpectralInitConfig::default());
        assert!(matches!(result, Err(SpectralError::InvalidGraph(_))));
    }

    #[test]
    fn spectral_init_empty_graph_returns_error() {
        let g = CsMatI::<f32, u32, usize>::new((0, 0), vec![0usize], vec![], vec![]);
        let result = spectral_init(&g, 2, 42, None, SpectralInitConfig::default());
        assert!(matches!(result, Err(SpectralError::InvalidGraph(_))));
    }

    #[test]
    fn spectral_init_too_few_nodes_returns_error() {
        // 2-node connected graph, asking for 3 dimensions
        let g = CsMatI::<f32, u32, usize>::new((2, 2), vec![0, 1, 2], vec![1u32, 0u32], vec![1.0f32; 2]);
        let result = spectral_init(&g, 3, 42, None, SpectralInitConfig::default());
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
        let result = spectral_init(&g, 2, 42, None, SpectralInitConfig::default());
        let arr = result.expect("spectral_init on connected graph should succeed");
        assert_eq!(arr.shape(), &[4, 2]);
        for &v in arr.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    // REQ-API-002, REQ-API-004
    #[test]
    fn default_config_is_python_compat() {
        let cfg = SpectralInitConfig::default();
        assert_eq!(cfg.compute_mode, ComputeMode::PythonCompat);
    }

    // REQ-TRAIT-001: all required derives
    #[test]
    fn compute_mode_copy_clone_eq() {
        let a = ComputeMode::PythonCompat;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b); // PartialEq, Eq
        assert_eq!(a, c);
        let _ = format!("{a:?}"); // Debug
    }

    // REQ-TRAIT-001: RustNative variant
    #[test]
    fn compute_mode_rust_native_neq_python_compat() {
        assert_ne!(ComputeMode::RustNative, ComputeMode::PythonCompat);
    }

    // REQ-API-001: both variants exist and are distinguishable
    #[test]
    fn compute_mode_variants_exhaustive() {
        fn accept_mode(_: ComputeMode) {}
        accept_mode(ComputeMode::PythonCompat);
        accept_mode(ComputeMode::RustNative);
    }

    // REQ-PLUMB-002: no behavioral change — both modes produce identical output
    #[test]
    fn python_compat_and_rust_native_same_output() {
        let g = CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 3, 5, 6],
            vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
            vec![1.0f32; 6],
        );
        let r1 = spectral_init(&g, 2, 42, None, SpectralInitConfig::default())
            .expect("PythonCompat should succeed");
        let r2 = spectral_init(
            &g,
            2,
            42,
            None,
            SpectralInitConfig { compute_mode: ComputeMode::RustNative },
        )
        .expect("RustNative should succeed");
        // Identical outputs because no branching exists yet
        assert_eq!(r1, r2);
    }

    // REQ-API-003: config compiles with zero boilerplate using Default
    #[test]
    fn spectral_init_with_default_config_compiles_and_runs() {
        let g = CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 3, 5, 6],
            vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
            vec![1.0f32; 6],
        );
        let result = spectral_init(&g, 2, 42, None, SpectralInitConfig::default());
        assert!(result.is_ok());
    }
}
