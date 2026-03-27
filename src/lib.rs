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
    crate::solvers::rsvd::rsvd_solve(laplacian, n_components, seed)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn rsvd_solve_accurate(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
    k_sub: usize,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    crate::solvers::rsvd::rsvd_solve_accurate(laplacian, n_components, seed, k_sub)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
/// Exposes `solve_eigenproblem` for integration tests that need to assert which
/// solver level was reached.
///
/// **Test-only seam — not part of the stable public API.**  The returned `u8`
/// encodes the solver level (0 = dense EVD, 1 = LOBPCG, 2 = sinv LOBPCG,
/// 3 = LOBPCG+reg, 4 = rSVD, 5 = forced dense EVD) and may change if the
/// escalation chain is reordered or extended.  Test code that asserts on this
/// value is intentionally coupled to the current chain ordering.
pub fn solve_eigenproblem_pub(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> ((ndarray::Array1<f64>, ndarray::Array2<f64>), u8) {
    let n = laplacian.rows();
    let sqrt_deg = ndarray::Array1::ones(n);
    crate::solvers::solve_eigenproblem(laplacian, n_components, seed, &sqrt_deg)
}

#[cfg(all(feature = "testing", any(target_arch = "x86", target_arch = "x86_64")))]
#[doc(hidden)]
/// SIMD variant of `solve_eigenproblem_pub`: routes LOBPCG levels through
/// `CsrOperatorSimd` (AVX2 gather kernel) instead of `CsrOperator`.
///
/// **Test-only seam — not part of the stable public API.**  Solver-level encoding
/// is identical to `solve_eigenproblem_pub`.
pub fn solve_eigenproblem_simd_pub(
    laplacian: &sprs::CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> ((ndarray::Array1<f64>, ndarray::Array2<f64>), u8) {
    let n = laplacian.rows();
    let sqrt_deg = ndarray::Array1::ones(n);
    crate::solvers::solve_eigenproblem_simd(laplacian, n_components, seed, &sqrt_deg)
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::solvers::lobpcg::lobpcg_solve;

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn embed_disconnected(
    graph: &sprs::CsMatI<f32, u32, usize>,
    component_labels: &[usize],
    n_conn_components: usize,
    n_embedding_dims: usize,
    seed: u64,
    data: Option<ndarray::ArrayView2<'_, f32>>,
    compute_mode: ComputeMode,
) -> Result<ndarray::Array2<f64>, SpectralError> {
    crate::multi_component::embed_disconnected(
        graph,
        component_labels,
        n_conn_components,
        n_embedding_dims,
        seed,
        data,
        compute_mode,
    )
}

#[cfg(feature = "testing")]
#[doc(hidden)]
pub use crate::solvers::sinv::lobpcg_sinv_solve;

#[cfg(feature = "testing")]
#[doc(hidden)]
pub fn scale_and_add_noise_pub(
    coords: ndarray::Array2<f64>,
    seed: u64,
) -> Result<ndarray::Array2<f32>, SpectralError> {
    crate::scaling::scale_and_add_noise(coords, seed)
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
    let (_degrees, sqrt_deg) = laplacian::compute_degrees(graph, config.compute_mode);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();

    // ── Component C: normalized Laplacian ─────────────────────────────────
    let lap = laplacian::build_normalized_laplacian(graph, &inv_sqrt_deg);

    // ── Component D: eigensolver escalation chain ─────────────────────────
    let ((eigenvalues, eigenvectors), _) = solvers::solve_eigenproblem(&lap, n_components, seed, &sqrt_deg);

    // ── Component E: eigenvector selection ────────────────────────────────
    let mut selected = selection::select_eigenvectors(
        eigenvalues.as_slice_memory_order().expect("eigenvalues must be contiguous"),
        &eigenvectors,
        n_components,
    );

    // ── Component E.5: sign normalization ─────────────────────────────────
    // Enforce argmax sign convention: element with largest absolute value must be positive.
    // Ensures consistent signs across runs; verified by the sign-convention integration test.
    scaling::normalize_signs(&mut selected);

    // ── Component F: scale and add noise ──────────────────────────────────
    scaling::scale_and_add_noise(selected, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "testing", any(target_arch = "x86", target_arch = "x86_64"), test))]
    #[test]
    fn solve_eigenproblem_simd_pub_returns_correct_shape() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        // 6-node path Laplacian (n=6 < dense threshold → Level 0; SIMD not exercised
        // but symbol reachability and return type are confirmed)
        let lap = sprs::CsMatI::<f64, usize>::new(
            (6, 6),
            vec![0usize, 2, 5, 8, 11, 14, 16],
            vec![0usize,1, 0,1,2, 1,2,3, 2,3,4, 3,4,5, 4,5],
            vec![1.,-1., -1.,2.,-1., -1.,2.,-1., -1.,2.,-1., -1.,2.,-1., -1.,1.],
        );
        let ((eigs, vecs), _level) = solve_eigenproblem_simd_pub(&lap, 2, 42);
        assert_eq!(eigs.len(), 3);
        assert_eq!(vecs.shape(), &[6, 3]);
    }

    #[cfg(all(feature = "testing", test))]
    #[test]
    fn scale_and_add_noise_pub_is_accessible() {
        let coords = ndarray::Array2::<f64>::from_shape_vec(
            (2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let result = scale_and_add_noise_pub(coords, 42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape(), &[2, 2]);
    }

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

    #[test]
    fn spectral_init_output_respects_sign_convention() {
        // 4-node path graph: 0-1-2-3
        let g = CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 3, 5, 6],
            vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
            vec![1.0f32; 6],
        );
        let result = spectral_init(&g, 2, 42, None, SpectralInitConfig::default())
            .expect("spectral_init should succeed");
        // For each column, the element with the largest absolute value must be positive.
        // normalize_signs (argmax convention) runs before scale_and_add_noise.
        // Noise scale is 1e-4; scaled coords are ~10 — noise cannot flip the argmax sign.
        // If scale_and_add_noise were to negate coordinates, the argmax element would become negative.
        for col in 0..result.ncols() {
            let col_view = result.column(col);
            let argmax_val = col_view.iter().copied()
                .reduce(|a, b| if b.abs() > a.abs() { b } else { a });
            let v = argmax_val.unwrap_or_else(|| {
                panic!("column {col}: argmax returned None — coordinate column is degenerate")
            });
            assert!(
                v > 0.0,
                "column {col}: sign convention violated — argmax element is \
                 {v:.6}, expected positive"
            );
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn default_config_is_python_compat() {
        let cfg = SpectralInitConfig::default();
        assert_eq!(cfg.compute_mode, ComputeMode::PythonCompat);
    }

    #[test]
    fn compute_mode_copy_clone_eq() {
        let a = ComputeMode::PythonCompat;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b); // PartialEq, Eq
        assert_eq!(a, c);
        let _ = format!("{a:?}"); // Debug
    }

    #[test]
    fn compute_mode_rust_native_neq_python_compat() {
        assert_ne!(ComputeMode::RustNative, ComputeMode::PythonCompat);
    }

    #[test]
    fn compute_mode_variants_exhaustive() {
        // Exhaustive match — compiler enforces this if a new variant is added.
        let modes = [ComputeMode::PythonCompat, ComputeMode::RustNative];
        for mode in &modes {
            let label = match mode {
                ComputeMode::PythonCompat => "python_compat",
                ComputeMode::RustNative => "rust_native",
            };
            assert!(!label.is_empty(), "variant label must be non-empty");
        }
    }

    #[test]
    fn python_compat_and_rust_native_same_subspace() {
        // Sign-agnostic subspace equivalence check: each embedding column may be
        // sign-flipped independently, so compare |cos θ| ≈ 1 per column pair.
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
        assert_eq!(r1.shape(), r2.shape());
        let n_cols = r1.shape()[1];
        for col in 0..n_cols {
            let c1 = r1.column(col);
            let c2 = r2.column(col);
            let dot: f32 = c1.iter().zip(c2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f32 = c1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm2: f32 = c2.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos_sim = dot.abs() / (norm1 * norm2).max(f32::EPSILON);
            assert!(
                cos_sim > 0.999,
                "column {col}: |cos θ| = {cos_sim:.6}, expected ≈ 1.0 (same subspace up to sign)"
            );
        }
    }

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
