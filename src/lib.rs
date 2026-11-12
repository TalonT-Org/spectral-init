//! Spectral initialization for UMAP embeddings.
//!
//! Computes the Laplacian eigenvectors of a fuzzy k-NN graph to provide
//! globally-aware starting coordinates for the SGD optimization phase of UMAP.
//! This is the primary reason Python UMAP produces superior embeddings compared to
//! random or PCA initialization: eigenvectors of the graph Laplacian encode the
//! global manifold structure before a single gradient step is taken.
//!
//! # Quick Start
//!
//! ```
//! use spectral_init::sprs::CsMatI;
//! use spectral_init::{SpectralInitConfig, spectral_init};
//!
//! // 4-node path graph: 0 – 1 – 2 – 3 (symmetric, f32 weights)
//! let graph = CsMatI::<f32, u32, usize>::new(
//!     (4, 4),
//!     vec![0usize, 1, 3, 5, 6],
//!     vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
//!     vec![1.0f32; 6],
//! );
//! let config = SpectralInitConfig::new(2).with_seed(42);
//! let coords = spectral_init(graph.view(), None, &config).expect("spectral_init failed");
//! assert_eq!(coords.shape(), &[4, 2]);
//! for &v in coords.iter() {
//!     assert!(v.is_finite());
//! }
//! ```
//!
//! # Algorithm
//!
//! Builds the symmetric normalized Laplacian
//!
//! $$L = I - D^{-1/2} W D^{-1/2}$$
//!
//! where $W$ is the fuzzy k-NN adjacency matrix and $D$ is the degree matrix.
//! The $k$ smallest non-trivial eigenvectors of $L$ (skipping the all-ones trivial
//! eigenvector at $\lambda = 0$) form the initial coordinate matrix. Coordinates are
//! then scaled to $[-10, 10]$ and perturbed with $\mathcal{N}(0, 10^{-4})$ noise to
//! break symmetry.
//!
//! The eigensolver uses a 6-level escalation chain (dense EVD → LOBPCG →
//! shift-invert LOBPCG → LOBPCG+regularization → randomized SVD → forced dense EVD)
//! to guarantee convergence on any valid graph.
//!
//! # Compute Mode
//!
//! [`ComputeMode::PythonCompat`] (default) matches Python UMAP's degree accumulation
//! order and solver paths exactly, enabling bit-for-bit validation against reference
//! outputs. [`ComputeMode::RustNative`] routes large-graph LOBPCG levels through an
//! AVX2+FMA SIMD matrix-vector kernel (~1 ULP divergence from Python), offering
//! improved throughput on x86_64 hardware with AVX2 support.
//!
//! # Feature Flags
//!
//! The crate has no public Cargo features.
//!
//! # Integration with umap-rs
//!
//! Pass the graph returned by `umap_rs::Umap::graph()` directly — its type
//! `CsMatI<f32, u32, usize>` matches this crate's input type exactly.
//!
//! # Error Handling
//!
//! Every failure is a [`SpectralError`] variant; the crate never substitutes a random
//! layout. Inputs are validated before any computation: the graph must be square,
//! symmetric and have finite, non-negative weights
//! ([`SpectralError::NotSquare`], [`SpectralError::NotSymmetric`],
//! [`SpectralError::NonFiniteWeight`], [`SpectralError::NegativeWeight`]), with more
//! nodes than embedding dimensions ([`SpectralError::TooFewNodes`]).
//! [`SpectralError::DenseBudgetExceeded`] and [`SpectralError::ConvergenceFailure`]
//! report that no solver could produce acceptable eigenvectors within the configured
//! dense memory budget. See [`spectral_init`] for the full list and a fallback example.
//!
//! # Public Dependencies
//!
//! The graph and array types in the signature of [`spectral_init`] come from the
//! [`sprs`] and [`ndarray`] crates, which are re-exported so that callers can name the
//! exact versions this crate was built against.
//!
//! # User Guide
//!
//! For step-by-step integration instructions with `umap-rs` and standalone usage,
//! see the [user guide](https://github.com/TalonT-Org/spectral-init/blob/main/docs/user-guide.md).
#![warn(missing_docs)]

mod components;
mod config;
mod error;
mod laplacian;
mod multi_component;
mod operator;
mod scaling;
mod selection;
mod solvers;
mod validate;

pub use config::{ComputeMode, SpectralInitConfig};
pub use error::SpectralError;
pub use {ndarray, sprs};

/// Internal pipeline pieces for this crate's own integration tests and research harness.
///
/// Exempt from semver: anything in this module may change or disappear in any release.
#[cfg(feature = "__internal")]
#[doc(hidden)]
pub mod __internal {
    #[doc(hidden)]
    pub use crate::components::find_components;
    #[doc(hidden)]
    pub use crate::laplacian::{build_normalized_laplacian, compute_degrees};
    #[doc(hidden)]
    pub use crate::multi_component::embed_disconnected;
    #[doc(hidden)]
    pub use crate::operator::{CsrOperator, LinearOperator, spmv_csr};
    #[cfg(target_arch = "x86_64")]
    #[doc(hidden)]
    pub use crate::operator::{CsrOperatorSimd, spmv_avx2_gather};
    #[doc(hidden)]
    pub use crate::scaling::{normalize_signs, scale_and_add_noise, scale_coords};
    #[doc(hidden)]
    pub use crate::selection::select_eigenvectors;
    #[doc(hidden)]
    pub use crate::solvers::dense_evd;
    #[doc(hidden)]
    pub use crate::solvers::lobpcg::lobpcg_solve;
    #[doc(hidden)]
    pub use crate::solvers::quality::*;
    #[doc(hidden)]
    pub use crate::solvers::rsvd::{rsvd_solve, rsvd_solve_accurate};
    #[doc(hidden)]
    pub use crate::solvers::sinv::lobpcg_sinv_solve;
    #[doc(hidden)]
    pub use crate::solvers::{
        EigenResult, QualityGates, SolveReport, SolverLevel, SolverParams, solve_eigenproblem,
    };
}

use ndarray::{Array2, ArrayView2};
use sprs::CsMatViewI;

/// Computes spectral initialization coordinates for a symmetric, non-negative graph.
///
/// Builds the symmetric normalized Laplacian of `graph`, computes its `n_components`
/// smallest non-trivial eigenvectors via an automatic solver escalation chain, scales
/// the coordinates so that the largest magnitude is 10, and adds Gaussian noise with
/// standard deviation 10⁻⁴.
///
/// # Inputs
///
/// - `graph` — the weighted adjacency matrix: square, symmetric, with finite
///   non-negative weights, in CSR or CSC storage. Stored zeros are absent edges.
///   Diagonal entries (self-loops) count towards both degree and adjacency.
/// - `data` — the points the graph was built from, one row per node. It is consulted
///   only when the graph has more than `2 · n_components` connected components, where
///   it places the components relative to one another; otherwise it may be `None`.
///   When supplied it is always validated, even if it ends up unused.
/// - `config` — embedding dimensionality, seed, [`ComputeMode`] and dense memory
///   budget; see [`SpectralInitConfig`].
///
/// # Returns
///
/// An `(n, n_components)` array in standard layout (C-contiguous, row-major), so
/// `as_slice()` succeeds on it. For a fixed `Cargo.lock`, target, CPU feature set and
/// thread configuration the result is deterministic in `config`.
///
/// # Errors
///
/// Inputs are validated first, in this order of precedence:
///
/// - [`SpectralError::InvalidParameter`] — `n_components` is 0.
/// - [`SpectralError::NotSquare`] — `graph` is not square.
/// - [`SpectralError::TooFewNodes`] — the graph does not have more nodes than
///   `n_components` (this covers the empty graph).
/// - [`SpectralError::NonFiniteWeight`] — a weight is NaN or infinite.
/// - [`SpectralError::NegativeWeight`] — a weight is negative.
/// - [`SpectralError::NotSymmetric`] — an entry has no matching mirror entry (relative
///   differences up to 10⁻⁶ are tolerated).
/// - [`SpectralError::DegreeOverflow`] — a node's summed weights overflow `f32`.
/// - [`SpectralError::DataShapeMismatch`] — `data` does not have one row per node.
/// - [`SpectralError::NonFiniteData`] — `data` contains NaN or an infinity.
///
/// The computation itself can fail with:
///
/// - [`SpectralError::DataRequired`] — the graph has more than `2 · n_components`
///   connected components and `data` is `None`.
/// - [`SpectralError::DenseBudgetExceeded`] — the iterative solvers did not converge and
///   the dense fallback needs a matrix larger than the dense memory budget, or the graph
///   has so many connected components that their affinity matrix exceeds it.
/// - [`SpectralError::ConvergenceFailure`] — no solver produced eigenvectors of
///   acceptable quality.
/// - [`SpectralError::DegenerateEmbedding`] — the computed coordinates are all zero.
///
/// The function does not panic on any input. A dense matrix within the budget is
/// allocated like any other Rust allocation: if the system cannot provide the memory,
/// the process aborts.
///
/// # Examples
///
/// ```
/// use spectral_init::sprs::CsMatI;
/// use spectral_init::{SpectralInitConfig, spectral_init};
///
/// // 4-node path graph: 0 – 1 – 2 – 3
/// let graph = CsMatI::<f32, u32, usize>::new(
///     (4, 4),
///     vec![0usize, 1, 3, 5, 6],
///     vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
///     vec![1.0f32; 6],
/// );
/// let coords = spectral_init(graph.view(), None, &SpectralInitConfig::new(2))?;
/// assert_eq!(coords.shape(), &[4, 2]);
/// assert!(coords.is_standard_layout());
/// # Ok::<(), spectral_init::SpectralError>(())
/// ```
///
/// umap-learn falls back to a random layout when its spectral layout fails. This crate
/// reports the failure instead, so the caller decides what to substitute:
///
/// ```
/// use spectral_init::ndarray::Array2;
/// use spectral_init::sprs::CsMatI;
/// use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};
///
/// /// The caller's own fallback layout (here: a fixed grid).
/// fn fallback_init(n: usize, n_components: usize) -> Array2<f32> {
///     Array2::from_shape_fn((n, n_components), |(i, j)| (i * (j + 1)) as f32)
/// }
///
/// fn initial_coordinates(
///     graph: &CsMatI<f32, u32, usize>,
///     config: &SpectralInitConfig,
/// ) -> Result<Array2<f32>, SpectralError> {
///     match spectral_init(graph.view(), None, config) {
///         Ok(coords) => Ok(coords),
///         // The solvers gave up: substitute the fallback.
///         Err(
///             SpectralError::ConvergenceFailure { .. }
///             | SpectralError::DenseBudgetExceeded { .. }
///             | SpectralError::DegenerateEmbedding,
///         ) => Ok(fallback_init(graph.rows(), config.n_components())),
///         // Anything else is a problem with the inputs: report it.
///         Err(other) => Err(other),
///     }
/// }
///
/// let graph = CsMatI::<f32, u32, usize>::new(
///     (4, 4),
///     vec![0usize, 1, 3, 5, 6],
///     vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
///     vec![1.0f32; 6],
/// );
/// let coords = initial_coordinates(&graph, &SpectralInitConfig::new(2))?;
/// assert_eq!(coords.shape(), &[4, 2]);
/// # Ok::<(), SpectralError>(())
/// ```
pub fn spectral_init(
    graph: CsMatViewI<'_, f32, u32, usize>,
    data: Option<ArrayView2<'_, f32>>,
    config: &SpectralInitConfig,
) -> Result<Array2<f32>, SpectralError> {
    let graph = validate::validate(graph, data, config)?;
    let n = graph.rows();
    let n_components = config.n_components();
    let seed = config.seed();
    let compute_mode = config.compute_mode();
    let solver_params = solvers::SolverParams::from_config(config);

    // ── Component A: connectivity check ───────────────────────────────────
    let (labels, n_conn_components) = components::find_components(graph);
    let coords = if n_conn_components > 1 {
        multi_component::embed_disconnected(
            graph,
            &labels,
            n_conn_components,
            n_components,
            seed,
            data,
            compute_mode,
            &solver_params,
        )?
    } else {
        // ── Component B: degrees and inverse-square-root degrees ──────────
        let (_degrees, sqrt_deg) = laplacian::compute_degrees(graph, compute_mode);
        let inv_sqrt_deg: Vec<f64> = sqrt_deg
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();

        // ── Component C: normalized Laplacian ─────────────────────────────
        let lap = laplacian::build_normalized_laplacian(graph, &inv_sqrt_deg);

        // ── Component D: eigensolver escalation chain ─────────────────────
        let ((eigenvalues, eigenvectors), report) = solvers::solve_eigenproblem(
            &lap,
            n_components,
            seed,
            &sqrt_deg,
            compute_mode,
            &solver_params,
        )?;
        report.log("connected graph", n);

        // ── Component E: eigenvector selection ────────────────────────────
        // The solver returns `n_components + 1` eigenpairs, as `select_eigenvectors` needs.
        let mut selected =
            selection::select_eigenvectors(&eigenvalues.to_vec(), &eigenvectors, n_components);

        // ── Component E.5: sign normalization ─────────────────────────────
        // Enforce argmax sign convention: element with largest absolute value must be positive.
        // Ensures consistent signs across runs; verified by the sign-convention integration test.
        scaling::normalize_signs(&mut selected);
        selected
    };

    // ── Component F: scale and add noise ──────────────────────────────────
    let embedding = scaling::scale_and_add_noise(coords, seed)?;
    Ok(into_standard_layout(embedding))
}

/// Returns `array` in standard (C-contiguous, row-major) layout, copying only when it is
/// not already laid out that way.
fn into_standard_layout(array: Array2<f32>) -> Array2<f32> {
    if array.is_standard_layout() {
        array
    } else {
        array.as_standard_layout().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    /// 4-node path graph: 0-1-2-3
    fn path_4() -> CsMatI<f32, u32, usize> {
        CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 3, 5, 6],
            vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
            vec![1.0f32; 6],
        )
    }

    fn config() -> SpectralInitConfig {
        SpectralInitConfig::new(2).with_seed(42)
    }

    #[test]
    fn spectral_error_is_std_error() {
        let e: &dyn std::error::Error = &SpectralError::ConvergenceFailure { n: 3 };
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn spectral_init_zero_components_returns_error() {
        let result = spectral_init(path_4().view(), None, &SpectralInitConfig::new(0));
        assert!(matches!(
            result,
            Err(SpectralError::InvalidParameter {
                name: "n_components",
                ..
            })
        ));
    }

    #[test]
    fn spectral_init_empty_graph_returns_error() {
        let g = CsMatI::<f32, u32, usize>::new((0, 0), vec![0usize], vec![], vec![]);
        let result = spectral_init(g.view(), None, &config());
        assert_eq!(
            result,
            Err(SpectralError::TooFewNodes {
                n: 0,
                n_components: 2
            })
        );
    }

    #[test]
    fn spectral_init_too_few_nodes_returns_error() {
        // 2-node connected graph, asking for 3 dimensions
        let g = CsMatI::<f32, u32, usize>::new(
            (2, 2),
            vec![0, 1, 2],
            vec![1u32, 0u32],
            vec![1.0f32; 2],
        );
        let result = spectral_init(g.view(), None, &SpectralInitConfig::new(3));
        assert_eq!(
            result,
            Err(SpectralError::TooFewNodes {
                n: 2,
                n_components: 3
            })
        );
    }

    #[test]
    fn spectral_init_connected_graph_succeeds_and_has_correct_shape() {
        let result = spectral_init(path_4().view(), None, &config());
        let arr = result.expect("spectral_init on connected graph should succeed");
        assert_eq!(arr.shape(), &[4, 2]);
        for &v in arr.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn spectral_init_output_respects_sign_convention() {
        let result =
            spectral_init(path_4().view(), None, &config()).expect("spectral_init should succeed");
        // For each column, the element with the largest absolute value must be positive.
        // normalize_signs (argmax convention) runs before scale_and_add_noise.
        // Noise scale is 1e-4; scaled coords are ~10 — noise cannot flip the argmax sign.
        // If scale_and_add_noise were to negate coordinates, the argmax element would become negative.
        for col in 0..result.ncols() {
            let col_view = result.column(col);
            let argmax_val = col_view
                .iter()
                .copied()
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

    #[test]
    fn into_standard_layout_preserves_values() {
        let fortran = ndarray::Array2::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 2)),
            vec![1.0_f32, 3.0, 5.0, 2.0, 4.0, 6.0],
        )
        .unwrap();
        assert!(!fortran.is_standard_layout());
        assert_eq!(
            fortran,
            ndarray::array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]
        );

        let standard = into_standard_layout(fortran);
        assert!(standard.is_standard_layout());
        assert_eq!(
            standard.as_slice(),
            Some(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..])
        );
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn new_config_has_documented_defaults() {
        let cfg = SpectralInitConfig::new(3);
        assert_eq!(cfg.n_components(), 3);
        assert_eq!(cfg.seed(), 0);
        assert_eq!(cfg.compute_mode(), ComputeMode::PythonCompat);
        assert_eq!(
            cfg.dense_memory_budget(),
            SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET
        );
        assert_eq!(SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET, 1 << 32);
    }

    #[test]
    fn config_setters_change_only_their_field() {
        let cfg = SpectralInitConfig::new(2)
            .with_seed(7)
            .with_compute_mode(ComputeMode::RustNative)
            .with_dense_memory_budget(1024);
        assert_eq!(cfg.n_components(), 2);
        assert_eq!(cfg.seed(), 7);
        assert_eq!(cfg.compute_mode(), ComputeMode::RustNative);
        assert_eq!(cfg.dense_memory_budget(), 1024);
        assert_eq!(cfg.clone(), cfg);
        assert_ne!(cfg, SpectralInitConfig::new(2));
    }

    #[test]
    fn compute_mode_copy_clone_eq() {
        fn clone_of<T: Clone>(value: &T) -> T {
            value.clone()
        }
        let a = ComputeMode::PythonCompat;
        let b = a; // Copy
        let c = clone_of(&a); // Clone
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
        let python_compat = SpectralInitConfig::new(2).with_seed(42);
        let rust_native = python_compat
            .clone()
            .with_compute_mode(ComputeMode::RustNative);
        let r1 =
            spectral_init(g.view(), None, &python_compat).expect("PythonCompat should succeed");
        let r2 = spectral_init(g.view(), None, &rust_native).expect("RustNative should succeed");
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
}
