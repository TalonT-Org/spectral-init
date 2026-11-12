//! Spectral initialization for UMAP embeddings.
//!
//! Computes the spectral layout that umap-learn uses as its default starting point for
//! stochastic gradient descent: the eigenvectors of the normalized graph Laplacian that
//! belong to its smallest non-zero eigenvalues. These eigenvectors vary slowly across the
//! graph, so the coordinates reflect its large-scale structure before the optimizer
//! takes a step.
//!
//! The input is a symmetric, non-negative sparse graph in [`sprs`] storage, such as the
//! fuzzy k-nearest-neighbour graph of UMAP. The crate depends on no UMAP implementation.
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
//! where $W$ is the weighted adjacency matrix and $D$ the diagonal matrix of its row
//! sums, in double precision. The eigenvectors of the $k$ smallest non-zero eigenvalues
//! of $L$ form the coordinate matrix; the eigenvector of $\lambda = 0$, proportional to
//! $\sqrt{d}$, is skipped. Each column is given a fixed sign, all coordinates are scaled
//! by one factor so that the largest magnitude is 10, and Gaussian noise with standard
//! deviation $10^{-4}$ is added.
//!
//! A graph with several connected components is embedded component by component, and
//! the components are then placed relative to one another; see [`spectral_init`].
//!
//! The eigenvectors come from a chain of six solvers of increasing cost: a dense
//! eigendecomposition for small graphs, then LOBPCG, shift-and-invert LOBPCG, regularized
//! LOBPCG, a randomized SVD and, last, a dense eigendecomposition for any size that fits
//! the [dense memory budget](SpectralInitConfig::with_dense_memory_budget). A result is
//! accepted only if its residual $\lVert L v - \lambda v \rVert / \lVert v \rVert$ is
//! below the threshold of its solver; when no solver produces one, the call returns an
//! error.
//!
//! # Compute Mode
//!
//! [`ComputeMode::PythonCompat`] (default) follows umap-learn's arithmetic where it
//! affects the result: degrees are accumulated in single precision, in SciPy's order, and
//! the iterative solvers use a scalar matrix-vector product.
//! [`ComputeMode::RustNative`] sums degrees in double precision on every target and, on
//! x86_64 CPUs with AVX2 and FMA, runs the matrix-vector product of the iterative solvers
//! through a SIMD kernel. The two modes span the same embedding subspace and differ in
//! the coordinates by rounding.
//!
//! The reference for `PythonCompat` is the spectral layout of umap-learn 0.5.x, which
//! umap-learn 0.6 keeps as its default. The test suite compares every stage with values
//! recorded from umap-learn 0.5.11: degrees within $10^{-15}$, Laplacian entries within
//! $10^{-14}$, eigenvalues within the threshold of the solver that produced them, and
//! coordinates up to the sign of each column. Where eigenvalues nearly coincide, the
//! spanned subspace is compared instead of the individual vectors.
//!
//! # Feature Flags
//!
//! The crate has no public Cargo features.
//!
//! # Use with a UMAP Implementation
//!
//! `umap-rs` exposes the graph this crate needs: `Umap::learn_manifold(..).graph()` is a
//! `CsMatI<f32, u32, usize>`, and its `Optimizer` takes the initial coordinates. The
//! [integration guide] has the full sequence, and `examples/umap_rs.rs` is a runnable
//! version. An implementation that keeps its graph as raw CSR arrays can lend them to
//! [`spectral_init`] as a [`sprs::CsMatViewI`] without copying.
//!
//! [integration guide]: https://github.com/TalonT-Org/spectral-init/blob/main/docs/umap-rs-integration.md
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
//! # Reproducibility
//!
//! For a fixed `Cargo.lock`, target, CPU feature set and thread configuration, the output
//! is a deterministic function of the graph, the data and the configuration. Random
//! numbers come from the ChaCha12 generator (`rand_chacha::ChaCha12Rng`), seeded with
//! [`SpectralInitConfig::with_seed`].
//!
//! Across thread configurations the output agrees to rounding, not bit for bit: the dense
//! eigensolver splits larger problems across threads. Measured on x86_64 with faer 0.24.4,
//! one thread against four changed no coordinate for a 500-node graph and changed
//! coordinates by at most 1.2 × 10⁻¹⁰ for a 1500-node graph (coordinates are scaled to a
//! maximum magnitude of 10); the test suite accepts differences up to 10⁻⁵. To pin the
//! thread configuration, call `faer::set_global_parallelism` before [`spectral_init`], or
//! set the `RAYON_NUM_THREADS` environment variable, which sizes the thread pool the
//! solvers share. A different target or CPU feature set can change results by rounding in
//! the same way; no bitwise agreement is promised across them.
//!
//! # Public Dependencies
//!
//! The graph and array types in the signature of [`spectral_init`] come from the
//! [`sprs`] and [`ndarray`] crates, which are re-exported so that callers can name the
//! exact versions this crate was built against. Moving to a semver-incompatible release
//! of either is a breaking change of this crate.
//!
//! # User Guide
//!
//! The [user guide](https://github.com/TalonT-Org/spectral-init/blob/main/docs/user-guide.md)
//! covers building the input graph, configuration, disconnected graphs and
//! troubleshooting.

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
    pub use crate::components::find_components;
    pub use crate::laplacian::{build_normalized_laplacian, compute_degrees};
    pub use crate::multi_component::embed_disconnected;
    pub use crate::operator::{CsrOperator, LinearOperator, spmv_csr};
    #[cfg(target_arch = "x86_64")]
    pub use crate::operator::{CsrOperatorSimd, simd_index_fits, spmv_avx2_gather};
    pub use crate::scaling::{normalize_signs, scale_and_add_noise, scale_coords};
    pub use crate::selection::select_eigenvectors;
    pub use crate::solvers::dense_evd;
    pub use crate::solvers::lobpcg::lobpcg_solve;
    pub use crate::solvers::quality::*;
    pub use crate::solvers::rsvd::{rsvd_solve, rsvd_solve_accurate};
    pub use crate::solvers::sinv::lobpcg_sinv_solve;
    pub use crate::solvers::{
        EigenResult, QualityGates, SolveReport, SolverLevel, SolverParams, solve_eigenproblem,
    };
}

/// The README and the guides shipped with the crate, attached as documentation so that
/// their Rust snippets are compiled and run as doctests.
#[cfg(doctest)]
mod guides {
    #[doc = include_str!("../README.md")]
    struct Readme;

    #[doc = include_str!("../docs/user-guide.md")]
    struct UserGuide;

    #[doc = include_str!("../docs/umap-rs-integration.md")]
    struct UmapRsIntegration;
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
