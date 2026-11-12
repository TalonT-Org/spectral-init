/// Controls whether to match Python UMAP behavior exactly or use
/// best-available Rust-native algorithms.
///
/// # Examples
///
/// ```
/// use spectral_init::{ComputeMode, SpectralInitConfig};
///
/// // `PythonCompat` is the default.
/// assert_eq!(SpectralInitConfig::new(2).compute_mode(), ComputeMode::PythonCompat);
///
/// let config = SpectralInitConfig::new(2).with_compute_mode(ComputeMode::RustNative);
/// assert_eq!(config.compute_mode(), ComputeMode::RustNative);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ComputeMode {
    /// Match Python UMAP behavior exactly. This is the default.
    ///
    /// Degree accumulation uses f32 column-sum scatter-add to match SciPy's `csc_matvec`
    /// behavior, and LOBPCG levels use the scalar matrix-vector kernel. Produces outputs
    /// that are bit-for-bit comparable to the Python UMAP reference implementation for
    /// validation purposes.
    #[default]
    PythonCompat,
    /// Use best-available Rust-native algorithms.
    ///
    /// On x86_64 with AVX2+FMA, LOBPCG levels route through a SIMD matrix-vector kernel
    /// for improved throughput. Degree accumulation uses f64 row-sum for better numerical
    /// precision. Results may diverge from Python UMAP by approximately 1 ULP. Use this
    /// mode when performance matters more than exact Python parity.
    RustNative,
}

/// Configuration for [`spectral_init`](crate::spectral_init).
///
/// Create one with [`SpectralInitConfig::new`], giving the number of embedding
/// dimensions, and adjust it with the `with_*` methods.
///
/// # Examples
///
/// ```
/// use spectral_init::{ComputeMode, SpectralInitConfig};
///
/// let config = SpectralInitConfig::new(2);
/// assert_eq!(config.n_components(), 2);
/// assert_eq!(config.seed(), 0);
/// assert_eq!(config.compute_mode(), ComputeMode::PythonCompat);
///
/// let config = SpectralInitConfig::new(3)
///     .with_seed(42)
///     .with_compute_mode(ComputeMode::RustNative)
///     .with_dense_memory_budget(1 << 30);
/// assert_eq!(config.seed(), 42);
/// assert_eq!(config.dense_memory_budget(), 1 << 30);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralInitConfig {
    n_components: usize,
    seed: u64,
    compute_mode: ComputeMode,
    dense_memory_budget: u64,
}

impl SpectralInitConfig {
    /// Default [dense memory budget](Self::with_dense_memory_budget): 4 GiB.
    pub const DEFAULT_DENSE_MEMORY_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

    /// Creates a configuration for an embedding with `n_components` dimensions.
    ///
    /// The seed is 0, the compute mode is [`ComputeMode::PythonCompat`] and the dense
    /// memory budget is [`Self::DEFAULT_DENSE_MEMORY_BUDGET`].
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            seed: 0,
            compute_mode: ComputeMode::default(),
            dense_memory_budget: Self::DEFAULT_DENSE_MEMORY_BUDGET,
        }
    }

    /// Sets the seed of the random number generators the solvers and the final noise use.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the [`ComputeMode`].
    #[must_use]
    pub fn with_compute_mode(mut self, compute_mode: ComputeMode) -> Self {
        self.compute_mode = compute_mode;
        self
    }

    /// Sets the dense memory budget, in bytes.
    ///
    /// The budget caps the size of every dense matrix a solver allocates: a solver that
    /// would need a larger one is skipped, and [`SpectralError::DenseBudgetExceeded`] is
    /// returned when no remaining solver can run within it. It bounds each dense matrix
    /// individually, not the peak memory of the process — some solvers hold a few
    /// matrices of that size at once.
    ///
    /// [`SpectralError::DenseBudgetExceeded`]: crate::SpectralError::DenseBudgetExceeded
    #[must_use]
    pub fn with_dense_memory_budget(mut self, bytes: u64) -> Self {
        self.dense_memory_budget = bytes;
        self
    }

    /// Number of embedding dimensions.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Seed of the random number generators.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The [`ComputeMode`].
    pub fn compute_mode(&self) -> ComputeMode {
        self.compute_mode
    }

    /// Dense memory budget, in bytes.
    pub fn dense_memory_budget(&self) -> u64 {
        self.dense_memory_budget
    }
}
