/// Controls whether to match Python UMAP behavior exactly or use
/// best-available Rust-native algorithms.
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

/// Configuration for the spectral initialization pipeline.
///
/// Construct with `SpectralInitConfig::default()` for Python-compatible behavior.
///
/// # Examples
///
/// ```
/// use spectral_init::{SpectralInitConfig, ComputeMode};
///
/// // Default: Python-compatible behavior
/// let config = SpectralInitConfig::default();
/// assert_eq!(config.compute_mode, ComputeMode::PythonCompat);
///
/// // Custom: Rust-native SIMD acceleration (may differ ~1 ULP from Python)
/// let mut fast = SpectralInitConfig::default();
/// fast.compute_mode = ComputeMode::RustNative;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct SpectralInitConfig {
    /// Controls which algorithm variant to use when Rust-native and Python-compatible
    /// implementations differ in quality.
    pub compute_mode: ComputeMode,
}
