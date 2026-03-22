/// Controls whether to match Python UMAP behavior exactly or use
/// best-available Rust-native algorithms.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ComputeMode {
    /// Match Python UMAP behavior exactly. This is the default.
    /// Produces outputs comparable to the reference implementation.
    #[default]
    PythonCompat,
    /// Use best-available Rust-native algorithms where they are
    /// genuinely superior (more accurate, more precise, faster, etc.).
    /// May diverge from Python reference outputs.
    RustNative,
}

/// Configuration for the spectral initialization pipeline.
///
/// Construct with `SpectralInitConfig::default()` for Python-compatible behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct SpectralInitConfig {
    /// Controls which algorithm variant to use when Rust-native and Python-compatible
    /// implementations differ in quality.
    pub compute_mode: ComputeMode,
}
