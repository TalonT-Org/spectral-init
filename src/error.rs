/// Errors returned by [`spectral_init`](crate::spectral_init).
///
/// The enum and each of its struct-like variants are `#[non_exhaustive]`: match with a
/// wildcard arm and with `..` in variant patterns, so that variants and fields added in
/// later releases do not break the build.
///
/// # Examples
///
/// ```
/// use spectral_init::{SpectralError, SpectralInitConfig, spectral_init, sprs::CsMatI};
///
/// // One directed edge 0 → 1 with no edge back: not a symmetric graph.
/// let graph = CsMatI::<f32, u32, usize>::new(
///     (3, 3),
///     vec![0, 1, 1, 1],
///     vec![1],
///     vec![1.0],
/// );
/// match spectral_init(graph.view(), None, &SpectralInitConfig::new(2)) {
///     Err(SpectralError::NotSymmetric { row, col, .. }) => assert_eq!((row, col), (0, 1)),
///     other => panic!("expected NotSymmetric, got {other:?}"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SpectralError {
    /// A configuration value is outside its valid range.
    #[error("invalid parameter `{name}`: {reason}")]
    #[non_exhaustive]
    InvalidParameter {
        /// Name of the rejected parameter.
        name: &'static str,
        /// Why the value was rejected.
        reason: &'static str,
    },

    /// The graph matrix is not square.
    #[error("graph matrix is not square: {rows} rows, {cols} columns")]
    #[non_exhaustive]
    NotSquare {
        /// Number of rows of the graph matrix.
        rows: usize,
        /// Number of columns of the graph matrix.
        cols: usize,
    },

    /// The graph has too few nodes for the requested embedding dimensionality:
    /// `n_components` coordinates need at least `n_components + 1` nodes.
    #[error("graph has too few nodes ({n}) for a {n_components}-dimensional embedding")]
    #[non_exhaustive]
    TooFewNodes {
        /// Number of nodes in the graph.
        n: usize,
        /// Requested number of embedding dimensions.
        n_components: usize,
    },

    /// A stored edge weight is NaN or infinite.
    #[error("edge weight at ({row}, {col}) is not finite")]
    #[non_exhaustive]
    NonFiniteWeight {
        /// Row of the offending entry, in the orientation the caller supplied.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },

    /// A stored edge weight is negative.
    #[error("edge weight at ({row}, {col}) is negative")]
    #[non_exhaustive]
    NegativeWeight {
        /// Row of the offending entry, in the orientation the caller supplied.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },

    /// The graph matrix is not symmetric: the entry at `(row, col)` has no matching
    /// entry at `(col, row)`.
    #[error("graph matrix is not symmetric: entry ({row}, {col}) differs from ({col}, {row})")]
    #[non_exhaustive]
    NotSymmetric {
        /// Row of the offending entry, in the orientation the caller supplied.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },

    /// The summed edge weights of a node overflow single precision.
    #[error("degree of node {node} is not finite")]
    #[non_exhaustive]
    DegreeOverflow {
        /// Index of the node whose degree overflowed.
        node: usize,
    },

    /// The graph has more than `2 · n_components` connected components, and placing
    /// that many components relative to each other needs the original data.
    #[error(
        "graph has {connected_components} connected components; `data` is required when \
         there are more than twice the {n_components} embedding dimensions"
    )]
    #[non_exhaustive]
    DataRequired {
        /// Number of connected components found in the graph.
        connected_components: usize,
        /// Requested number of embedding dimensions.
        n_components: usize,
    },

    /// `data` does not have one row per graph node.
    #[error("data has {actual_rows} rows but the graph has {expected_rows} nodes")]
    #[non_exhaustive]
    DataShapeMismatch {
        /// Number of graph nodes.
        expected_rows: usize,
        /// Number of rows in `data`.
        actual_rows: usize,
    },

    /// A value in `data` is NaN or infinite.
    #[error("data value at ({row}, {col}) is not finite")]
    #[non_exhaustive]
    NonFiniteData {
        /// Row of the offending value.
        row: usize,
        /// Column of the offending value.
        col: usize,
    },

    /// Solving the problem would need a dense matrix larger than the configured
    /// [dense memory budget](crate::SpectralInitConfig::with_dense_memory_budget).
    #[error(
        "a dense matrix of {required_bytes} bytes exceeds the dense memory budget of \
         {budget_bytes} bytes"
    )]
    #[non_exhaustive]
    DenseBudgetExceeded {
        /// Size in bytes of the dense matrix that was needed.
        required_bytes: u64,
        /// The configured budget in bytes.
        budget_bytes: u64,
    },

    /// No solver in the escalation chain produced eigenvectors of acceptable quality.
    #[error("eigensolver failed to converge on a problem of size {n}")]
    #[non_exhaustive]
    ConvergenceFailure {
        /// Size of the eigenproblem whose solve failed.
        n: usize,
    },

    /// The computed coordinates are all zero, so they cannot be scaled.
    #[error("the embedding is degenerate: all coordinates are zero")]
    DegenerateEmbedding,
}
