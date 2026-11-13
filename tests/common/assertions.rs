//! Assertion macros and helpers shared by the test binaries.

use proptest::prelude::ProptestConfig;
use proptest::test_runner::FileFailurePersistence;
use spectral_init::ndarray::Array2;
use spectral_init::sprs::CsMatI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

/// Property-test configuration. Failing cases persist to
/// `<test file>.proptest-regressions`, next to the file that runs the property.
pub fn property_config(cases: u32, max_shrink_iters: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        max_shrink_iters,
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..Default::default()
    }
}

/// Outcome of validating `graph` alone: a data argument with no rows turns "the graph is
/// valid" into `DataShapeMismatch`, the last-but-one check, without running a solver.
pub fn validate_graph(graph: &CsMatI<f32, u32, usize>) -> Result<(), SpectralError> {
    let no_rows = Array2::<f32>::zeros((0, 1));
    match spectral_init(
        graph.view(),
        Some(no_rows.view()),
        &SpectralInitConfig::new(1),
    ) {
        Err(SpectralError::DataShapeMismatch { .. }) => Ok(()),
        Err(other) => Err(other),
        Ok(_) => panic!("a data argument without rows must be rejected"),
    }
}

/// Asserts that an expression is an `Err` matching a pattern.
///
/// The variants of `SpectralError` are `#[non_exhaustive]`, so code outside the crate
/// cannot build one to compare against; it matches them with `..` instead.
macro_rules! assert_err {
    ($result:expr, $pattern:pat $(,)?) => {
        match $result {
            Err($pattern) => {}
            other => panic!(
                "expected Err({}), got {:?}",
                stringify!($pattern),
                other
            ),
        }
    };
    ($result:expr, $pattern:pat, $($context:tt)+) => {
        match $result {
            Err($pattern) => {}
            other => panic!(
                "{}: expected Err({}), got {:?}",
                format_args!($($context)+),
                stringify!($pattern),
                other
            ),
        }
    };
}
