//! Assertion macros shared by the test binaries.

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
