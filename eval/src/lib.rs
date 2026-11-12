//! Research harness for `spectral-init`.
//!
//! Holds the measurements and report structures shared by the assessment tests in
//! `tests/`, the `init_evidence` binary and the tooling that reads their JSON output,
//! and the locations the harness reads from and writes to.

pub mod assess;
#[path = "../../tests/common/diagnostics.rs"]
pub mod diagnostics;
#[path = "../../tests/common/fixtures.rs"]
pub mod fixtures;
#[path = "../../tests/common/internals.rs"]
pub mod internals;

use std::path::{Path, PathBuf};

/// Root of the workspace this package belongs to. Reference fixtures and the
/// visual-evaluation data live under it.
pub const WORKSPACE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// Directory the harness writes its results to, created if absent: `$RESULTS_DIR` when
/// that variable is set, otherwise `temp/eval/` under the workspace root.
pub fn results_dir() -> PathBuf {
    let dir = match std::env::var_os("RESULTS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(WORKSPACE_ROOT).join("temp").join("eval"),
    };
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create the results directory {dir:?}: {e}"));
    dir
}

/// One measured quantity with the threshold it was judged against.
#[derive(Debug, Clone)]
pub struct MetricResult {
    pub name: String,
    pub value: f64,
    pub threshold: f64,
    pub passed: bool,
}

/// All metrics measured on one dataset.
#[derive(Debug, Clone)]
pub struct AssessmentReport {
    pub dataset: String,
    pub n: usize,
    pub metrics: Vec<MetricResult>,
}

/// One assessment run across datasets.
#[derive(Debug, Clone)]
pub struct ExperimentMetrics {
    pub generated_at: String,
    pub datasets: Vec<AssessmentReport>,
}
