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

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Root of the workspace this package belongs to. Reference fixtures and the
/// visual-evaluation data live under it.
pub const WORKSPACE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// Directory the harness writes its results to: `$RESULTS_DIR` when that variable is
/// set, otherwise `temp/eval/` under the workspace root.
pub fn results_dir() -> PathBuf {
    match std::env::var_os("RESULTS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(WORKSPACE_ROOT).join("temp").join("eval"),
    }
}

/// One measured quantity with the threshold it was judged against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub name: String,
    pub dimension: usize,
    pub value: f64,
    pub threshold: f64,
    pub passed: bool,
}

/// All metrics measured on one dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentReport {
    pub dataset: String,
    pub n: usize,
    pub metrics: Vec<MetricResult>,
}

/// One assessment run across datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub generated_at: String,
    pub datasets: Vec<AssessmentReport>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_serde_01_round_trip_metric_result() {
        let m = MetricResult {
            name: "test_metric".to_string(),
            dimension: 2,
            value: 0.001,
            threshold: 0.01,
            passed: true,
        };
        let json = serde_json::to_string(&m).expect("serialize MetricResult");
        let m2: MetricResult = serde_json::from_str(&json).expect("deserialize MetricResult");
        assert_eq!(m2.name, m.name);
        assert_eq!(m2.dimension, m.dimension);
        assert_eq!(m2.value, m.value);
        assert_eq!(m2.threshold, m.threshold);
        assert_eq!(m2.passed, m.passed);
    }

    #[test]
    fn t_serde_02_round_trip_assessment_report() {
        let r = AssessmentReport {
            dataset: "test_dataset".to_string(),
            n: 100,
            metrics: vec![MetricResult {
                name: "residual".to_string(),
                dimension: 2,
                value: 1e-7,
                threshold: 1e-6,
                passed: true,
            }],
        };
        let json = serde_json::to_string(&r).expect("serialize AssessmentReport");
        let r2: AssessmentReport =
            serde_json::from_str(&json).expect("deserialize AssessmentReport");
        assert_eq!(r2.dataset, r.dataset);
        assert_eq!(r2.n, r.n);
        assert_eq!(r2.metrics.len(), 1);
        assert_eq!(r2.metrics[0].name, "residual");
    }

    #[test]
    fn t_serde_03_round_trip_experiment_metrics() {
        let e = ExperimentMetrics {
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            datasets: vec![AssessmentReport {
                dataset: "iris".to_string(),
                n: 150,
                metrics: vec![],
            }],
        };
        let json = serde_json::to_string(&e).expect("serialize ExperimentMetrics");
        let e2: ExperimentMetrics =
            serde_json::from_str(&json).expect("deserialize ExperimentMetrics");
        assert_eq!(e2.generated_at, e.generated_at);
        assert_eq!(e2.datasets.len(), 1);
        assert_eq!(e2.datasets[0].dataset, "iris");
    }
}
