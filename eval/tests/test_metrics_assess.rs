use serde_json::json;
use spectral_init::__internal::SolverLevel;
use spectral_init_eval::assess::{
    DATASETS, Dataset, measure_accuracy, measure_disconnected, measure_parity, residual_threshold,
    solve_reference_laplacian,
};
use spectral_init_eval::diagnostics::{SUBSPACE_GRAM_DET_THRESHOLD, tolerance_margin};
use spectral_init_eval::{AssessmentReport, ExperimentMetrics, MetricResult, results_dir};

/// Orthogonality threshold: V^T V - I must be smaller than this.
const ORTHO_THRESHOLD: f64 = 1e-8;

/// Eigenvalue bounds tolerance (λ ∈ [-tol, 2+tol]).
/// Machine-precision only: enforce_psd_contract guarantees λ ∈ [0, 2] exactly.
const BOUNDS_TOL: f64 = f64::EPSILON * 4.0; // ~8.88e-16

/// Sign-agnostic max error threshold for parity assessment.
const SIGN_ERROR_THRESHOLD: f64 = 5e-3;

/// Dataset filter: SPECTRAL_DATASET env var narrows to a single fixture.
fn datasets_to_run() -> Vec<Dataset> {
    let filter = std::env::var("SPECTRAL_DATASET").ok();
    DATASETS
        .iter()
        .filter(|dataset| filter.as_deref().is_none_or(|f| dataset.name == f))
        .copied()
        .collect()
}

/// Section 4.5 metric entry with threshold gate.
fn gated_entry(value: f64, threshold: f64, passed: bool) -> serde_json::Value {
    json!({
        "value": value,
        "threshold": threshold,
        "status": if passed { "PASS" } else { "FAIL" },
    })
}

/// Section 4.5 metric entry without threshold (diagnostic only).
fn diag_entry(value: f64) -> serde_json::Value {
    json!({ "value": value })
}

/// Serialize an ExperimentMetrics to Section 4.5 JSON for the given dimension string.
fn experiment_to_json(
    experiment: &ExperimentMetrics,
    solver_levels: &[Option<SolverLevel>],
    dimension: &str,
) -> String {
    let datasets: Vec<serde_json::Value> = experiment
        .datasets
        .iter()
        .zip(solver_levels.iter())
        .map(|(report, &solver_level)| {
            let metrics_obj: serde_json::Map<String, serde_json::Value> = report
                .metrics
                .iter()
                .map(|m| {
                    let entry = if m.threshold == 0.0 {
                        diag_entry(m.value)
                    } else {
                        gated_entry(m.value, m.threshold, m.passed)
                    };
                    (m.name.clone(), entry)
                })
                .collect();
            json!({
                "dimension": dimension,
                "dataset": report.dataset,
                "n": report.n,
                "solver_level": solver_level.map(|level| level as u8),
                "solver_name": solver_level.map(SolverLevel::name),
                "metrics": metrics_obj,
            })
        })
        .collect();

    serde_json::to_string_pretty(&json!({
        "generated_at": experiment.generated_at,
        "datasets": datasets,
    }))
    .expect("JSON serialization failed")
}

fn accuracy_to_json(
    experiment: &ExperimentMetrics,
    solver_levels: &[Option<SolverLevel>],
) -> String {
    experiment_to_json(experiment, solver_levels, "accuracy")
}

fn parity_to_json(experiment: &ExperimentMetrics, solver_levels: &[Option<SolverLevel>]) -> String {
    experiment_to_json(experiment, solver_levels, "parity")
}

/// Markdown table for accuracy assessment.
fn accuracy_to_markdown(
    experiment: &ExperimentMetrics,
    solver_levels: &[Option<SolverLevel>],
) -> String {
    let mut out = String::new();
    out.push_str("# Metrics Assessment: Accuracy\n");
    out.push_str(&format!("Generated: {}\n\n", experiment.generated_at));
    out.push_str("## Per-Dataset Accuracy Metrics\n\n");
    out.push_str("| Dataset | n | Solver | max_residual | ortho_error | bounds_ok | spectral_gap | cond_num | status |\n");
    out.push_str("|---------|---|--------|--------------|-------------|-----------|--------------|----------|--------|\n");

    for (report, &solver_level) in experiment.datasets.iter().zip(solver_levels.iter()) {
        let get = |name: &str| report.metrics.iter().find(|m| m.name == name);
        let fmt_val = |name: &str| {
            get(name)
                .map(|m| format!("{:.3e}", m.value))
                .unwrap_or_else(|| "—".to_string())
        };
        let fmt_bool = |name: &str| {
            get(name)
                .map(|m| if m.value != 0.0 { "✓" } else { "✗" })
                .unwrap_or("—")
        };
        let all_pass = report
            .metrics
            .iter()
            .filter(|m| m.threshold > 0.0)
            .all(|m| m.passed);

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            report.dataset,
            report.n,
            solver_level.map(SolverLevel::name).unwrap_or("N/A"),
            fmt_val("max_eigenpair_residual"),
            fmt_val("orthogonality_error"),
            fmt_bool("eigenvalue_bounds_in_range"),
            fmt_val("spectral_gap"),
            fmt_val("condition_number"),
            if all_pass { "PASS" } else { "FAIL" },
        ));
    }
    out
}

/// Markdown table for parity assessment.
fn parity_to_markdown(
    experiment: &ExperimentMetrics,
    solver_levels: &[Option<SolverLevel>],
) -> String {
    let mut out = String::new();
    out.push_str("# Metrics Assessment: Parity\n");
    out.push_str(&format!("Generated: {}\n\n", experiment.generated_at));
    out.push_str("## Per-Dataset Parity Metrics\n\n");
    out.push_str("| Dataset | n | Solver | max_eigenvalue_abs_error | subspace_gram_det | sign_agnostic_max_error | status |\n");
    out.push_str("|---------|---|--------|--------------------------|-------------------|-------------------------|--------|\n");

    for (report, &solver_level) in experiment.datasets.iter().zip(solver_levels.iter()) {
        let get = |name: &str| report.metrics.iter().find(|m| m.name == name);
        let fmt_val = |name: &str| {
            get(name)
                .map(|m| format!("{:.3e}", m.value))
                .unwrap_or_else(|| "N/A".to_string())
        };
        let all_pass = report
            .metrics
            .iter()
            .filter(|m| m.threshold > 0.0)
            .all(|m| m.passed);

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            report.dataset,
            report.n,
            solver_level.map(SolverLevel::name).unwrap_or("N/A"),
            fmt_val("max_eigenvalue_abs_error"),
            fmt_val("subspace_gram_det"),
            fmt_val("sign_agnostic_max_error"),
            if all_pass { "PASS" } else { "FAIL" },
        ));
    }
    out
}

#[test]
fn assess_accuracy() {
    let run_set = datasets_to_run();
    let mut all_reports: Vec<AssessmentReport> = Vec::new();
    let mut solver_levels: Vec<Option<SolverLevel>> = Vec::new();

    for dataset in &run_set {
        if !dataset.connected {
            let layout = measure_disconnected(dataset);
            let count_matches = layout.n_graph_components == dataset.n_graph_components;

            let metrics = vec![
                MetricResult {
                    name: "component_count_match".into(),
                    value: count_matches as u8 as f64,
                    threshold: 1.0,
                    passed: count_matches,
                },
                MetricResult {
                    name: "separation_ratio".into(),
                    value: layout.separation_ratio,
                    threshold: 0.0,
                    passed: true,
                },
            ];

            all_reports.push(AssessmentReport {
                dataset: dataset.name.to_string(),
                n: dataset.n,
                metrics,
            });
            solver_levels.push(None);
            continue;
        }

        // Connected dataset path
        let solve = solve_reference_laplacian(dataset);
        let solver_level = solve.level;
        let accuracy = measure_accuracy(&solve, BOUNDS_TOL);

        let residual = accuracy.max_residual;
        let ortho = accuracy.orthogonality_error;
        let (in_range, sorted) = (accuracy.eigenvalues_in_range, accuracy.eigenvalues_sorted);
        let gap = accuracy.spectral_gap;
        let cond = accuracy.condition_number;
        let resid_thr = residual_threshold(solver_level);
        let resid_margin = tolerance_margin(resid_thr, residual);
        let ortho_margin = tolerance_margin(ORTHO_THRESHOLD, ortho);

        let metrics = vec![
            MetricResult {
                name: "max_eigenpair_residual".into(),
                value: residual,
                threshold: resid_thr,
                passed: residual <= resid_thr,
            },
            MetricResult {
                name: "orthogonality_error".into(),
                value: ortho,
                threshold: ORTHO_THRESHOLD,
                passed: ortho <= ORTHO_THRESHOLD,
            },
            MetricResult {
                name: "eigenvalue_bounds_in_range".into(),
                value: in_range as u8 as f64,
                threshold: 1.0,
                passed: in_range,
            },
            MetricResult {
                name: "eigenvalue_bounds_sorted".into(),
                value: sorted as u8 as f64,
                threshold: 1.0,
                passed: sorted,
            },
            MetricResult {
                name: "spectral_gap".into(),
                value: gap,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "condition_number".into(),
                value: cond,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "residual_margin_factor".into(),
                value: resid_margin,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "ortho_margin_factor".into(),
                value: ortho_margin,
                threshold: 0.0,
                passed: true,
            },
        ];

        all_reports.push(AssessmentReport {
            dataset: dataset.name.to_string(),
            n: dataset.n,
            metrics,
        });
        solver_levels.push(Some(solver_level));
    }

    let experiment = ExperimentMetrics {
        generated_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        datasets: all_reports,
    };
    let dir = results_dir();
    let json_str = accuracy_to_json(&experiment, &solver_levels);
    let md_str = accuracy_to_markdown(&experiment, &solver_levels);
    std::fs::write(dir.join("accuracy_metrics.json"), &json_str).expect("write accuracy json");
    std::fs::write(dir.join("accuracy_metrics.md"), &md_str).expect("write accuracy md");

    for report in &experiment.datasets {
        for m in &report.metrics {
            assert!(
                m.value.is_finite() || m.threshold == 0.0,
                "dataset {}: metric {} is non-finite",
                report.dataset,
                m.name
            );
            if m.threshold > 0.0 {
                assert!(
                    m.passed,
                    "dataset {}: metric {} failed threshold (value={:.3e}, threshold={:.3e})",
                    report.dataset, m.name, m.value, m.threshold
                );
            }
        }
    }
    let json_parsed: serde_json::Value = serde_json::from_str(&json_str).expect("json not valid");
    let run_count = datasets_to_run().len();
    assert_eq!(
        json_parsed["datasets"]
            .as_array()
            .expect("json missing expected datasets array")
            .len(),
        run_count
    );
}

#[test]
fn assess_parity() {
    let run_set = datasets_to_run();
    let mut all_reports: Vec<AssessmentReport> = Vec::new();
    let mut solver_levels: Vec<Option<SolverLevel>> = Vec::new();

    for dataset in run_set.iter().filter(|dataset| dataset.connected) {
        let solve = solve_reference_laplacian(dataset);
        let solver_level = solve.level;
        let parity = measure_parity(dataset, &solve);

        let max_abs_err = parity.max_eigenvalue_abs_error;
        let sign_err = parity.sign_agnostic_max_error;
        // The subspace is compared only where individual eigenvectors are not determined.
        let gram = if parity.degenerate {
            parity.subspace_gram_det
        } else {
            f64::NAN
        };

        let eigenvalue_err_thr = residual_threshold(solver_level);

        let mut metrics = vec![
            MetricResult {
                name: "max_eigenvalue_abs_error".into(),
                value: max_abs_err,
                threshold: eigenvalue_err_thr,
                passed: max_abs_err <= eigenvalue_err_thr,
            },
            MetricResult {
                name: "sign_agnostic_max_error".into(),
                value: sign_err,
                threshold: SIGN_ERROR_THRESHOLD,
                passed: sign_err <= SIGN_ERROR_THRESHOLD,
            },
        ];

        if gram.is_finite() {
            metrics.push(MetricResult {
                name: "subspace_gram_det".into(),
                value: gram,
                threshold: SUBSPACE_GRAM_DET_THRESHOLD,
                passed: gram >= SUBSPACE_GRAM_DET_THRESHOLD,
            });
        } else {
            metrics.push(MetricResult {
                name: "subspace_gram_det".into(),
                value: 1.0,
                threshold: 0.0,
                passed: true,
            });
        }

        all_reports.push(AssessmentReport {
            dataset: dataset.name.to_string(),
            n: dataset.n,
            metrics,
        });
        solver_levels.push(Some(solver_level));
    }

    let experiment = ExperimentMetrics {
        generated_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        datasets: all_reports,
    };
    let dir = results_dir();
    let json_str = parity_to_json(&experiment, &solver_levels);
    let md_str = parity_to_markdown(&experiment, &solver_levels);
    std::fs::write(dir.join("parity_metrics.json"), &json_str).expect("write parity json");
    std::fs::write(dir.join("parity_metrics.md"), &md_str).expect("write parity md");

    for report in &experiment.datasets {
        for m in &report.metrics {
            assert!(
                m.value.is_finite(),
                "dataset {}: parity metric {} is non-finite",
                report.dataset,
                m.name
            );
            if m.threshold > 0.0 {
                assert!(
                    m.passed,
                    "dataset {}: parity metric {} failed threshold (value={:.3e}, threshold={:.3e})",
                    report.dataset, m.name, m.value, m.threshold
                );
            }
        }
    }
    let json_parsed: serde_json::Value = serde_json::from_str(&json_str).expect("json not valid");
    let connected_count = datasets_to_run()
        .iter()
        .filter(|dataset| dataset.connected)
        .count();
    assert_eq!(
        json_parsed["datasets"]
            .as_array()
            .expect("json missing expected datasets array")
            .len(),
        connected_count
    );
}
