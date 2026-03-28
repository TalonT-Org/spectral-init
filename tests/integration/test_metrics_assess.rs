#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2};
use ndarray::s;
use serde_json::json;
use spectral_init::{
    check_eigenvalue_bounds,
    eigenvalue_abs_errors,
    eigenvalue_condition_number,
    embed_disconnected,
    find_components,
    max_eigenpair_residual,
    orthogonality_error,
    select_eigenvectors,
    separation_ratio,
    sign_agnostic_max_error,
    solve_eigenproblem_pub,
    spectral_gap,
    subspace_gram_det_kd,
    tolerance_margin,
    AssessmentReport,
    ComputeMode,
    ExperimentMetrics,
    MetricResult,
    DEGENERATE_GAP_THRESHOLD,
    DENSE_EVD_QUALITY_THRESHOLD,
    LOBPCG_QUALITY_THRESHOLD,
    RSVD_QUALITY_THRESHOLD,
    SINV_LOBPCG_QUALITY_THRESHOLD,
    SUBSPACE_GRAM_DET_THRESHOLD,
};

/// (name, n, is_connected, expected_n_components)
const DATASETS: &[(&str, usize, bool, usize)] = &[
    ("blobs_50",             50,   false, 3),
    ("blobs_500",            500,  false, 4),
    ("blobs_5000",           5000, false, 2),
    ("blobs_connected_200",  200,  true,  1),
    ("blobs_connected_2000", 2000, true,  1),
    ("circles_300",          300,  true,  1),
    ("disconnected_200",     200,  false, 4),
    ("moons_200",            200,  true,  1),
    ("near_dupes_100",       100,  true,  1),
];

/// Orthogonality threshold: V^T V - I must be smaller than this.
const ORTHO_THRESHOLD: f64 = 1e-8;

/// Eigenvalue bounds tolerance (λ ∈ [-tol, 2+tol]).
const BOUNDS_TOL: f64 = 1e-12;

/// Sign-agnostic max error threshold for parity assessment.
const SIGN_ERROR_THRESHOLD: f64 = 5e-3;

/// Output directory: RESULTS_DIR env var or default.
fn results_dir() -> std::path::PathBuf {
    match std::env::var("RESULTS_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".autoskillit/temp/run-experiment"),
    }
}

/// Dataset filter: SPECTRAL_DATASET env var narrows to a single fixture.
fn datasets_to_run() -> Vec<(&'static str, usize, bool, usize)> {
    let filter = std::env::var("SPECTRAL_DATASET").ok();
    DATASETS
        .iter()
        .filter(|(name, ..)| filter.as_deref().map_or(true, |f| *name == f))
        .copied()
        .collect()
}

/// Canonical solver name from level byte.
fn solver_name(level: u8) -> &'static str {
    match level {
        0 => "Dense EVD",
        1 => "LOBPCG",
        2 => "SINV LOBPCG",
        3 => "LOBPCG+reg",
        4 => "rSVD",
        5 => "Forced Dense EVD",
        _ => "Unknown",
    }
}

/// Per-solver residual threshold.
fn residual_threshold(level: u8) -> f64 {
    match level {
        0 | 5 => DENSE_EVD_QUALITY_THRESHOLD,
        1 | 3 => LOBPCG_QUALITY_THRESHOLD,
        2     => SINV_LOBPCG_QUALITY_THRESHOLD,
        _     => RSVD_QUALITY_THRESHOLD,
    }
}

/// Scale eigenvectors to pre-noise f32 (matching Python's 10/max_abs expansion).
fn compute_pre_noise(selected: &Array2<f64>) -> Array2<f32> {
    let max_abs = selected.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 {
        return Array2::<f32>::zeros(selected.raw_dim());
    }
    let expansion = 10.0 / max_abs;
    selected.mapv(|v| (v * expansion) as f32)
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
    solver_levels: &[Option<u8>],
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
                "solver_level": solver_level,
                "solver_name": solver_level.map(solver_name),
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

fn accuracy_to_json(experiment: &ExperimentMetrics, solver_levels: &[Option<u8>]) -> String {
    experiment_to_json(experiment, solver_levels, "accuracy")
}

fn parity_to_json(experiment: &ExperimentMetrics, solver_levels: &[Option<u8>]) -> String {
    experiment_to_json(experiment, solver_levels, "parity")
}

/// Markdown table for accuracy assessment.
fn accuracy_to_markdown(experiment: &ExperimentMetrics, solver_levels: &[Option<u8>]) -> String {
    let mut out = String::new();
    out.push_str("# Metrics Assessment: Accuracy\n");
    out.push_str(&format!("Generated: {}\n\n", experiment.generated_at));
    out.push_str("## Per-Dataset Accuracy Metrics\n\n");
    out.push_str("| Dataset | n | Solver | max_residual | ortho_error | bounds_ok | spectral_gap | cond_num | status |\n");
    out.push_str("|---------|---|--------|--------------|-------------|-----------|--------------|----------|--------|\n");

    for (report, &solver_level) in experiment.datasets.iter().zip(solver_levels.iter()) {
        let get = |name: &str| report.metrics.iter().find(|m| m.name == name);
        let fmt_val = |name: &str| get(name).map(|m| format!("{:.3e}", m.value)).unwrap_or_else(|| "—".to_string());
        let fmt_bool = |name: &str| get(name).map(|m| if m.value != 0.0 { "✓" } else { "✗" }).unwrap_or("—");
        let all_pass = report.metrics.iter().filter(|m| m.threshold > 0.0).all(|m| m.passed);

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            report.dataset,
            report.n,
            solver_level.map(solver_name).unwrap_or("N/A"),
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
fn parity_to_markdown(experiment: &ExperimentMetrics, solver_levels: &[Option<u8>]) -> String {
    let mut out = String::new();
    out.push_str("# Metrics Assessment: Parity\n");
    out.push_str(&format!("Generated: {}\n\n", experiment.generated_at));
    out.push_str("## Per-Dataset Parity Metrics\n\n");
    out.push_str("| Dataset | n | Solver | max_eigenvalue_abs_error | subspace_gram_det | sign_agnostic_max_error | status |\n");
    out.push_str("|---------|---|--------|--------------------------|-------------------|-------------------------|--------|\n");

    for (report, &solver_level) in experiment.datasets.iter().zip(solver_levels.iter()) {
        let get = |name: &str| report.metrics.iter().find(|m| m.name == name);
        let fmt_val = |name: &str| get(name).map(|m| format!("{:.3e}", m.value)).unwrap_or_else(|| "N/A".to_string());
        let all_pass = report.metrics.iter().filter(|m| m.threshold > 0.0).all(|m| m.passed);

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            report.dataset,
            report.n,
            solver_level.map(solver_name).unwrap_or("N/A"),
            fmt_val("max_eigenvalue_abs_error"),
            fmt_val("subspace_gram_det"),
            fmt_val("sign_agnostic_max_error"),
            if all_pass { "PASS" } else { "FAIL" },
        ));
    }
    out
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn assess_accuracy() {
    let run_set = datasets_to_run();
    let mut all_reports: Vec<AssessmentReport> = Vec::new();
    let mut solver_levels: Vec<Option<u8>> = Vec::new();

    for (dataset, n, is_connected, expected_n_components) in &run_set {
        let base = common::fixture_path(dataset, "");

        let ref_eigenvalues: Array1<f64> =
            common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalues");
        assert!(
            !ref_eigenvalues.is_empty(),
            "{dataset}: ref eigenvalues array is empty — fixture may be corrupt"
        );
        let n_components_dim = ref_eigenvalues.len() - 1;

        if !is_connected {
            let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
            let (rust_labels, rust_n_components) = find_components(&graph);

            let embedding = embed_disconnected(
                &graph,
                &rust_labels,
                rust_n_components,
                n_components_dim,
                42,
                None,
                ComputeMode::PythonCompat,
            )
            .unwrap_or_else(|e| panic!("{dataset}: embed_disconnected failed: {e}"));

            let py_labels_i32: Array1<i32> =
                common::load_dense(&base.join("comp_c_components.npz"), "labels");
            let labels: Vec<usize> = py_labels_i32.iter().map(|&x| x as usize).collect();

            let sep = separation_ratio(embedding.view(), &labels);

            let metrics = vec![
                MetricResult {
                    name: "component_count_match".into(),
                    dimension: 0,
                    value: (rust_n_components == *expected_n_components) as u8 as f64,
                    threshold: 1.0,
                    passed: rust_n_components == *expected_n_components,
                },
                MetricResult {
                    name: "separation_ratio".into(),
                    dimension: 0,
                    value: sep,
                    threshold: 0.0,
                    passed: true,
                },
            ];

            all_reports.push(AssessmentReport {
                dataset: dataset.to_string(),
                n: *n,
                metrics,
            });
            solver_levels.push(None);
            continue;
        }

        // Connected dataset path
        let laplacian = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
        let ((eigenvalues, eigenvectors), solver_level) =
            solve_eigenproblem_pub(&laplacian, n_components_dim, 42);

        let residual = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
        let ortho = orthogonality_error(&eigenvectors);
        let (in_range, sorted) = check_eigenvalue_bounds(&eigenvalues, BOUNDS_TOL);
        let gap = spectral_gap(&eigenvalues);
        let cond = if eigenvalues.len() > 1 && eigenvalues[1] > 0.0 {
            eigenvalue_condition_number(&eigenvalues)
        } else {
            f64::NAN
        };
        let resid_thr = residual_threshold(solver_level);
        let resid_margin = tolerance_margin(resid_thr, residual);
        let ortho_margin = tolerance_margin(ORTHO_THRESHOLD, ortho);

        let metrics = vec![
            MetricResult {
                name: "max_eigenpair_residual".into(),
                dimension: 0,
                value: residual,
                threshold: resid_thr,
                passed: residual <= resid_thr,
            },
            MetricResult {
                name: "orthogonality_error".into(),
                dimension: 0,
                value: ortho,
                threshold: ORTHO_THRESHOLD,
                passed: ortho <= ORTHO_THRESHOLD,
            },
            MetricResult {
                name: "eigenvalue_bounds_in_range".into(),
                dimension: 0,
                value: in_range as u8 as f64,
                threshold: 1.0,
                passed: in_range,
            },
            MetricResult {
                name: "eigenvalue_bounds_sorted".into(),
                dimension: 0,
                value: sorted as u8 as f64,
                threshold: 1.0,
                passed: sorted,
            },
            MetricResult {
                name: "spectral_gap".into(),
                dimension: 0,
                value: gap,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "condition_number".into(),
                dimension: 0,
                value: cond,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "residual_margin_factor".into(),
                dimension: 0,
                value: resid_margin,
                threshold: 0.0,
                passed: true,
            },
            MetricResult {
                name: "ortho_margin_factor".into(),
                dimension: 0,
                value: ortho_margin,
                threshold: 0.0,
                passed: true,
            },
        ];

        all_reports.push(AssessmentReport {
            dataset: dataset.to_string(),
            n: *n,
            metrics,
        });
        solver_levels.push(Some(solver_level));
    }

    let experiment = ExperimentMetrics {
        generated_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        datasets: all_reports,
    };
    let dir = results_dir();
    std::fs::create_dir_all(&dir).expect("cannot create RESULTS_DIR");
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
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn assess_parity() {
    let run_set = datasets_to_run();
    let mut all_reports: Vec<AssessmentReport> = Vec::new();
    let mut solver_levels: Vec<Option<u8>> = Vec::new();

    for (dataset, n, _, _) in run_set.iter().filter(|(_, _, is_conn, _)| *is_conn) {
        let base = common::fixture_path(dataset, "");

        let laplacian = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
        let ref_eigenvalues: Array1<f64> =
            common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalues");
        let ref_eigenvectors: Array2<f64> =
            common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvectors");
        let ref_pre_noise: Array2<f32> =
            common::load_dense(&base.join("comp_f_scaling.npz"), "pre_noise");

        assert!(
            !ref_eigenvalues.is_empty(),
            "{dataset}: ref eigenvalues array is empty — fixture may be corrupt"
        );
        let n_components_dim = ref_eigenvalues.len() - 1;

        let ((eigenvalues, eigenvectors), solver_level) =
            solve_eigenproblem_pub(&laplacian, n_components_dim, 42);

        let abs_errors: Array1<f64> = eigenvalue_abs_errors(&eigenvalues, &ref_eigenvalues);
        let max_abs_err = abs_errors.iter().cloned().fold(0.0f64, f64::max);

        let selected = select_eigenvectors(
            eigenvalues.as_slice_memory_order().expect("eigenvalues contiguous"),
            &eigenvectors,
            n_components_dim,
        );
        let rust_pre_noise = compute_pre_noise(&selected);
        let sign_err = sign_agnostic_max_error(&rust_pre_noise, &ref_pre_noise);

        let evals_slice = ref_eigenvalues
            .as_slice_memory_order()
            .expect("ref eigenvalues contiguous");
        let has_degenerate = n_components_dim >= 2
            && evals_slice[1..].windows(2).any(|w| w[1] - w[0] < DEGENERATE_GAP_THRESHOLD);
        let gram = if has_degenerate {
            subspace_gram_det_kd(
                eigenvectors.slice(s![.., 1..]),
                ref_eigenvectors.slice(s![.., 1..]),
            )
        } else {
            f64::NAN
        };

        let eigenvalue_err_thr = residual_threshold(solver_level);

        let mut metrics = vec![
            MetricResult {
                name: "max_eigenvalue_abs_error".into(),
                dimension: 0,
                value: max_abs_err,
                threshold: eigenvalue_err_thr,
                passed: max_abs_err <= eigenvalue_err_thr,
            },
            MetricResult {
                name: "sign_agnostic_max_error".into(),
                dimension: 0,
                value: sign_err,
                threshold: SIGN_ERROR_THRESHOLD,
                passed: sign_err <= SIGN_ERROR_THRESHOLD,
            },
        ];

        if gram.is_finite() {
            metrics.push(MetricResult {
                name: "subspace_gram_det".into(),
                dimension: 0,
                value: gram,
                threshold: SUBSPACE_GRAM_DET_THRESHOLD,
                passed: gram >= SUBSPACE_GRAM_DET_THRESHOLD,
            });
        } else {
            metrics.push(MetricResult {
                name: "subspace_gram_det".into(),
                dimension: 0,
                value: 1.0,
                threshold: 0.0,
                passed: true,
            });
        }

        all_reports.push(AssessmentReport {
            dataset: dataset.to_string(),
            n: *n,
            metrics,
        });
        solver_levels.push(Some(solver_level));
    }

    let experiment = ExperimentMetrics {
        generated_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        datasets: all_reports,
    };
    let dir = results_dir();
    std::fs::create_dir_all(&dir).expect("cannot create RESULTS_DIR");
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
        .filter(|(_, _, is_conn, _)| *is_conn)
        .count();
    assert_eq!(
        json_parsed["datasets"]
            .as_array()
            .expect("json missing expected datasets array")
            .len(),
        connected_count
    );
}
