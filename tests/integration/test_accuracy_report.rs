#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2, ArrayView1};
use spectral_init::{
    build_normalized_laplacian, compute_degrees, find_components,
    select_eigenvectors, solve_eigenproblem_pub, spectral_init, SpectralInitConfig,
};
use std::path::Path;

/// (name, n, is_connected, expected_n_components)
const DATASETS: &[(&str, usize, bool, usize)] = &[
    ("blobs_50",              50,   false, 3),
    ("blobs_500",             500,  false, 4),
    ("blobs_5000",            5000, false, 2),
    ("blobs_connected_200",   200,  true,  1),
    ("blobs_connected_2000",  2000, true,  1),
    ("circles_300",           300,  true,  1),
    ("disconnected_200",      200,  false, 4),
    ("moons_200",             200,  true,  1),
    ("near_dupes_100",        100,  true,  1),
];

#[allow(dead_code)]
struct DatasetMetrics {
    dataset: String,
    n: usize,
    n_components: usize,
    is_connected: bool,
    solver_level: Option<u8>,
    solver_name: &'static str,
    eigenvalues_rust: Vec<f64>,
    eigenvalues_ref: Vec<f64>,
    eigenvalue_abs_errors: Vec<f64>,
    eigenvalue_rel_errors: Vec<f64>,
    per_eigenvec_residuals: Vec<f64>,
    max_residual: f64,
    subspace_gram_det: Option<f64>,
    pre_noise_max_err: f64,
    e2e_max_residual: Option<f64>,
    component_count_matches: bool,
}

struct ToleranceMarginEntry {
    component: String,
    tolerance: f64,
    tolerance_type: &'static str,
    worst_actual: f64,
    margin_factor: f64,
}

struct AccuracyReport {
    generated_at: String,
    datasets: Vec<DatasetMetrics>,
    tolerance_margins: Vec<ToleranceMarginEntry>,
}

fn gram_det_2d(
    v1: ArrayView1<f64>, v2: ArrayView1<f64>,
    r1: ArrayView1<f64>, r2: ArrayView1<f64>,
) -> f64 {
    let norm = |v: ArrayView1<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    let dot  = |a: ArrayView1<f64>, b: ArrayView1<f64>|
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let n1 = norm(v1); let n2 = norm(v2); let nr1 = norm(r1); let nr2 = norm(r2);
    let a = dot(r1, v1) / (nr1 * n1);
    let b = dot(r1, v2) / (nr1 * n2);
    let c = dot(r2, v1) / (nr2 * n1);
    let d = dot(r2, v2) / (nr2 * n2);
    (a * d - b * c).abs()
}

fn compute_rust_pre_noise(selected: &Array2<f64>) -> Array2<f32> {
    let max_abs = selected.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 {
        return Array2::<f32>::zeros(selected.raw_dim());
    }
    let expansion = 10.0 / max_abs;
    selected.mapv(|v| (v * expansion) as f32)
}

fn pre_noise_max_err(rust: &Array2<f32>, reference: &Array2<f32>) -> f64 {
    let mut worst = 0.0f64;
    for col in 0..rust.ncols() {
        let r = rust.column(col);
        let rf = reference.column(col);
        let err_pos: f64 = r.iter().zip(rf.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        let err_neg: f64 = r.iter().zip(rf.iter())
            .map(|(&a, &b)| (a as f64 + b as f64).abs())
            .fold(0.0f64, f64::max);
        worst = worst.max(err_pos.min(err_neg));
    }
    worst
}

fn solver_name(level: u8) -> &'static str {
    match level {
        0 => "Dense EVD",
        1 => "LOBPCG",
        2 => "LOBPCG+reg",
        3 => "rSVD",
        4 => "Forced Dense EVD",
        _ => "Unknown",
    }
}

fn process_dataset(dataset: &str, n: usize, is_connected: bool, expected_n_components: usize) -> DatasetMetrics {
    let base = common::fixture_path(dataset, "");

    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
    let laplacian = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));

    let ref_eigenvalues: Array1<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalues");
    let ref_eigenvectors: Array2<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvectors");

    let ref_pre_noise: Array2<f32> =
        common::load_dense(&base.join("comp_f_scaling.npz"), "pre_noise");
    let expansion_arr: ndarray::Array0<f64> =
        common::load_dense(&base.join("comp_f_scaling.npz"), "expansion");
    let expansion_val = expansion_arr.into_scalar();

    let k = ref_eigenvalues.len();
    let n_components_dim = k - 1;

    // NOTE: solve_eigenproblem_pub uses sqrt_deg = ones(n) internally.
    let ((eigenvalues, eigenvectors), solver_level) =
        solve_eigenproblem_pub(&laplacian, n_components_dim, 42);

    let eigenvalue_abs_errors: Vec<f64> = eigenvalues.iter()
        .zip(ref_eigenvalues.iter())
        .map(|(r, re)| (r - re).abs())
        .collect();
    let eigenvalue_rel_errors: Vec<f64> = eigenvalues.iter()
        .zip(ref_eigenvalues.iter())
        .map(|(r, re)| (r - re).abs() / re.abs().max(1e-300))
        .collect();

    let per_eigenvec_residuals: Vec<f64> = (0..eigenvectors.ncols())
        .map(|j| common::residual_spmv(&laplacian, eigenvectors.column(j), eigenvalues[j]))
        .collect();
    let max_residual = per_eigenvec_residuals.iter().cloned().fold(0.0f64, f64::max);

    let subspace_gram_det = if k >= 3 && ref_eigenvectors.ncols() >= 3 {
        Some(gram_det_2d(
            eigenvectors.column(1), eigenvectors.column(2),
            ref_eigenvectors.column(1), ref_eigenvectors.column(2),
        ))
    } else {
        None
    };

    let selected = select_eigenvectors(
        eigenvalues.as_slice_memory_order().expect("eigenvalues contiguous"),
        &eigenvectors,
        n_components_dim,
    );
    let rust_pre_noise = compute_rust_pre_noise(&selected);
    let pn_max_err = pre_noise_max_err(&rust_pre_noise, &ref_pre_noise);

    let e2e_max_residual = if is_connected {
        let result = spectral_init(&graph, n_components_dim, 42, None, SpectralInitConfig::default())
            .unwrap_or_else(|e| panic!("{dataset}: spectral_init failed: {e}"));
        let e2e_max = (0..n_components_dim).map(|col| {
            let evec: Array1<f64> = result.column(col).mapv(|v| v as f64 / expansion_val);
            common::residual_spmv(&laplacian, evec.view(), ref_eigenvalues[col + 1])
        }).fold(0.0f64, f64::max);
        Some(e2e_max)
    } else {
        None
    };

    let (_, rust_n_components) = find_components(&graph);
    let component_count_matches = rust_n_components == expected_n_components;

    DatasetMetrics {
        dataset: dataset.to_string(),
        n,
        n_components: expected_n_components,
        is_connected,
        solver_level: Some(solver_level),
        solver_name: solver_name(solver_level),
        eigenvalues_rust: eigenvalues.to_vec(),
        eigenvalues_ref: ref_eigenvalues.to_vec(),
        eigenvalue_abs_errors,
        eigenvalue_rel_errors,
        per_eigenvec_residuals,
        max_residual,
        subspace_gram_det,
        pre_noise_max_err: pn_max_err,
        e2e_max_residual,
        component_count_matches,
    }
}

fn build_tolerance_margin_table(all_metrics: &[DatasetMetrics]) -> Vec<ToleranceMarginEntry> {
    let mut margins = Vec::new();

    // comp_a: max relative degree error
    let mut worst_deg_rel = 0.0f64;
    for metric in all_metrics {
        let base = common::fixture_path(&metric.dataset, "");
        let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
        let (degrees, _) = compute_degrees(&graph);
        let ref_deg: Array1<f64> = common::load_dense(&base.join("comp_a_degrees.npz"), "degrees");
        for (&got, &want) in degrees.iter().zip(ref_deg.iter()) {
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            worst_deg_rel = worst_deg_rel.max(rel_err);
        }
    }
    margins.push(ToleranceMarginEntry {
        component: "comp_a degrees (relative)".to_string(),
        tolerance: 3e-7,
        tolerance_type: "relative",
        worst_actual: worst_deg_rel,
        margin_factor: if worst_deg_rel == 0.0 { f64::INFINITY } else { 3e-7 / worst_deg_rel },
    });

    // comp_b: max absolute Laplacian entry error
    let mut worst_lap_abs = 0.0f64;
    for metric in all_metrics {
        let base = common::fixture_path(&metric.dataset, "");
        let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
        let (_, sqrt_deg) = compute_degrees(&graph);
        let inv_sqrt_deg: Vec<f64> = sqrt_deg.iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        let rust_lap = build_normalized_laplacian(&graph, &inv_sqrt_deg);
        let ref_lap = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
        for (val, (row, col)) in ref_lap.iter() {
            let rust_val = rust_lap.get(row, col).copied().unwrap_or(0.0);
            worst_lap_abs = worst_lap_abs.max((rust_val - val).abs());
        }
    }
    margins.push(ToleranceMarginEntry {
        component: "comp_b Laplacian entries (absolute)".to_string(),
        tolerance: 1e-14,
        tolerance_type: "absolute",
        worst_actual: worst_lap_abs,
        margin_factor: if worst_lap_abs == 0.0 { f64::INFINITY } else { 1e-14 / worst_lap_abs },
    });

    // comp_d: eigenvalue errors — split by n (dense EVD: n<2000, LOBPCG: n>=2000)
    let worst_eig_dense = all_metrics.iter()
        .filter(|m| m.n < 2000)
        .flat_map(|m| m.eigenvalue_abs_errors.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    margins.push(ToleranceMarginEntry {
        component: "comp_d eigenvalues, Dense EVD (n<2000)".to_string(),
        tolerance: 1e-8,
        tolerance_type: "absolute",
        worst_actual: worst_eig_dense,
        margin_factor: if worst_eig_dense == 0.0 { f64::INFINITY } else { 1e-8 / worst_eig_dense },
    });

    let worst_eig_lobpcg = all_metrics.iter()
        .filter(|m| m.n >= 2000)
        .flat_map(|m| m.eigenvalue_abs_errors.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    margins.push(ToleranceMarginEntry {
        component: "comp_d eigenvalues, LOBPCG (n>=2000)".to_string(),
        tolerance: 1e-6,
        tolerance_type: "absolute",
        worst_actual: worst_eig_lobpcg,
        margin_factor: if worst_eig_lobpcg == 0.0 { f64::INFINITY } else { 1e-6 / worst_eig_lobpcg },
    });

    // comp_d: residuals — split by n
    let worst_res_dense = all_metrics.iter()
        .filter(|m| m.n < 2000)
        .map(|m| m.max_residual)
        .fold(0.0f64, f64::max);
    margins.push(ToleranceMarginEntry {
        component: "comp_d residuals, Dense EVD (n<2000)".to_string(),
        tolerance: 1e-10,
        tolerance_type: "absolute",
        worst_actual: worst_res_dense,
        margin_factor: if worst_res_dense == 0.0 { f64::INFINITY } else { 1e-10 / worst_res_dense },
    });

    let worst_res_lobpcg = all_metrics.iter()
        .filter(|m| m.n >= 2000)
        .map(|m| m.max_residual)
        .fold(0.0f64, f64::max);
    margins.push(ToleranceMarginEntry {
        component: "comp_d residuals, LOBPCG/rSVD (n>=2000)".to_string(),
        tolerance: 1e-4,
        tolerance_type: "absolute",
        worst_actual: worst_res_lobpcg,
        margin_factor: if worst_res_lobpcg == 0.0 { f64::INFINITY } else { 1e-4 / worst_res_lobpcg },
    });

    // E2E residuals (connected datasets only)
    let worst_e2e = all_metrics.iter()
        .filter_map(|m| m.e2e_max_residual)
        .fold(0.0f64, f64::max);
    margins.push(ToleranceMarginEntry {
        component: "E2E residuals (connected, post-noise)".to_string(),
        tolerance: 5e-3,
        tolerance_type: "absolute",
        worst_actual: worst_e2e,
        margin_factor: if worst_e2e == 0.0 { f64::INFINITY } else { 5e-3 / worst_e2e },
    });

    // comp_f pre_noise scaling
    let worst_pn = all_metrics.iter()
        .map(|m| m.pre_noise_max_err)
        .fold(0.0f64, f64::max);
    let f32_eps = f32::EPSILON as f64;
    margins.push(ToleranceMarginEntry {
        component: "comp_f pre_noise scaling".to_string(),
        tolerance: f32_eps,
        tolerance_type: "absolute",
        worst_actual: worst_pn,
        margin_factor: if worst_pn == 0.0 { f64::INFINITY } else { f32_eps / worst_pn },
    });

    margins
}

fn to_json(report: &AccuracyReport) -> String {
    use serde_json::json;

    let datasets: Vec<serde_json::Value> = report.datasets.iter().map(|m| {
        let lambda_3_abs_err = m.eigenvalue_abs_errors.get(2).copied();
        let lambda_3_rel_err = m.eigenvalue_rel_errors.get(2).copied();
        json!({
            "dataset": m.dataset,
            "n": m.n,
            "n_components": m.n_components,
            "is_connected": m.is_connected,
            "solver_level": m.solver_level,
            "solver_name": m.solver_name,
            "lambda_2_abs_err": m.eigenvalue_abs_errors.get(1).copied(),
            "lambda_2_rel_err": m.eigenvalue_rel_errors.get(1).copied(),
            "lambda_3_abs_err": lambda_3_abs_err,
            "lambda_3_rel_err": lambda_3_rel_err,
            "per_eigenvec_residuals": m.per_eigenvec_residuals,
            "max_residual": m.max_residual,
            "subspace_gram_det": m.subspace_gram_det,
            "pre_noise_max_err": m.pre_noise_max_err,
            "e2e_max_residual": m.e2e_max_residual,
            "component_count_matches": m.component_count_matches,
        })
    }).collect();

    let margins: Vec<serde_json::Value> = report.tolerance_margins.iter().map(|m| {
        let margin_factor = if m.margin_factor.is_infinite() || m.margin_factor.is_nan() {
            json!(null)
        } else {
            json!(m.margin_factor)
        };
        json!({
            "component": m.component,
            "tolerance": m.tolerance,
            "tolerance_type": m.tolerance_type,
            "worst_actual": m.worst_actual,
            "margin_factor": margin_factor,
        })
    }).collect();

    let report_val = json!({
        "generated_at": report.generated_at,
        "datasets": datasets,
        "tolerance_margins": margins,
    });

    serde_json::to_string_pretty(&report_val).expect("JSON serialization failed")
}

fn to_markdown(report: &AccuracyReport) -> String {
    let mut out = String::new();

    out.push_str("# Spectral Init Accuracy Report\n");
    out.push_str(&format!("Generated: {}\n\n", report.generated_at));

    out.push_str("## Per-Dataset Summary\n\n");
    out.push_str("| Dataset | n | Components | Solver | λ₂ abs err | λ₃ abs err | max residual | subspace det | pre_noise max_err | E2E residual |\n");
    out.push_str("|---------|---|------------|--------|------------|------------|--------------|--------------|-------------------|--------------|\n");

    for m in &report.datasets {
        let lambda2 = m.eigenvalue_abs_errors.get(1)
            .map(|v| format!("{:.2e}", v))
            .unwrap_or_else(|| "—".to_string());
        let lambda3 = m.eigenvalue_abs_errors.get(2)
            .map(|v| format!("{:.2e}", v))
            .unwrap_or_else(|| "—".to_string());
        let subspace = m.subspace_gram_det
            .map(|v| format!("{:.6}", v))
            .unwrap_or_else(|| "—".to_string());
        let pn_err = if m.pre_noise_max_err == 0.0 {
            "(exact)".to_string()
        } else {
            format!("{:.2e}", m.pre_noise_max_err)
        };
        let e2e = if m.is_connected {
            m.e2e_max_residual
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_else(|| "—".to_string())
        } else {
            "N/A".to_string()
        };

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.2e} | {} | {} | {} |\n",
            m.dataset, m.n, m.n_components, m.solver_name,
            lambda2, lambda3, m.max_residual, subspace, pn_err, e2e,
        ));
    }

    out.push_str("\n## Tolerance Margin Analysis\n\n");
    out.push_str("| Component | Tolerance | Worst Actual | Margin Factor |\n");
    out.push_str("|-----------|-----------|--------------|---------------|\n");

    for m in &report.tolerance_margins {
        let margin_str = if m.margin_factor.is_infinite() {
            "∞x".to_string()
        } else {
            format!("{:.2}x", m.margin_factor)
        };
        out.push_str(&format!(
            "| {} | {:.2e} | {:.2e} | {} |\n",
            m.component, m.tolerance, m.worst_actual, margin_str,
        ));
    }

    out
}

#[test]
fn generate_accuracy_report() {
    let mut all_metrics: Vec<DatasetMetrics> = Vec::new();
    for &(dataset, n, is_connected, expected_n_components) in DATASETS {
        let metrics = process_dataset(dataset, n, is_connected, expected_n_components);
        all_metrics.push(metrics);
    }

    let margins = build_tolerance_margin_table(&all_metrics);

    let report = AccuracyReport {
        generated_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        datasets: all_metrics,
        tolerance_margins: margins,
    };

    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    std::fs::create_dir_all(&target_dir).expect("cannot create target/");
    let md_path   = target_dir.join("accuracy-report.md");
    let json_path = target_dir.join("accuracy-report.json");
    let md_text   = to_markdown(&report);
    let json_text = to_json(&report);
    std::fs::write(&md_path,   &md_text).expect("cannot write accuracy-report.md");
    std::fs::write(&json_path, &json_text).expect("cannot write accuracy-report.json");
    println!("Accuracy report written to:\n  {}\n  {}", md_path.display(), json_path.display());

    assert!(md_path.exists());
    assert!(json_path.exists());

    let json: serde_json::Value = serde_json::from_str(&json_text).expect("invalid JSON");
    assert_eq!(json["datasets"].as_array().unwrap().len(), 9,
        "Report must cover all 9 datasets");
    for entry in json["datasets"].as_array().unwrap() {
        let max_r = entry["max_residual"].as_f64()
            .expect("max_residual must be a number");
        assert!(max_r.is_finite(), "max_residual must be finite for {}", entry["dataset"]);
        assert!(
            entry["component_count_matches"].as_bool().unwrap_or(false),
            "component count mismatch for {}", entry["dataset"]
        );
    }
    assert!(!json["tolerance_margins"].as_array().unwrap().is_empty());
}
