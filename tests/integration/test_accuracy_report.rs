#[path = "../common/mod.rs"]
mod common;

use ndarray::{Array1, Array2, ArrayView1};
use spectral_init::{
    build_normalized_laplacian, compute_degrees, embed_disconnected, find_components,
    select_eigenvectors, solve_eigenproblem_pub, spectral_init, ComputeMode, SpectralInitConfig,
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
    disconnected_path: Option<DisconnectedPathMetrics>,
}

struct DisconnectedPathMetrics {
    n_components: usize,
    per_comp_max_residual: Vec<f64>,
    separation_ratio: f64,
    e2e_output_is_finite: bool,
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

fn extract_subgraph_local(
    graph: &sprs::CsMatI<f32, u32, usize>,
    node_indices: &[usize],
) -> sprs::CsMatI<f32, u32, usize> {
    let n_global = graph.rows();
    let n_comp = node_indices.len();
    let mut lookup = vec![usize::MAX; n_global];
    for (local, &orig) in node_indices.iter().enumerate() {
        lookup[orig] = local;
    }
    let mut indptr = vec![0usize; n_comp + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for (local_row, &orig_row) in node_indices.iter().enumerate() {
        if let Some(row_vec) = graph.outer_view(orig_row) {
            for (orig_col, &weight) in row_vec.iter() {
                if lookup[orig_col] != usize::MAX {
                    indices.push(
                        u32::try_from(lookup[orig_col])
                            .expect("local index overflows u32"),
                    );
                    data.push(weight);
                }
            }
        }
        indptr[local_row + 1] = indices.len();
    }
    sprs::CsMatI::<f32, u32, usize>::new((n_comp, n_comp), indptr, indices, data)
}

fn process_disconnected_path(dataset: &str, n_embedding_dims: usize) -> DisconnectedPathMetrics {
    let base = common::fixture_path(dataset, "");

    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
    let raw_data: ndarray::Array2<f32> = {
        let f64_data: ndarray::Array2<f64> =
            common::load_dense(&base.join("step0_raw_data.npz"), "X");
        f64_data.mapv(|x| x as f32)
    };

    let (labels, n_conn_components) = find_components(&graph);

    let mut component_members: Vec<Vec<usize>> = vec![Vec::new(); n_conn_components];
    for (node, &label) in labels.iter().enumerate() {
        component_members[label].push(node);
    }

    let embedding = embed_disconnected(
        &graph,
        &labels,
        n_conn_components,
        n_embedding_dims,
        42,
        Some(raw_data.view()),
        ComputeMode::PythonCompat,
    )
    .unwrap_or_else(|e| panic!("{dataset}: embed_disconnected failed: {e}"));

    let per_comp_max_residual: Vec<f64> = component_members
        .iter()
        .map(|members| {
            let n_c = members.len();
            if n_c < 2 {
                return 0.0;
            }
            let sub_graph = extract_subgraph_local(&graph, members);
            let (_, sqrt_deg_c) = compute_degrees(&sub_graph);
            let inv_sqrt_c: Vec<f64> = sqrt_deg_c
                .iter()
                .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
                .collect();
            let lap_c = build_normalized_laplacian(&sub_graph, &inv_sqrt_c);

            let mut coords_c = ndarray::Array2::<f64>::zeros((n_c, n_embedding_dims));
            for (local_i, &orig_i) in members.iter().enumerate() {
                for d in 0..n_embedding_dims {
                    coords_c[[local_i, d]] = embedding[[orig_i, d]];
                }
            }

            for d in 0..n_embedding_dims {
                let mean = coords_c.column(d).sum() / n_c as f64;
                for i in 0..n_c {
                    coords_c[[i, d]] -= mean;
                }
            }

            (0..n_embedding_dims)
                .map(|d| {
                    let v = coords_c.column(d).to_owned();
                    let v_sq: f64 = v.iter().map(|&x| x * x).sum();
                    if v_sq < 1e-30 {
                        return 0.0;
                    }
                    let mut lv = vec![0.0f64; n_c];
                    for (val, (row, col)) in lap_c.iter() {
                        lv[row] += val * v[col];
                    }
                    let lambda = lv.iter().zip(v.iter()).map(|(&li, &vi)| li * vi).sum::<f64>()
                        / v_sq;
                    common::residual_spmv(&lap_c, v.view(), lambda)
                })
                .fold(0.0f64, f64::max)
        })
        .collect();

    let centroids: Vec<Vec<f64>> = component_members
        .iter()
        .map(|members| {
            let n_c = members.len() as f64;
            let mut c = vec![0.0f64; n_embedding_dims];
            for &orig_i in members {
                for d in 0..n_embedding_dims {
                    c[d] += embedding[[orig_i, d]];
                }
            }
            c.iter_mut().for_each(|x| *x /= n_c);
            c
        })
        .collect();

    let min_inter = (0..n_conn_components)
        .flat_map(|i| ((i + 1)..n_conn_components).map(move |j| (i, j)))
        .map(|(i, j)| {
            (0..n_embedding_dims)
                .map(|d| (centroids[i][d] - centroids[j][d]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min);

    let max_intra = component_members
        .iter()
        .enumerate()
        .map(|(c_idx, members)| {
            members
                .iter()
                .map(|&orig_i| {
                    (0..n_embedding_dims)
                        .map(|d| (embedding[[orig_i, d]] - centroids[c_idx][d]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);

    let separation_ratio = if max_intra > 0.0 {
        min_inter / max_intra
    } else {
        f64::INFINITY
    };

    let e2e_result =
        spectral_init(&graph, n_embedding_dims, 42, Some(raw_data.view()), SpectralInitConfig::default())
            .unwrap_or_else(|e| panic!("{dataset}: spectral_init failed: {e}"));
    let e2e_output_is_finite = e2e_result.iter().all(|&v: &f32| v.is_finite());

    DisconnectedPathMetrics {
        n_components: n_conn_components,
        per_comp_max_residual,
        separation_ratio,
        e2e_output_is_finite,
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

    if !is_connected {
        let (_, rust_n_components) = find_components(&graph);
        let component_count_matches = rust_n_components == expected_n_components;
        let disconnected_path = Some(process_disconnected_path(dataset, n_components_dim));
        return DatasetMetrics {
            dataset: dataset.to_string(),
            n,
            n_components: expected_n_components,
            is_connected,
            solver_level: None,
            solver_name: "N/A (disconnected — use disconnected_path metrics)",
            eigenvalues_rust: vec![],
            eigenvalues_ref: ref_eigenvalues.to_vec(),
            eigenvalue_abs_errors: vec![],
            eigenvalue_rel_errors: vec![],
            per_eigenvec_residuals: vec![],
            max_residual: 0.0,
            subspace_gram_det: None,
            pre_noise_max_err: 0.0,
            e2e_max_residual: None,
            component_count_matches,
            disconnected_path,
        };
    }

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
        disconnected_path: None,
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

    // comp_b isolated: Python sqrt_deg (from comp_a_degrees.npz) → Rust Laplacian
    // This mirrors test_comp_b_laplacian.rs; tolerance 1e-14 is valid here.
    let mut worst_lap_iso = 0.0f64;
    for metric in all_metrics {
        let base = common::fixture_path(&metric.dataset, "");
        let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
        let sqrt_deg: Array1<f64> =
            common::load_dense(&base.join("comp_a_degrees.npz"), "sqrt_deg");
        let inv_sqrt_deg: Vec<f64> = sqrt_deg
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        let rust_lap = build_normalized_laplacian(&graph, &inv_sqrt_deg);
        let ref_lap = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
        for (val, (row, col)) in ref_lap.iter() {
            let rust_val = rust_lap.get(row, col).copied().unwrap_or(0.0);
            worst_lap_iso = worst_lap_iso.max((rust_val - val).abs());
        }
    }
    margins.push(ToleranceMarginEntry {
        component: "comp_b Laplacian (isolated: Python degrees)".to_string(),
        tolerance: 1e-14,
        tolerance_type: "absolute",
        worst_actual: worst_lap_iso,
        margin_factor: if worst_lap_iso == 0.0 { f64::INFINITY } else { 1e-14 / worst_lap_iso },
    });

    // comp_b chained: Rust compute_degrees() → Rust Laplacian
    // Inherits f32 graph weight precision; tolerance 1e-7 is appropriate.
    let mut worst_lap_chain = 0.0f64;
    for metric in all_metrics {
        let base = common::fixture_path(&metric.dataset, "");
        let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
        let (_, sqrt_deg) = compute_degrees(&graph);
        let inv_sqrt_deg: Vec<f64> = sqrt_deg
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        let rust_lap = build_normalized_laplacian(&graph, &inv_sqrt_deg);
        let ref_lap = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
        for (val, (row, col)) in ref_lap.iter() {
            let rust_val = rust_lap.get(row, col).copied().unwrap_or(0.0);
            worst_lap_chain = worst_lap_chain.max((rust_val - val).abs());
        }
    }
    margins.push(ToleranceMarginEntry {
        component: "comp_b Laplacian (chained: Rust degrees)".to_string(),
        tolerance: 1e-7,
        tolerance_type: "absolute",
        worst_actual: worst_lap_chain,
        margin_factor: if worst_lap_chain == 0.0 { f64::INFINITY } else { 1e-7 / worst_lap_chain },
    });

    // comp_d: eigenvalue errors — split by n (dense EVD: n<2000, LOBPCG: n>=2000)
    let worst_eig_dense = all_metrics.iter()
        .filter(|m| m.is_connected)
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
        .filter(|m| m.is_connected)
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
        .filter(|m| m.is_connected)
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
        .filter(|m| m.is_connected)
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
            "disconnected_path": m.disconnected_path.as_ref().map(|dp| json!({
                "n_components": dp.n_components,
                "per_comp_max_residual": dp.per_comp_max_residual,
                "separation_ratio": dp.separation_ratio,
                "e2e_output_is_finite": dp.e2e_output_is_finite,
            })),
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
        let (lambda2, lambda3, subspace, pn_err, max_res) = if m.is_connected {
            let l2 = m.eigenvalue_abs_errors.get(1)
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_else(|| "—".to_string());
            let l3 = m.eigenvalue_abs_errors.get(2)
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_else(|| "—".to_string());
            let sub = m.subspace_gram_det
                .map(|v| format!("{:.6}", v))
                .unwrap_or_else(|| "—".to_string());
            let pn = if m.pre_noise_max_err == 0.0 {
                "(exact)".to_string()
            } else {
                format!("{:.2e}", m.pre_noise_max_err)
            };
            let mr = format!("{:.2e}", m.max_residual);
            (l2, l3, sub, pn, mr)
        } else {
            ("—".to_string(), "—".to_string(), "—".to_string(), "—".to_string(), "—".to_string())
        };
        let e2e = if m.is_connected {
            m.e2e_max_residual
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_else(|| "—".to_string())
        } else {
            "N/A".to_string()
        };

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            m.dataset, m.n, m.n_components, m.solver_name,
            lambda2, lambda3, max_res, subspace, pn_err, e2e,
        ));
    }

    out.push_str("\n## Disconnected Dataset Production-Path Metrics\n\n");
    out.push_str("| Dataset | Components | Max Comp Residual | Separation Ratio | E2E Finite |\n");
    out.push_str("|---------|------------|-------------------|------------------|------------|\n");
    for m in report.datasets.iter().filter(|m| !m.is_connected) {
        let dp = m.disconnected_path.as_ref().unwrap();
        let max_comp_res = dp.per_comp_max_residual.iter().cloned().fold(0.0f64, f64::max);
        out.push_str(&format!(
            "| {} | {} | {:.2e} | {:.4} | {} |\n",
            m.dataset, dp.n_components, max_comp_res, dp.separation_ratio, dp.e2e_output_is_finite,
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

    // REQ-SPLIT-001: two separate comp_b rows must exist
    let margins_arr = json["tolerance_margins"].as_array().unwrap();

    let isolated_row = margins_arr.iter().find(|e| {
        e["component"].as_str().unwrap_or("").contains("isolated: Python degrees")
    });
    assert!(isolated_row.is_some(),
        "Expected a comp_b isolated (Python degrees) row in tolerance_margins");

    let chained_row = margins_arr.iter().find(|e| {
        e["component"].as_str().unwrap_or("").contains("chained: Rust degrees")
    });
    assert!(chained_row.is_some(),
        "Expected a comp_b chained (Rust degrees) row in tolerance_margins");

    // REQ-SPLIT-002: isolated row tolerance must be 1e-14
    let iso_tol = isolated_row.unwrap()["tolerance"].as_f64().unwrap();
    assert!(
        (iso_tol - 1e-14).abs() < 1e-20,
        "Isolated comp_b tolerance must be 1e-14, got {iso_tol}"
    );

    // REQ-SPLIT-003: chained row tolerance must be 1e-7
    let chain_tol = chained_row.unwrap()["tolerance"].as_f64().unwrap();
    assert!(
        (chain_tol - 1e-7).abs() < 1e-13,
        "Chained comp_b tolerance must be 1e-7, got {chain_tol}"
    );

    // REQ-SPLIT-004: the old conflated row must no longer exist
    let old_row = margins_arr.iter().find(|e| {
        e["component"].as_str().unwrap_or("") == "comp_b Laplacian entries (absolute)"
    });
    assert!(old_row.is_none(),
        "Old conflated comp_b row must be removed from tolerance_margins");

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

    for entry in json["datasets"].as_array().unwrap() {
        if !entry["is_connected"].as_bool().unwrap_or(true) {
            let dp = entry["disconnected_path"]
                .as_object()
                .unwrap_or_else(|| panic!(
                    "disconnected_path must be present for {}",
                    entry["dataset"]
                ));
            assert!(
                dp["e2e_output_is_finite"].as_bool().unwrap_or(false),
                "e2e_output_is_finite must be true for {}",
                entry["dataset"]
            );
            let sep_ratio = dp["separation_ratio"]
                .as_f64()
                .expect("separation_ratio must be a number");
            assert!(
                sep_ratio > 1.0,
                "separation_ratio must be > 1.0 for {}, got {sep_ratio}",
                entry["dataset"]
            );
            for residual in dp["per_comp_max_residual"].as_array().unwrap() {
                let r = residual.as_f64().expect("per_comp_max_residual must be a number");
                assert!(
                    r.is_finite(),
                    "per_comp_max_residual must be finite for {}",
                    entry["dataset"]
                );
            }
        }
    }

    for entry in json["datasets"].as_array().unwrap() {
        if !entry["is_connected"].as_bool().unwrap_or(true) {
            assert!(
                entry["subspace_gram_det"].is_null(),
                "subspace_gram_det must be null for disconnected dataset {}",
                entry["dataset"]
            );
        }
    }
}
