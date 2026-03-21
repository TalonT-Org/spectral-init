#[path = "../common/mod.rs"]
mod common;

use spectral_init::spectral_init;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Reverse the scale_and_add_noise scaling step.
/// Returns an (n, 2) f64 array: output_f32 / expansion.
fn descale_embedding(output: &ndarray::Array2<f32>, expansion: f64) -> ndarray::Array2<f64> {
    output.mapv(|v| v as f64 / expansion)
}

fn vec_norm(v: ndarray::ArrayView1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_dot(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

/// Load graph, run spectral_init, check residuals for a connected dataset.
fn run_e2e_residual_check(dataset: &str, n: usize) {
    let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);

    let result = spectral_init(&graph, 2, 42, None)
        .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));
    assert_eq!(result.shape(), &[n, 2]);
    for &v in result.iter() {
        assert!(v.is_finite(), "non-finite in {dataset}: {v}");
    }

    // Load expansion factor (stored as 0-D scalar in the fixture)
    let scaling_path = common::fixture_path(dataset, "comp_f_scaling.npz");
    let expansion: ndarray::Array0<f64> = common::load_dense(&scaling_path, "expansion");
    let expansion_val = expansion.into_scalar();
    assert!(
        expansion_val.is_normal() && expansion_val > 0.0,
        "fixture expansion value must be positive and normal, got {expansion_val}"
    );

    // Load Laplacian and reference eigenvalues
    let laplacian =
        common::load_sparse_csr(&common::fixture_path(dataset, "comp_b_laplacian.npz"));
    let ref_eigenvalues: ndarray::Array1<f64> = common::load_dense(
        &common::fixture_path(dataset, "comp_d_eigensolver.npz"),
        "eigenvalues",
    );

    // De-scale to recover approximate eigenvectors
    let descaled = descale_embedding(&result, expansion_val);

    // Check residual for each of the 2 selected eigenvectors.
    // Output column 0 → eigenvalue[1], column 1 → eigenvalue[2] (skip trivial index 0).
    for col in 0..2usize {
        let evec = descaled.column(col);
        let eigval = ref_eigenvalues[col + 1];
        let r = common::residual_spmv(&laplacian, evec, eigval);
        // Tolerance is 0.005: this bound must hold across the full solver escalation
        // chain (dense EVD → LOBPCG → randomized SVD). Dense EVD residuals are ~1e-8
        // pre-noise; LOBPCG and rSVD paths yield somewhat larger residuals but remain
        // well below this threshold. Gaussian noise (sigma=0.0001) adds at most ~1e-4
        // after de-scaling by expansion (~10); f32→f64 de-scaling contributes ~1e-7.
        // 0.005 is 50× above the expected noise floor, catching genuine solver
        // failures while rejecting the misleadingly loose 0.05 bound.
        let residual_tol = 0.005_f64;
        assert!(
            r < residual_tol,
            "residual quality check failed for {dataset} column {col}: residual={r:.6e} >= {residual_tol}"
        );
    }
}

/// Load exact-KNN graph, run spectral_init, check shape and finiteness only.
/// No residual check: comp_b_laplacian.npz is built from the approx-KNN graph;
/// the exact-KNN graph may have different neighbor sets → different Laplacian →
/// cross-path residual failures even for correct output.
fn run_e2e_exact_knn_check(dataset: &str, n: usize) {
    let graph_path = common::fixture_path(dataset, "step5a_pruned_exact.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);
    let result = spectral_init(&graph, 2, 42, None)
        .unwrap_or_else(|e| panic!("exact-KNN spectral_init failed for {dataset}: {e}"));
    assert_eq!(result.shape(), &[n, 2]);
    for &v in result.iter() {
        assert!(v.is_finite(), "non-finite in exact-KNN {dataset}: {v}");
    }
}

// ── residual quality tests (7 connected datasets) ────────────────────────────
// blobs_500 and blobs_5000 are excluded: fixture generation confirms they are
// disconnected (4 and 2 components respectively). Their embeddings are produced
// by embed_disconnected() and are not Laplacian eigenvectors of the full graph.

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_residual_quality_blobs_connected_200() {
    run_e2e_residual_check("blobs_connected_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_residual_quality_blobs_connected_2000() {
    run_e2e_residual_check("blobs_connected_2000", 2000);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_residual_quality_moons_200() {
    run_e2e_residual_check("moons_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_residual_quality_circles_300() {
    run_e2e_residual_check("circles_300", 300);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_residual_quality_near_dupes_100() {
    run_e2e_residual_check("near_dupes_100", 100);
}

// ── subspace comparison tests (near-degenerate eigenvalues) ──────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_subspace_circles_300() {
    let dataset = "circles_300";

    let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);
    let result = spectral_init(&graph, 2, 42, None)
        .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));

    let scaling_path = common::fixture_path(dataset, "comp_f_scaling.npz");
    let expansion: ndarray::Array0<f64> = common::load_dense(&scaling_path, "expansion");
    let expansion_val = expansion.into_scalar();
    assert!(
        expansion_val.is_normal() && expansion_val > 0.0,
        "fixture expansion value must be positive and normal, got {expansion_val}"
    );
    let descaled = descale_embedding(&result, expansion_val);

    // Reference eigenvectors: columns 1 and 2 (skip trivial column 0)
    let ref_evecs: ndarray::Array2<f64> = common::load_dense(
        &common::fixture_path(dataset, "comp_d_eigensolver.npz"),
        "eigenvectors",
    );

    // Compute 2×2 Gram matrix between reference and Rust-computed eigenvector pair:
    // G[i][j] = dot(ref_col[i], rust_col[j])
    // det(G) = a*d - b*c; det > 0.95 means the subspaces are nearly identical.
    let ref_v1 = ref_evecs.column(1);
    let ref_v2 = ref_evecs.column(2);
    let rust_v1 = descaled.column(0);
    let rust_v2 = descaled.column(1);

    let n_ref_v1 = vec_norm(ref_v1);
    let n_ref_v2 = vec_norm(ref_v2);
    let n_rust_v1 = vec_norm(rust_v1);
    let n_rust_v2 = vec_norm(rust_v2);

    let a = vec_dot(ref_v1, rust_v1) / (n_ref_v1 * n_rust_v1);
    let b = vec_dot(ref_v1, rust_v2) / (n_ref_v1 * n_rust_v2);
    let c = vec_dot(ref_v2, rust_v1) / (n_ref_v2 * n_rust_v1);
    let d = vec_dot(ref_v2, rust_v2) / (n_ref_v2 * n_rust_v2);
    let gram_det = (a * d - b * c).abs();

    assert!(
        gram_det > 0.95,
        "{dataset}: Gram determinant {gram_det:.6} <= 0.95; eigenvector subspaces do not match"
    );
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_subspace_near_dupes_100() {
    let dataset = "near_dupes_100";

    let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);
    let result = spectral_init(&graph, 2, 42, None)
        .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));

    let scaling_path = common::fixture_path(dataset, "comp_f_scaling.npz");
    let expansion: ndarray::Array0<f64> = common::load_dense(&scaling_path, "expansion");
    let expansion_val = expansion.into_scalar();
    assert!(
        expansion_val.is_normal() && expansion_val > 0.0,
        "fixture expansion value must be positive and normal, got {expansion_val}"
    );
    let descaled = descale_embedding(&result, expansion_val);

    let ref_evecs: ndarray::Array2<f64> = common::load_dense(
        &common::fixture_path(dataset, "comp_d_eigensolver.npz"),
        "eigenvectors",
    );

    let ref_v1 = ref_evecs.column(1);
    let ref_v2 = ref_evecs.column(2);
    let rust_v1 = descaled.column(0);
    let rust_v2 = descaled.column(1);

    let a = vec_dot(ref_v1, rust_v1) / (vec_norm(ref_v1) * vec_norm(rust_v1));
    let b = vec_dot(ref_v1, rust_v2) / (vec_norm(ref_v1) * vec_norm(rust_v2));
    let c = vec_dot(ref_v2, rust_v1) / (vec_norm(ref_v2) * vec_norm(rust_v1));
    let d = vec_dot(ref_v2, rust_v2) / (vec_norm(ref_v2) * vec_norm(rust_v2));
    let gram_det = (a * d - b * c).abs();

    assert!(
        gram_det > 0.95,
        "{dataset}: Gram determinant {gram_det:.6} <= 0.95; eigenvector subspaces do not match"
    );
}

// ── exact-KNN alternative path tests (5 small datasets) ──────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_exact_knn_blobs_50() {
    run_e2e_exact_knn_check("blobs_50", 50);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_exact_knn_moons_200() {
    run_e2e_exact_knn_check("moons_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_exact_knn_circles_300() {
    run_e2e_exact_knn_check("circles_300", 300);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_exact_knn_blobs_connected_200() {
    run_e2e_exact_knn_check("blobs_connected_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_exact_knn_near_dupes_100() {
    run_e2e_exact_knn_check("near_dupes_100", 100);
}

// ── disconnected component separation ────────────────────────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_disconnected_200_component_separation() {
    let dataset = "disconnected_200";

    let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);
    let result = spectral_init(&graph, 2, 42, None)
        .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));

    let n = result.shape()[0];
    assert!(result.shape()[1] == 2);
    for &v in result.iter() {
        assert!(v.is_finite(), "non-finite in {dataset}: {v}");
    }

    // Load component labels (stored as int32 in the fixture)
    let labels: ndarray::Array1<i32> = common::load_dense(
        &common::fixture_path(dataset, "comp_c_components.npz"),
        "labels",
    );
    assert_eq!(labels.len(), n);

    // Compute component centroids.
    // For 4 components in 2D with max-10 scaling, per-point strict separation
    // (max_intra < min_inter) is not achievable — scale_and_add_noise expands all
    // coordinates to max 10, making per-component spread large relative to the gap
    // between adjacent clusters in the ring layout. Instead, verify that the spectral
    // meta-embedding places each component centroid at a distinct location: no two
    // centroids should be closer than 1.0 (much less than the ~8 unit centroid ring
    // radius that spectral placement produces for 4 components).
    let unique_labels: Vec<i32> = {
        let mut s: Vec<i32> = labels.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
        s.sort();
        s
    };
    let mut centroids: Vec<(f64, f64)> = Vec::new();
    for &comp in &unique_labels {
        let pts: Vec<(f64, f64)> = (0..n)
            .filter(|&i| labels[i] == comp)
            .map(|i| (result[[i, 0]] as f64, result[[i, 1]] as f64))
            .collect();
        assert!(!pts.is_empty(), "component {comp} has no points — fixture labels may be corrupt");
        let cx = pts.iter().map(|&(x, _)| x).sum::<f64>() / pts.len() as f64;
        let cy = pts.iter().map(|&(_, y)| y).sum::<f64>() / pts.len() as f64;
        centroids.push((cx, cy));
    }
    // Assert all component centroids are mutually distinct (separated by at least 1.0).
    // This confirms the meta-embedding produces distinct positions for each component.
    let min_centroid_dist: f64 = (0..unique_labels.len())
        .flat_map(|i| ((i + 1)..unique_labels.len()).map(move |j| (i, j)))
        .map(|(i, j)| {
            let (cx_i, cy_i) = centroids[i];
            let (cx_j, cy_j) = centroids[j];
            ((cx_i - cx_j).powi(2) + (cy_i - cy_j).powi(2)).sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_centroid_dist > 1.0,
        "{dataset}: component centroids not sufficiently distinct: \
         min centroid-to-centroid distance = {min_centroid_dist:.4} <= 1.0"
    );
}

// ── performance gate ──────────────────────────────────────────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py --knn-method both"]
fn test_e2e_performance_blobs_5000() {
    let path = common::fixture_path("blobs_5000", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&path);
    let start = std::time::Instant::now();
    let result = spectral_init(&graph, 2, 42, None).expect("blobs_5000 spectral_init failed");
    let elapsed = start.elapsed();
    assert_eq!(result.shape()[1], 2);
    assert!(
        elapsed.as_secs() < 30,
        "blobs_5000 took {:?}, expected < 30s",
        elapsed
    );
}
