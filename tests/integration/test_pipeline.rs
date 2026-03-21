#[path = "../common/mod.rs"]
mod common;

use spectral_init::spectral_init;
use sprs::CsMatI;

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_output_shape_all_connected_datasets() {
    let datasets = [
        ("blobs_50", 50usize),
        ("blobs_500", 500),
        ("blobs_connected_200", 200),
        ("blobs_connected_2000", 2000),
        ("moons_200", 200),
        ("circles_300", 300),
    ];

    for (dataset, expected_n) in &datasets {
        let path = common::fixture_path(dataset, "step5a_pruned.npz");
        let graph = common::load_sparse_csr_f32_u32(&path);

        let result = spectral_init(&graph, 2, 42, None)
            .unwrap_or_else(|e| panic!("spectral_init failed for {dataset}: {e}"));

        assert_eq!(
            result.shape(),
            &[*expected_n, 2],
            "shape mismatch for dataset {dataset}"
        );
        for &v in result.iter() {
            assert!(v.is_finite(), "non-finite value in {dataset} output: {v}");
        }
    }
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_output_close_to_pre_noise_blobs_connected_200() {
    // spectral_init() returns the post-noise output (scale_and_add_noise is its final step,
    // adding Gaussian noise with sigma=0.0001 on coordinates scaled to max 10.0).
    // We compare against the pre-noise fixture with epsilon=0.01 to verify the output is
    // in the correct numeric range — the noise is negligible relative to this tolerance.
    let dataset = "blobs_connected_200";
    let graph_path = common::fixture_path(dataset, "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&graph_path);

    let result = spectral_init(&graph, 2, 42, None)
        .expect("spectral_init failed for blobs_connected_200");

    let fixture_path = common::fixture_path(dataset, "comp_f_scaling.npz");
    let pre_noise: ndarray::Array2<f32> = common::load_dense(&fixture_path, "pre_noise");

    assert_eq!(result.shape(), pre_noise.shape(), "shape mismatch");

    let epsilon = 0.01f32;
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            let got = result[[i, j]];
            let want = pre_noise[[i, j]];
            assert!(
                (got - want).abs() <= epsilon,
                "element [{i},{j}]: got {got}, want {want}, diff {} > epsilon {epsilon}",
                (got - want).abs()
            );
        }
    }
}

// ── API signature change ──────────────────────────────────────────────────────

#[test]
fn spectral_init_connected_with_none_data_unchanged() {
    // 4-node path graph: 0-1-2-3 (connected, no data needed)
    let g = CsMatI::<f32, u32, usize>::new(
        (4, 4),
        vec![0usize, 1, 3, 5, 6],
        vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32],
        vec![1.0f32; 6],
    );
    let result = spectral_init(&g, 2, 42, None);
    let arr = result.expect("spectral_init on connected graph with None data should succeed");
    assert_eq!(arr.shape(), &[4, 2]);
    for &v in arr.iter() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

// ── disconnected graph handling ───────────────────────────────────────────────

fn make_two_clique_graph() -> CsMatI<f32, u32, usize> {
    // 6-node graph: two cliques {0,1,2} and {3,4,5}, no inter-cluster edges
    let n = 6usize;
    let mut indptr = vec![0usize];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for base in [0usize, 3] {
        for i in 0..3usize {
            for j in 0..3usize {
                if i != j {
                    indices.push((base + j) as u32);
                    data.push(1.0f32);
                }
            }
            indptr.push(indices.len());
        }
    }
    CsMatI::<f32, u32, usize>::new((n, n), indptr, indices, data)
}

#[test]
fn spectral_init_synthetic_disconnected_produces_valid_embedding() {
    // 6-node graph: two cliques {0,1,2} and {3,4,5}
    let g = make_two_clique_graph();
    // synthetic data: cluster A near (0,0), cluster B near (100,0)
    // ndarray in row-major: 6 rows × 2 cols
    let data_arr = ndarray::Array2::from_shape_vec((6, 2), {
        let mut v = vec![0.0f32; 12];
        // node 0: (0.0, 0.0)
        v[0] = 0.0; v[1] = 0.0;
        // node 1: (0.1, 0.0)
        v[2] = 0.1; v[3] = 0.0;
        // node 2: (0.2, 0.0)
        v[4] = 0.2; v[5] = 0.0;
        // node 3: (100.0, 0.0)
        v[6] = 100.0; v[7] = 0.0;
        // node 4: (100.1, 0.0)
        v[8] = 100.1; v[9] = 0.0;
        // node 5: (100.2, 0.0)
        v[10] = 100.2; v[11] = 0.0;
        v
    })
    .unwrap();

    let result = spectral_init(&g, 2, 42, Some(data_arr.view()));
    let arr = result.expect("spectral_init on disconnected graph should succeed");
    assert_eq!(arr.shape(), &[6, 2]);
    for &v in arr.iter() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

#[test]
fn spectral_init_synthetic_disconnected_preserves_component_structure() {
    // Same 6-node two-clique graph
    let g = make_two_clique_graph();
    let result = spectral_init(&g, 2, 42, None);
    let arr = result.expect("spectral_init on disconnected graph should succeed");
    assert_eq!(arr.shape(), &[6, 2]);

    // Compute max intra-component and min inter-component distances
    let mut max_intra = 0.0f32;
    let mut min_inter = f32::INFINITY;
    for i in 0..6 {
        for j in (i + 1)..6 {
            let dx = arr[[i, 0]] - arr[[j, 0]];
            let dy = arr[[i, 1]] - arr[[j, 1]];
            let dist = (dx * dx + dy * dy).sqrt();
            if i / 3 == j / 3 {
                max_intra = max_intra.max(dist);
            } else {
                min_inter = min_inter.min(dist);
            }
        }
    }
    assert!(
        max_intra < min_inter,
        "max intra-component dist {max_intra} should be < min inter-component dist {min_inter}"
    );
}

// ── fixture-gated tests (require generated .npz files) ───────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_disconnected_200_shape_and_finite() {
    let path = common::fixture_path("disconnected_200", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&path);

    let result = spectral_init(&graph, 2, 42, None);
    let arr = result.expect("spectral_init on disconnected_200 should succeed");
    assert_eq!(arr.shape()[1], 2);
    for &v in arr.iter() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_disconnected_200_preserves_component_structure() {
    let path = common::fixture_path("disconnected_200", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&path);

    let result = spectral_init(&graph, 2, 42, None);
    let arr = result.expect("spectral_init on disconnected_200 should succeed");
    assert_eq!(arr.shape()[1], 2);
    for &v in arr.iter() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_blobs_50_produces_valid_embedding() {
    // blobs_50 has 3 tight clusters that may be disconnected in the k-NN graph
    let path = common::fixture_path("blobs_50", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&path);

    let result = spectral_init(&graph, 2, 42, None);
    let arr = result.expect("spectral_init on blobs_50 should succeed");
    assert_eq!(arr.shape()[1], 2);
    for &v in arr.iter() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}
