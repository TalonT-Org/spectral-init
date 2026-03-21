#[path = "../common/mod.rs"]
mod common;

use spectral_init::{spectral_init, SpectralError};

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

        let result = spectral_init(&graph, 2, 42)
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

    let result = spectral_init(&graph, 2, 42)
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

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn spectral_init_disconnected_returns_disconnected_graph_error() {
    let path = common::fixture_path("disconnected_200", "step5a_pruned.npz");
    let graph = common::load_sparse_csr_f32_u32(&path);

    let result = spectral_init(&graph, 2, 42);
    assert!(
        matches!(result, Err(SpectralError::DisconnectedGraph)),
        "expected DisconnectedGraph error, got: {result:?}"
    );
}
