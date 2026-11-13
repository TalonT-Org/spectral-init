//! Spectral initialization for `umap-rs`.
//!
//! `umap-rs` leaves the initial coordinates to the caller. This example builds the fuzzy
//! graph with `umap-rs`, computes the spectral initialization from that graph, and hands
//! it to the `umap-rs` optimizer:
//!
//! 1. synthetic data and a brute-force k-nearest-neighbour search,
//! 2. `Umap::learn_manifold` — the fuzzy graph,
//! 3. `spectral_init` on `manifold.graph()`,
//! 4. `Optimizer` — stochastic gradient descent from the spectral coordinates.
//!
//! Run with: cargo run --example umap_rs

use spectral_init::ndarray::{Array2, ArrayView2};
use spectral_init::{SpectralInitConfig, spectral_init};
use umap_rs::{EuclideanMetric, GraphParams, Metric, Optimizer, Umap, UmapConfig};

/// Three Gaussian-like blobs in five dimensions, `points_per_blob` points each, close
/// enough to one another that their neighbourhoods overlap.
fn blobs(points_per_blob: usize) -> Array2<f32> {
    const CENTERS: [[f32; 5]; 3] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 2.0, 0.0, 1.0],
    ];
    // A small linear congruential generator keeps the example free of extra crates.
    let mut state: u32 = 12345;
    let mut uniform = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    Array2::from_shape_fn((CENTERS.len() * points_per_blob, 5), |(i, j)| {
        // The sum of four uniform draws is bell-shaped with unit-order spread.
        let jitter: f32 = (0..4).map(|_| uniform() - 0.5).sum();
        CENTERS[i / points_per_blob][j] + 1.5 * jitter
    })
}

/// Exact k-nearest neighbours by Euclidean distance.
///
/// Each row lists the point itself first (distance 0), as `umap-rs` expects from a k-NN
/// search over the data set; it skips that entry when it builds the graph.
fn brute_force_knn(data: ArrayView2<'_, f32>, n_neighbors: usize) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    let mut indices = Array2::<u32>::zeros((n, n_neighbors));
    let mut distances = Array2::<f32>::zeros((n, n_neighbors));
    for i in 0..n {
        let mut neighbours: Vec<(f32, u32)> = (0..n)
            .map(|j| {
                let squared: f32 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j))
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (squared.sqrt(), j as u32)
            })
            .collect();
        neighbours.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (slot, &(distance, index)) in neighbours.iter().take(n_neighbors).enumerate() {
            indices[[i, slot]] = index;
            distances[[i, slot]] = distance;
        }
    }
    (indices, distances)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_components = 2;
    let n_neighbors = 15;

    let data = blobs(100);
    let n = data.nrows();
    // `learn_manifold` panics unless both k-NN arrays are (n, n_neighbors) and
    // n > n_neighbors.
    let (knn_indices, knn_dists) = brute_force_knn(data.view(), n_neighbors);

    // The fuzzy graph. `symmetrize` must stay `true` (the default): spectral
    // initialization needs a symmetric graph.
    let umap_config = UmapConfig {
        n_components,
        graph: GraphParams {
            n_neighbors,
            ..Default::default()
        },
        ..Default::default()
    };
    let umap = Umap::new(umap_config.clone());
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    // Spectral initialization from the graph. Both configurations name the same
    // number of components. `data` is read only if the graph has more than
    // `2 * n_components` connected components.
    let init_config = SpectralInitConfig::new(n_components).with_seed(42);
    let init = spectral_init(manifold.graph().view(), Some(data.view()), &init_config)?;
    // `Optimizer::new` does not check the shape of the initialization.
    assert_eq!(init.dim(), (n, n_components));

    // The epoch count `Umap::fit` would choose.
    let total_epochs =
        umap_config
            .optimization
            .n_epochs
            .unwrap_or(if n <= 10_000 { 500 } else { 200 });

    let metric = EuclideanMetric;
    let mut optimizer = Optimizer::new(
        manifold,
        init,
        total_epochs,
        &umap_config,
        metric.metric_type(),
    );
    optimizer.step_epochs(total_epochs, &metric);
    let fitted = optimizer.into_fitted(umap_config);

    // Like `Umap::fit`, mark vertices without any edge as NaN: the optimizer never
    // moves them.
    let isolated: Vec<usize> = fitted
        .manifold()
        .graph()
        .outer_iterator()
        .enumerate()
        .filter(|(_, row)| row.data().iter().sum::<f32>() == 0.0)
        .map(|(vertex, _)| vertex)
        .collect();
    let mut embedding = fitted.into_embedding();
    for &vertex in &isolated {
        embedding.row_mut(vertex).fill(f32::NAN);
    }

    println!(
        "Embedded {n} points into {n_components} dimensions ({} isolated).",
        isolated.len()
    );
    for i in (0..n).step_by(n / 6) {
        println!(
            "  point {i:3}: [{:8.4}, {:8.4}]",
            embedding[[i, 0]],
            embedding[[i, 1]]
        );
    }

    assert_eq!(embedding.dim(), (n, n_components));
    for (i, row) in embedding.outer_iter().enumerate() {
        if !isolated.contains(&i) {
            assert!(
                row.iter().all(|v| v.is_finite()),
                "non-finite coordinates for point {i}"
            );
        }
    }
    Ok(())
}
