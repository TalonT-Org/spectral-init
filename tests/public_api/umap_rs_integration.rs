//! Integration with `umap-rs`: the graph it learns is a valid input, and the resulting
//! initialization is accepted by both of its fitting entry points.
//!
//! The optimizer of `umap-rs` draws from an unseeded generator, so nothing here asserts
//! the values of an optimized embedding.

use crate::fixtures;
use spectral_init::ndarray::{Array2, ArrayView2};
use spectral_init::sprs::CsMatI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};
use umap_rs::{
    EuclideanMetric, GraphParams, LearnedManifold, Metric, OptimizationParams, Optimizer, Umap,
    UmapConfig,
};

const N_COMPONENTS: usize = 2;

/// Epochs the optimizer runs: enough to move every vertex, few enough to stay quick.
const EPOCHS: usize = 30;

/// Raw data and k-nearest-neighbour arrays of a fixture dataset, in the types `umap-rs`
/// takes.
struct KnnFixture {
    data: Array2<f32>,
    knn_indices: Array2<u32>,
    knn_dists: Array2<f32>,
}

fn load_knn_fixture(dataset: &str) -> KnnFixture {
    let data: Array2<f64> =
        fixtures::load_dense(&fixtures::fixture_path(dataset, "step0_raw_data.npz"), "X");
    let knn_path = fixtures::fixture_path(dataset, "step1_knn.npz");
    let knn_indices: Array2<i32> = fixtures::load_dense(&knn_path, "knn_indices");
    let knn_dists: Array2<f32> = fixtures::load_dense(&knn_path, "knn_dists");
    KnnFixture {
        data: data.mapv(|v| v as f32),
        knn_indices: knn_indices.mapv(|i| u32::try_from(i).expect("neighbour index is negative")),
        knn_dists,
    }
}

fn umap_config(graph: GraphParams) -> UmapConfig {
    UmapConfig {
        n_components: N_COMPONENTS,
        graph,
        optimization: OptimizationParams {
            n_epochs: Some(EPOCHS),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn learn_manifold(fixture: &KnnFixture, config: &UmapConfig) -> LearnedManifold {
    assert_eq!(fixture.knn_indices.ncols(), config.graph.n_neighbors);
    Umap::new(config.clone()).learn_manifold(
        fixture.data.view(),
        fixture.knn_indices.view(),
        fixture.knn_dists.view(),
    )
}

/// The fuzzy graph `umap-rs` builds for a fixture dataset with the given
/// `set_op_mix_ratio`.
pub(crate) fn fuzzy_graph(dataset: &str, set_op_mix_ratio: f32) -> CsMatI<f32, u32, usize> {
    let config = umap_config(GraphParams {
        set_op_mix_ratio,
        ..Default::default()
    });
    learn_manifold(&load_knn_fixture(dataset), &config)
        .graph()
        .clone()
}

/// Vertices without any edge: `umap-rs` never moves them and `Umap::fit` reports them
/// as NaN.
fn isolated_vertices(manifold: &LearnedManifold) -> Vec<usize> {
    manifold
        .graph()
        .outer_iterator()
        .enumerate()
        .filter(|(_, row)| row.data().iter().sum::<f32>() == 0.0)
        .map(|(vertex, _)| vertex)
        .collect()
}

fn assert_rows_finite(embedding: ArrayView2<'_, f32>, isolated: &[usize], context: &str) {
    for (vertex, row) in embedding.outer_iter().enumerate() {
        if !isolated.contains(&vertex) {
            assert!(
                row.iter().all(|v| v.is_finite()),
                "{context}: non-finite coordinates for vertex {vertex}"
            );
        }
    }
}

#[test]
fn spectral_init_feeds_both_umap_rs_entry_points() {
    // A connected graph, a graph with several components, and a larger one.
    for dataset in ["blobs_connected_200", "disconnected_200", "blobs_500"] {
        let fixture = load_knn_fixture(dataset);
        let n = fixture.data.nrows();
        let config = umap_config(GraphParams::default());
        let manifold = learn_manifold(&fixture, &config);
        let isolated = isolated_vertices(&manifold);

        // The learned graph passes validation as it is.
        let init_config = SpectralInitConfig::new(N_COMPONENTS).with_seed(42);
        let init = spectral_init(
            manifold.graph().view(),
            Some(fixture.data.view()),
            &init_config,
        )
        .unwrap_or_else(|e| panic!("{dataset}: {e}"));
        assert_eq!(init.dim(), (n, N_COMPONENTS), "{dataset}");
        assert!(init.iter().all(|v| v.is_finite()), "{dataset}");
        // `Optimizer::new` reads the initialization through `as_slice_mut`.
        assert!(init.is_standard_layout(), "{dataset}");

        // The checkpointing path: `Optimizer` on the manifold the graph came from.
        let metric = EuclideanMetric;
        let mut optimizer = Optimizer::new(
            manifold,
            init.clone(),
            EPOCHS,
            &config,
            metric.metric_type(),
        );
        optimizer.step_epochs(EPOCHS, &metric);
        assert_eq!(optimizer.remaining_epochs(), 0);
        let fitted = optimizer.into_fitted(config.clone());
        assert_eq!(fitted.embedding().dim(), (n, N_COMPONENTS));
        assert_rows_finite(fitted.embedding(), &isolated, dataset);

        // The one-call path, which builds the graph again.
        let fitted = Umap::new(config).fit(
            fixture.data.view(),
            fixture.knn_indices.view(),
            fixture.knn_dists.view(),
            init.view(),
        );
        assert_eq!(fitted.embedding().dim(), (n, N_COMPONENTS));
        assert_rows_finite(fitted.embedding(), &isolated, dataset);
    }
}

#[test]
fn graph_without_symmetrization_is_rejected() {
    let fixture = load_knn_fixture("blobs_connected_200");
    let config = umap_config(GraphParams {
        symmetrize: false,
        ..Default::default()
    });
    let manifold = learn_manifold(&fixture, &config);
    assert_err!(
        spectral_init(
            manifold.graph().view(),
            Some(fixture.data.view()),
            &SpectralInitConfig::new(N_COMPONENTS),
        ),
        SpectralError::NotSymmetric { .. }
    );
}
