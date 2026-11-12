//! Optimizer-independent evidence for the spectral initialization.
//!
//! For every reference dataset under `tests/fixtures/` this prints, as Markdown:
//!
//! - agreement with the umap-learn reference: eigenpair residuals, eigenvalue errors and
//!   the subspace Gram determinant on connected graphs; component count and separation on
//!   disconnected ones;
//! - trustworthiness of the raw initial coordinates, for the spectral initialization and
//!   for umap-learn's random one (uniform in [-10, 10]).
//!
//! Nothing here runs a UMAP optimizer.
//!
//! Run with: cargo run --release -p spectral-init-eval --bin init_evidence

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use spectral_init::{SpectralInitConfig, spectral_init};
use spectral_init_eval::assess::{
    DATASETS, Dataset, measure_accuracy, measure_disconnected, measure_parity,
    solve_reference_laplacian,
};
use spectral_init_eval::fixtures;
use trustworthiness::trustworthiness;

/// Embedding dimensions and seed of the spectral initialization.
const N_COMPONENTS: usize = 2;
const SEED: u64 = 42;

/// Neighbourhood size of the trustworthiness score: the `n_neighbors` the reference
/// graphs were built with.
const TRUST_K: usize = 15;

/// Seeds the random baseline is averaged over.
const RANDOM_SEEDS: std::ops::Range<u64> = 0..5;

/// umap-learn's random initialization: uniform in [-10, 10].
fn random_init(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, N_COMPONENTS), |_| rng.random_range(-10.0..10.0))
}

/// Version of umap-learn that generated the reference fixtures of `dataset`.
fn reference_umap_version(dataset: &Dataset) -> String {
    let meta_path = fixtures::fixture_path(dataset.name, "meta.json");
    let meta = std::fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("cannot read {meta_path:?}: {e}"));
    let meta: serde_json::Value =
        serde_json::from_str(&meta).unwrap_or_else(|e| panic!("cannot parse {meta_path:?}: {e}"));
    meta["env"]["umap"]
        .as_str()
        .unwrap_or_else(|| panic!("{meta_path:?} names no umap-learn version"))
        .to_string()
}

fn main() {
    let generated_at = humantime::format_rfc3339_seconds(std::time::SystemTime::now());
    println!("# Initialization evidence\n");
    println!(
        "spectral-init {} · reference: umap-learn {} · generated {generated_at}\n",
        env!("CARGO_PKG_VERSION"),
        reference_umap_version(&DATASETS[0]),
    );

    println!("## Connected graphs: agreement with umap-learn\n");
    println!(
        "| Dataset | n | Solver | Max eigenpair residual | Max eigenvalue error | Subspace Gram determinant |"
    );
    println!("|---|---|---|---|---|---|");
    for dataset in DATASETS.iter().filter(|dataset| dataset.connected) {
        let solve = solve_reference_laplacian(dataset);
        let accuracy = measure_accuracy(&solve, 0.0);
        let parity = measure_parity(dataset, &solve);
        println!(
            "| {} | {} | {} | {:.1e} | {:.1e} | {:.9} |",
            dataset.name,
            dataset.n,
            solve.level.name(),
            accuracy.max_residual,
            parity.max_eigenvalue_abs_error,
            parity.subspace_gram_det,
        );
    }

    println!("\n## Disconnected graphs\n");
    println!(
        "| Dataset | n | Components (umap-learn) | Components (spectral-init) | Separation ratio |"
    );
    println!("|---|---|---|---|---|");
    for dataset in DATASETS.iter().filter(|dataset| !dataset.connected) {
        let layout = measure_disconnected(dataset);
        println!(
            "| {} | {} | {} | {} | {:.2} |",
            dataset.name,
            dataset.n,
            dataset.n_graph_components,
            layout.n_graph_components,
            layout.separation_ratio,
        );
    }

    println!("\n## Trustworthiness of the initial coordinates (k = {TRUST_K})\n");
    println!(
        "| Dataset | n | Input dimensions | Spectral | Random (mean of {} seeds) |",
        RANDOM_SEEDS.end - RANDOM_SEEDS.start
    );
    println!("|---|---|---|---|---|");
    let config = SpectralInitConfig::new(N_COMPONENTS).with_seed(SEED);
    for dataset in DATASETS {
        let data: Array2<f64> = fixtures::load_dense(
            &fixtures::fixture_path(dataset.name, "step0_raw_data.npz"),
            "X",
        );
        let graph = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(
            dataset.name,
            "step5a_pruned.npz",
        ));
        let data_f32 = data.mapv(|v| v as f32);
        let init = spectral_init(graph.view(), Some(data_f32.view()), &config)
            .unwrap_or_else(|e| panic!("{}: {e}", dataset.name))
            .mapv(f64::from);

        let spectral = trustworthiness(data.view(), init.view(), TRUST_K);
        let random = RANDOM_SEEDS
            .map(|seed| trustworthiness(data.view(), random_init(dataset.n, seed).view(), TRUST_K))
            .sum::<f64>()
            / (RANDOM_SEEDS.end - RANDOM_SEEDS.start) as f64;
        println!(
            "| {} | {} | {} | {:.4} | {:.4} |",
            dataset.name,
            dataset.n,
            data.ncols(),
            spectral,
            random,
        );
    }
}
