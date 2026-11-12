//! The seed contract: equal configurations give equal output within one process,
//! different seeds give different output, and the generator behind the seed is the
//! ChaCha12 stream pinned by a golden file.

use crate::fixtures;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;
use spectral_init::{ComputeMode, SpectralInitConfig, spectral_init};

const MODES: [ComputeMode; 2] = [ComputeMode::PythonCompat, ComputeMode::RustNative];

#[test]
fn same_config_gives_identical_output() {
    // A small graph (dense solver) and a fixture graph large enough for the iterative,
    // randomly initialized solvers.
    let small = fixtures::make_ring(60);
    let large = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(
        "blobs_connected_2000",
        "step5a_pruned.npz",
    ));
    for graph in [&small, &large] {
        for mode in MODES {
            let config = SpectralInitConfig::new(2)
                .with_seed(42)
                .with_compute_mode(mode);
            let first = spectral_init(graph.view(), None, &config).expect("first run");
            let second = spectral_init(graph.view(), None, &config).expect("second run");
            assert_eq!(first, second, "n={} mode={mode:?}", graph.rows());
        }
    }
}

#[test]
fn different_seeds_give_different_output() {
    let graph = fixtures::make_ring(60);
    for mode in MODES {
        let base = SpectralInitConfig::new(2).with_compute_mode(mode);
        let a = spectral_init(graph.view(), None, &base.clone().with_seed(1)).expect("seed 1");
        let b = spectral_init(graph.view(), None, &base.with_seed(2)).expect("seed 2");
        assert_eq!(a.dim(), b.dim());
        assert_ne!(a, b, "mode={mode:?}");
    }
}

#[test]
fn default_seed_is_zero() {
    let graph = fixtures::make_ring(60);
    let implicit = spectral_init(graph.view(), None, &SpectralInitConfig::new(2));
    let explicit = spectral_init(graph.view(), None, &SpectralInitConfig::new(2).with_seed(0));
    assert_eq!(implicit, explicit);
}

#[test]
fn chacha12_stream_matches_the_golden_file() {
    let golden: Vec<u64> = include_str!("golden/chacha12_seed42.txt")
        .lines()
        .map(|line| u64::from_str_radix(line, 16).expect("golden line is a hex u64"))
        .collect();
    assert_eq!(golden.len(), 64);

    let mut rng = ChaCha12Rng::seed_from_u64(42);
    let stream: Vec<u64> = (0..golden.len()).map(|_| rng.next_u64()).collect();
    assert_eq!(stream, golden);
}
