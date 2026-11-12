//! Thread configuration: a sequential and a four-thread dense solve agree within a stated
//! tolerance.
//!
//! The test changes faer's process-wide parallelism setting, so this binary holds exactly
//! one test.

#[path = "common/fixtures.rs"]
mod fixtures;

use faer::Par;
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralInitConfig, spectral_init};

/// Largest coordinate difference accepted between two thread configurations, on
/// coordinates scaled to a maximum magnitude of 10.
const THREAD_TOLERANCE: f32 = 1e-5;

/// Restores faer's global parallelism when dropped, so the setting is put back even if
/// an assertion fails.
struct RestoreParallelism(Par);

impl Drop for RestoreParallelism {
    fn drop(&mut self) {
        faer::set_global_parallelism(self.0);
    }
}

/// A connected 1500-node graph: a ring with chords of uneven weight. Large enough for
/// faer's dense eigensolver to split its work across threads, small enough to stay on
/// the dense path (n < 2000).
fn chorded_ring(n: usize) -> CsMatI<f32, u32, usize> {
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        for step in [1usize, 2, 5, 17] {
            let j = (i + step) % n;
            let weight = 0.2 + 0.8 * (((i * 31 + step * 7) % 97) as f32 / 97.0);
            tri.add_triplet(i, j, weight);
            tri.add_triplet(j, i, weight);
        }
    }
    tri.to_csr()
}

#[test]
fn sequential_and_parallel_dense_solves_agree() {
    let _restore = RestoreParallelism(faer::get_global_parallelism());

    let graphs = [
        (
            "blobs_500",
            fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(
                "blobs_500",
                "step5a_pruned.npz",
            )),
        ),
        ("chorded ring, n = 1500", chorded_ring(1500)),
    ];
    let config = SpectralInitConfig::new(2).with_seed(42);

    for (name, graph) in &graphs {
        assert!(graph.rows() < 2000, "{name} must take the dense path");

        faer::set_global_parallelism(Par::Seq);
        let sequential = spectral_init(graph.view(), None, &config)
            .unwrap_or_else(|e| panic!("{name}, sequential: {e}"));
        faer::set_global_parallelism(Par::rayon(4));
        let parallel = spectral_init(graph.view(), None, &config)
            .unwrap_or_else(|e| panic!("{name}, four threads: {e}"));

        assert_eq!(sequential.dim(), parallel.dim());
        let max_abs_diff = sequential
            .iter()
            .zip(parallel.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("{name}: max |sequential - four threads| = {max_abs_diff:e}");
        assert!(
            max_abs_diff <= THREAD_TOLERANCE,
            "{name}: thread configurations differ by {max_abs_diff:e} > {THREAD_TOLERANCE:e}"
        );
    }
}
