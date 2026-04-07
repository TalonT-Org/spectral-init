use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use kiddo::ImmutableKdTree;
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const DATA_DIR: &str = "research/2026-04-07-kdtree-y-knn-trustworthiness/data";
const K: usize = 15;
const DISTRIBUTIONS: &[&str] = &["uniform", "gauss"];
const N_VALUES: &[usize] = &[1_000, 5_000, 10_000, 50_000, 75_000, 100_000];

fn load_npy_pair(dist: &str, n: usize) -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> =
        read_npy(format!("{DATA_DIR}/{dist}_n{n}_x.npy")).unwrap();
    let y: Array2<f64> =
        read_npy(format!("{DATA_DIR}/{dist}_n{n}_y.npy")).unwrap();
    (x, y)
}

fn bench_variants(c: &mut Criterion) {
    let _ = rayon::current_num_threads(); // warm rayon

    for &dist in DISTRIBUTIONS {
        for &n in N_VALUES {
            let (x, y) = load_npy_pair(dist, n);

            // ── flat_simd group ──────────────────────────────────────────────
            let mut group = c.benchmark_group(format!("flat_simd_{dist}_n{n}"));
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);
            group.warm_up_time(Duration::from_secs(10));
            group.bench_function(BenchmarkId::from_parameter(n), |b| {
                b.iter(|| {
                    black_box(spectral_init::trustworthiness_inner(
                        x.view(),
                        y.view(),
                        K,
                        false,
                    ))
                });
            });
            group.finish();

            // ── kdtree group ─────────────────────────────────────────────────
            let mut group = c.benchmark_group(format!("kdtree_{dist}_n{n}"));
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);
            group.warm_up_time(Duration::from_secs(10));
            group.bench_function(BenchmarkId::from_parameter(n), |b| {
                // Capture one isolated build-time measurement per sample,
                // outside the Criterion measurement closure.
                let t_build = Instant::now();
                let points: Vec<[f64; 2]> = (0..n).map(|i| [y[[i, 0]], y[[i, 1]]]).collect();
                let _tree: Arc<ImmutableKdTree<f64, 2>> =
                    Arc::new(ImmutableKdTree::new_from_slice(&points));
                let build_ms = t_build.elapsed().as_secs_f64() * 1_000.0;
                eprintln!("[bench:build_ms] {build_ms:.6}");

                b.iter(|| {
                    black_box(spectral_init::trustworthiness_inner(
                        x.view(),
                        y.view(),
                        K,
                        true,
                    ))
                });
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_variants);
criterion_main!(benches);
