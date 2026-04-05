use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use std::path::PathBuf;
use std::time::Duration;
use std::hint::black_box;

const N_THREADS: usize = 8;

fn bench_tw_partial_rank_merfish(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build_global()
        .unwrap();

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-04-05-tw-perf-rerun-clean/data/merfish");

    let x: Array2<f64> = ndarray_npy::read_npy(data_dir.join("merfish_n50k_x.npy"))
        .expect("failed to load merfish_n50k_x.npy");
    let y: Array2<f64> = ndarray_npy::read_npy(data_dir.join("merfish_n50k_y.npy"))
        .expect("failed to load merfish_n50k_y.npy");
    let k = 15;
    let n = 50_000usize;

    let mut group = c.benchmark_group("tw_partial_rank_merfish");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(60));

    group.bench_with_input(BenchmarkId::new("partial_rank_merfish", n), &n, |b, _| {
        b.iter(|| spectral_init::trustworthiness_partial_rank(black_box(x.view()), black_box(y.view()), black_box(k)))
    });
    group.bench_with_input(BenchmarkId::new("baseline_merfish", n), &n, |b, _| {
        b.iter(|| spectral_init::trustworthiness(black_box(x.view()), black_box(y.view()), black_box(k)))
    });

    group.finish();
}

criterion_group!(benches, bench_tw_partial_rank_merfish);
criterion_main!(benches);
