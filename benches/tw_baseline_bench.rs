use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use ndarray::Array2;
use std::path::PathBuf;
use std::time::Duration;
use std::hint::black_box;

/// Physical core count of the benchmark machine.
/// Host: 1 socket × 8 cores × 2-way HT = 16 logical CPUs; pin to physical cores only.
const N_THREADS: usize = 8;

fn bench_tw_baseline(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build_global()
        .unwrap();

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-04-05-tw-perf-rerun-clean/data/gaussian");
    let k = 15;

    let mut group = c.benchmark_group("tw_baseline");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(60));

    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x: Array2<f64> = ndarray_npy::read_npy(data_dir.join(format!("gaussian_n{n}_x.npy")))
            .unwrap_or_else(|e| panic!("failed to load gaussian_n{n}_x.npy: {e}"));
        let y: Array2<f64> = ndarray_npy::read_npy(data_dir.join(format!("gaussian_n{n}_y.npy")))
            .unwrap_or_else(|e| panic!("failed to load gaussian_n{n}_y.npy: {e}"));
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| spectral_init::trustworthiness(black_box(x.view()), black_box(y.view()), black_box(k)))
        });
    }

    // n=100K: override parameters for long-running flat-sampling measurement
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(63);
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(1500));

    let n = 100_000usize;
    let x: Array2<f64> = ndarray_npy::read_npy(data_dir.join(format!("gaussian_n{n}_x.npy")))
        .unwrap_or_else(|e| panic!("failed to load gaussian_n{n}_x.npy: {e}"));
    let y: Array2<f64> = ndarray_npy::read_npy(data_dir.join(format!("gaussian_n{n}_y.npy")))
        .unwrap_or_else(|e| panic!("failed to load gaussian_n{n}_y.npy: {e}"));
    group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
        b.iter(|| spectral_init::trustworthiness(black_box(x.view()), black_box(y.view()), black_box(k)))
    });

    group.finish();
}

criterion_group!(benches, bench_tw_baseline);
criterion_main!(benches);
