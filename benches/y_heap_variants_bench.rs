use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use std::hint::black_box;
use std::time::Duration;

fn make_data(
    n: usize,
    d_x: usize,
    d_y: usize,
    seed: u64,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let x = ndarray::Array2::from_shape_fn((n, d_x), |_| rng.random::<f64>());
    let y = ndarray::Array2::from_shape_fn((n, d_y), |_| rng.random::<f64>());
    (x, y)
}

fn bench_baseline(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    let mut group = c.benchmark_group("y_heap_baseline");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(10));
    for &n in &[1_000usize, 5_000, 10_000] {
        let (x, y) = make_data(n, 10, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| black_box(spectral_init::trustworthiness(x.view(), y.view(), 15)));
        });
    }
    group.finish();
}

fn bench_heap_reuse(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    let mut group = c.benchmark_group("y_heap_heap_reuse");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(10));
    for &n in &[1_000usize, 5_000, 10_000] {
        let (x, y) = make_data(n, 10, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                black_box(spectral_init::trustworthiness_heap_reuse(
                    x.view(),
                    y.view(),
                    15,
                ))
            });
        });
    }
    group.finish();
}

fn bench_flat_partial(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    let mut group = c.benchmark_group("y_heap_flat_partial");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(10));
    for &n in &[1_000usize, 5_000, 10_000] {
        let (x, y) = make_data(n, 10, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                black_box(spectral_init::trustworthiness_flat_partial(
                    x.view(),
                    y.view(),
                    15,
                ))
            });
        });
    }
    group.finish();
}

fn bench_flat_simd(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    let mut group = c.benchmark_group("y_heap_flat_simd");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(10));
    for &n in &[1_000usize, 5_000, 10_000] {
        let (x, y) = make_data(n, 10, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                black_box(spectral_init::trustworthiness_flat_simd(
                    x.view(),
                    y.view(),
                    15,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline,
    bench_heap_reuse,
    bench_flat_partial,
    bench_flat_simd
);
criterion_main!(benches);
