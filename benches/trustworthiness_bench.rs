use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
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

fn bench_trustworthiness(c: &mut Criterion) {
    // Force Rayon thread-pool initialization before timing starts.
    let _ = rayon::current_num_threads();

    let mut group = c.benchmark_group("trustworthiness");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    for &n in &[1_000, 5_000, 50_000] {
        let (x, y) = make_data(n, 10, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| black_box(spectral_init::trustworthiness(x.view(), y.view(), 15)));
        });
    }
    group.finish();
}

fn bench_trustworthiness_d50(c: &mut Criterion) {
    // Force Rayon thread-pool initialization before timing starts.
    let _ = rayon::current_num_threads();

    let mut group = c.benchmark_group("trustworthiness_d50");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    for &n in &[1_000, 5_000, 10_000, 50_000] {
        let (x, y) = make_data(n, 50, 2, 42);
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| black_box(spectral_init::trustworthiness(x.view(), y.view(), 15)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_trustworthiness, bench_trustworthiness_d50);
criterion_main!(benches);
