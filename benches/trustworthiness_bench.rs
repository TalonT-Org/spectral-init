use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn make_data(n: usize, d: usize, seed: u64) -> ndarray::Array2<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f64, 1.0).unwrap();
    let data: Vec<f64> = (0..n * d).map(|_| dist.sample(&mut rng)).collect();
    ndarray::Array2::from_shape_vec((n, d), data).unwrap()
}

fn bench_tw_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_tw_baseline");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| {
                spectral_init::trustworthiness(
                    black_box(x.view()),
                    black_box(y.view()),
                    black_box(k),
                )
            })
        });
    }
    group.finish();
}

fn bench_tw_thread_local(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_tw_thread_local");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("thread_local", n), &n, |b, _| {
            b.iter(|| {
                spectral_init::trustworthiness_thread_local(
                    black_box(x.view()),
                    black_box(y.view()),
                    black_box(k),
                )
            })
        });
    }
    group.finish();
}

fn bench_tw_partial_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_tw_partial_rank");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("partial_rank", n), &n, |b, _| {
            b.iter(|| {
                spectral_init::trustworthiness_partial_rank(
                    black_box(x.view()),
                    black_box(y.view()),
                    black_box(k),
                )
            })
        });
    }
    group.finish();
}

fn bench_tw_avx2(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_tw_avx2");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("avx2_kernel", n), &n, |b, _| {
            b.iter(|| {
                spectral_init::trustworthiness_avx2_kernel(
                    black_box(x.view()),
                    black_box(y.view()),
                    black_box(k),
                )
            })
        });
    }
    group.finish();
}

fn bench_tw_combined(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_tw_combined");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("combined", n), &n, |b, _| {
            b.iter(|| {
                spectral_init::trustworthiness_combined(
                    black_box(x.view()),
                    black_box(y.view()),
                    black_box(k),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches,
    bench_tw_baseline, bench_tw_thread_local, bench_tw_partial_rank,
    bench_tw_avx2, bench_tw_combined,
);
criterion_main!(benches);
