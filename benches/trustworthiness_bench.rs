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

// Stub groups for groupB variants — NOT implemented here
fn bench_tw_thread_local(_c: &mut Criterion) {
    todo!("thread_local variant: completed in groupB")
}

fn bench_tw_partial_rank(_c: &mut Criterion) {
    todo!("partial_rank variant: completed in groupB")
}

fn bench_tw_avx2(_c: &mut Criterion) {
    todo!("avx2_kernel variant: completed in groupB")
}

fn bench_tw_combined(_c: &mut Criterion) {
    todo!("combined variant: completed in groupB")
}

criterion_group!(benches, bench_tw_baseline);
criterion_main!(benches);
