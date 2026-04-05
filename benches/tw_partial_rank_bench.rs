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

fn bench_tw_partial_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("tw_partial_rank");
    for n in [1000usize, 5000, 10000, 25000, 50000] {
        let x = make_data(n, 10, 0);
        let y = make_data(n, 2, 1);
        let k = 15;
        group.bench_with_input(BenchmarkId::new("partial_rank", n), &n, |b, _| {
            b.iter(|| spectral_init::trustworthiness_partial_rank(black_box(x.view()), black_box(y.view()), black_box(k)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tw_partial_rank);
criterion_main!(benches);
