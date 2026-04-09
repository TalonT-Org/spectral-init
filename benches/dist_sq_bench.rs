//! Criterion microbenchmark for `dist_sq_avx2_looped` vs scalar at d_x=10 and d_x=50.
//!
//! Run with: cargo bench --bench dist_sq_bench --features testing

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use spectral_init::metrics::dist_sq_avx2_looped;

fn make_vectors(d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut xi = Vec::with_capacity(d);
    let mut xj = Vec::with_capacity(d);
    for i in 0..d {
        let mut h = DefaultHasher::new();
        (seed, i as u64).hash(&mut h);
        let bits = h.finish();
        xi.push((bits as f64) / (u64::MAX as f64) * 2.0 - 1.0);
        let mut h2 = DefaultHasher::new();
        (seed.wrapping_add(1), i as u64).hash(&mut h2);
        let bits2 = h2.finish();
        xj.push((bits2 as f64) / (u64::MAX as f64) * 2.0 - 1.0);
    }
    (xi, xj)
}

fn bench_dist_sq_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("dist_sq_kernels");

    for &d in &[10_usize, 50] {
        let (xi, xj) = make_vectors(d, 42);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        group.bench_with_input(
            BenchmarkId::new("avx2_looped", d),
            &d,
            |b, _| b.iter(|| unsafe {
                dist_sq_avx2_looped(black_box(&xi), black_box(&xj))
            }),
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", d),
            &d,
            |b, _| b.iter(|| {
                black_box(&xi).iter().zip(black_box(&xj).iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
            }),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dist_sq_kernels);
criterion_main!(benches);
