//! Criterion microbenchmark for `dist_sq_avx2_looped` vs scalar at d_x=10 and d_x=50.
//!
//! Run with: cargo bench -p trustworthiness --features __bench --bench dist_sq_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[cfg(target_arch = "x86_64")]
use trustworthiness::dist_sq_avx2_looped;

// Note: the avx2_looped benchmark arm runs only when AVX2 and FMA are detected at runtime;
// on other hosts only the scalar variant runs.

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

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2_looped", d), &d, |b, _| {
                // SAFETY: AVX2 and FMA were detected above; both slices have `d` elements.
                b.iter(|| unsafe { dist_sq_avx2_looped(black_box(&xi), black_box(&xj)) })
            });
        }

        group.bench_with_input(BenchmarkId::new("scalar", d), &d, |b, _| {
            b.iter(|| {
                black_box(&xi)
                    .iter()
                    .zip(black_box(&xj).iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dist_sq_kernels);
criterion_main!(benches);
