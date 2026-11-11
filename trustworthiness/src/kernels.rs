//! AVX2 squared-distance kernels used by the trustworthiness metric on x86_64.

/// Squared Euclidean distance using AVX2+FMA intrinsics, looped over all elements.
///
/// 4-wide YMM loop with FMA accumulation, horizontal reduce, scalar tail 0-3.
///
/// # Safety
/// Both slices must have the same length and at least 10 elements (caller ensures
/// `xi.len() == xj.len()` and `d_x >= 10` at the dispatch site).
#[doc(hidden)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dist_sq_avx2_looped(xi: &[f64], xj: &[f64]) -> f64 {
    debug_assert_eq!(
        xi.len(),
        xj.len(),
        "dist_sq_avx2_looped: slices must have equal length"
    );
    use std::arch::x86_64::*;
    let n = xi.len().min(xj.len());
    unsafe {
        let mut acc = _mm256_setzero_pd();
        let mut k = 0usize;
        while k + 4 <= n {
            let a = _mm256_loadu_pd(xi.as_ptr().add(k));
            let b = _mm256_loadu_pd(xj.as_ptr().add(k));
            let d = _mm256_sub_pd(a, b);
            acc = _mm256_fmadd_pd(d, d, acc);
            k += 4;
        }
        // Horizontal reduce: sum 4 lanes
        let lo = _mm256_castpd256_pd128(acc);
        let hi = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(lo, hi);
        let halved = _mm_hadd_pd(sum128, sum128);
        let mut result = _mm_cvtsd_f64(halved);
        // Scalar tail (0–3 remaining elements)
        while k < n {
            let d = xi[k] - xj[k];
            result += d * d;
            k += 1;
        }
        result
    }
}

/// Batch squared Euclidean distances from a 2D query point to `n` target points
/// stored in row-major layout (`y_flat`, stride 2). Processes two target points per
/// SIMD iteration using `_mm256_hadd_pd` for horizontal reduction.
///
/// # Safety
/// - `yi` must have exactly 2 elements.
/// - `y_flat` must have at least `n * 2` elements (row-major, d_y = 2).
/// - `out` must have at least `n` elements.
///
/// Called only when `d_y == 2`, `y.is_standard_layout()`, and `avx2` is detected at runtime.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dist_sq_2d_avx2_batch(yi: &[f64], y_flat: &[f64], n: usize, out: &mut [f64]) {
    use std::arch::x86_64::*;
    debug_assert!(yi.len() >= 2, "yi must have at least 2 elements");
    debug_assert!(
        y_flat.len() >= n * 2,
        "y_flat must have at least n*2 elements"
    );
    debug_assert!(out.len() >= n, "out must have at least n elements");
    unsafe {
        // Broadcast query: [yi[1], yi[0], yi[1], yi[0]] — _mm256_set_pd args are (e3,e2,e1,e0)
        let yi_bc = _mm256_set_pd(yi[1], yi[0], yi[1], yi[0]);
        let mut j = 0usize;
        while j + 1 < n {
            // Load two target points: [yj[0], yj[1], yj+1[0], yj+1[1]]
            debug_assert!(j * 2 + 3 < y_flat.len(), "y_flat bounds exceeded at j={j}");
            let yj_pair = _mm256_loadu_pd(y_flat.as_ptr().add(j * 2));
            let diff = _mm256_sub_pd(yi_bc, yj_pair);
            let sq = _mm256_mul_pd(diff, diff);
            // hadd: lower 128-bit → sq[0]+sq[1] = dist_j; upper 128-bit → sq[2]+sq[3] = dist_{j+1}
            let hadd = _mm256_hadd_pd(sq, sq);
            out[j] = _mm_cvtsd_f64(_mm256_castpd256_pd128(hadd));
            out[j + 1] = _mm_cvtsd_f64(_mm256_extractf128_pd(hadd, 1));
            j += 2;
        }
        // Scalar tail for odd n
        if j < n {
            let base = j * 2;
            let d0 = yi[0] - y_flat[base];
            let d1 = yi[1] - y_flat[base + 1];
            out[j] = d0 * d0 + d1 * d1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn t_tw_06_avx2_looped_kernel_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return; // skip on CPUs without AVX2+FMA
        }
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for len in [10, 13, 16, 50, 100] {
            let a: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
            let b: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
            let scalar: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y) * (x - y))
                .sum();
            // SAFETY: runtime check above guarantees AVX2+FMA; both slices have `len` elements.
            let avx2 = unsafe { super::dist_sq_avx2_looped(&a, &b) };
            assert!(
                (avx2 - scalar).abs() < 1e-10,
                "dist_sq_avx2_looped mismatch at len={len}: avx2={avx2}, scalar={scalar}"
            );
        }
    }

    #[test]
    fn t_tw_09_avx2_2d_kernel_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // skip on CPUs without AVX2
        }
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        for n in [0usize, 1, 2, 3, 10, 20, 51] {
            let y_flat: Vec<f64> = (0..n * 2).map(|_| rng.random::<f64>()).collect();
            let yi: Vec<f64> = (0..2).map(|_| rng.random::<f64>()).collect();
            let mut out_avx2 = vec![0.0f64; n];
            // SAFETY: runtime check above guarantees AVX2; y_flat has n*2, yi has 2, out has n.
            unsafe {
                super::dist_sq_2d_avx2_batch(&yi, &y_flat, n, &mut out_avx2);
            }
            for j in 0..n {
                let scalar: f64 = yi
                    .iter()
                    .zip(y_flat[j * 2..j * 2 + 2].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                assert!(
                    (out_avx2[j] - scalar).abs() < 1e-10,
                    "dist_sq_2d_avx2_batch mismatch at n={n}, j={j}: avx2={}, scalar={scalar}",
                    out_avx2[j]
                );
            }
        }
    }
}
