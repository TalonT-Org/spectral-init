//! SIMD SpMV ISA baseline — Criterion benchmark harness (groupB: CSR scaling, groupC: SELL-C-σ).
//!
//! Run with: cargo bench --features testing --bench simd_spmv_exp

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray_npy::NpzReader;
use spectral_init::operator::spmv_csr;
use spectral_init::solve_eigenproblem_pub;
use std::hint::black_box;

// ─── Ring Laplacian builder ───────────────────────────────────────────────────

/// Build the symmetric normalized Laplacian for an `n`-node ring graph with
/// `half` neighbours on each side (2×half off-diagonal NNZ per row).
///
/// Returns raw CSR slices `(indptr, indices, data)` in f64.
/// - Diagonal entries: 1.0 (identity term of L = I - D^{-1/2} W D^{-1/2}).
/// - Off-diagonal entries: -1/(2*half) (uniform degree d = 2*half).
///
/// Precondition: n >= 2*half + 1 (no column-index collisions on wrap-around).
fn make_ring_lap(n: usize, half: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let d = (2 * half) as f64;
    let off_val = -1.0 / d;
    let nnz_per_row = 2 * half + 1; // half left + 1 diagonal + half right

    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n * nnz_per_row);
    let mut data = Vec::with_capacity(n * nnz_per_row);

    indptr.push(0usize);

    for i in 0..n {
        // Collect neighbour column indices (left and right), then add diagonal.
        let mut row_cols: Vec<usize> = (1..=half)
            .flat_map(|k| [(i + n - k) % n, (i + k) % n])
            .collect();
        row_cols.push(i);
        row_cols.sort_unstable();

        for &j in &row_cols {
            indices.push(j);
            data.push(if j == i { 1.0_f64 } else { off_val });
        }
        indptr.push(indptr.last().copied().unwrap() + nnz_per_row);
    }

    (indptr, indices, data)
}

// ─── SELL-C-σ format ──────────────────────────────────────────────────────────

struct SellCsigma {
    c: usize,               // chunk height
    num_rows: usize,        // original n
    chunk_len: Vec<usize>,  // max NNZ per chunk
    col_idx: Vec<usize>,    // padded column indices, interleaved row-major within chunk
    values: Vec<f64>,       // padded values, same layout as col_idx
    perm: Vec<usize>,       // perm[orig_row] = sorted_pos (σ-sort permutation)
    inv_perm: Vec<usize>,   // inv_perm[sorted_pos] = orig_row (inverse)
}

impl SellCsigma {
    fn from_csr(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        c: usize,
    ) -> SellCsigma {
        // 1. Compute NNZ per row.
        let nnz: Vec<usize> = (0..n).map(|i| indptr[i + 1] - indptr[i]).collect();

        // 2. σ-sort: sort row indices by NNZ descending.
        //    inv_perm[sorted_pos] = orig_row
        let mut inv_perm: Vec<usize> = (0..n).collect();
        inv_perm.sort_unstable_by(|&a, &b| nnz[b].cmp(&nnz[a]));

        // 3. Compute forward permutation: perm[orig_row] = sorted_pos
        let mut perm = vec![0usize; n];
        for (sorted_pos, &orig_row) in inv_perm.iter().enumerate() {
            perm[orig_row] = sorted_pos;
        }

        // 4. Partition into chunks of size c.
        let n_chunks = (n + c - 1) / c;
        let mut chunk_len = Vec::with_capacity(n_chunks);
        let mut col_idx_buf = Vec::new();
        let mut values_buf = Vec::new();

        for c_idx in 0..n_chunks {
            let chunk_start = c_idx * c;
            let chunk_end = (chunk_start + c).min(n);

            // 5. Find chunk_max_nnz over actual rows in this chunk.
            let chunk_max = (chunk_start..chunk_end)
                .map(|sorted_pos| nnz[inv_perm[sorted_pos]])
                .max()
                .unwrap_or(0);
            chunk_len.push(chunk_max);

            // 6. Pad all c rows to chunk_max, interleaved row-major within chunk.
            //    offset = c_idx * c * chunk_max + r * chunk_max + k
            for r in 0..c {
                let sorted_pos = chunk_start + r;
                let (row_nnz, row_start) = if sorted_pos < n {
                    let orig = inv_perm[sorted_pos];
                    (nnz[orig], indptr[orig])
                } else {
                    (0, 0) // virtual padding row
                };
                for k in 0..chunk_max {
                    if k < row_nnz {
                        col_idx_buf.push(indices[row_start + k]);
                        values_buf.push(data[row_start + k]);
                    } else {
                        col_idx_buf.push(0);   // always a valid index
                        values_buf.push(0.0);   // contributes zero
                    }
                }
            }
        }

        SellCsigma { c, num_rows: n, chunk_len, col_idx: col_idx_buf, values: values_buf, perm, inv_perm }
    }

    fn spmv_scalar(&self, x: &[f64], y: &mut [f64]) {
        let n_chunks = (self.num_rows + self.c - 1) / self.c;
        let mut y_perm = vec![0.0_f64; self.num_rows];

        for c_idx in 0..n_chunks {
            let clen = self.chunk_len[c_idx];
            if clen == 0 {
                continue; // chunk of zero-NNZ rows (can happen if n is divisible by c)
            }
            for r in 0..self.c {
                let sorted_pos = c_idx * self.c + r;
                if sorted_pos >= self.num_rows {
                    break; // last chunk: fewer than c real rows
                }
                let mut acc = 0.0_f64;
                let base = c_idx * self.c * clen + r * clen;
                for k in 0..clen {
                    let offset = base + k;
                    acc += self.values[offset] * x[self.col_idx[offset]];
                }
                y_perm[sorted_pos] = acc;
            }
        }

        // Un-permute: inv_perm[sorted_pos] = orig_row
        for i in 0..self.num_rows {
            y[self.inv_perm[i]] = y_perm[i];
        }
    }
}

// ─── AVX2 gather kernel ───────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn spmv_avx2_gather(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    use std::arch::x86_64::*;

    let n = y.len();
    for i in 0..n {
        let start = indptr[i];
        let end   = indptr[i + 1];

        // SAFETY: avx2+fma guaranteed by #[target_feature(enable = "avx2,fma")].
        let (mut result, mut base) = unsafe {
            let mut acc = _mm256_setzero_pd();
            let mut base = start;

            while base + 4 <= end {
                let vi = _mm_set_epi32(
                    indices[base + 3] as i32,
                    indices[base + 2] as i32,
                    indices[base + 1] as i32,
                    indices[base]     as i32,
                );
                let xv = _mm256_i32gather_pd(x.as_ptr(), vi, 8);
                let dv = _mm256_loadu_pd(data.as_ptr().add(base));
                acc = _mm256_fmadd_pd(dv, xv, acc);
                base += 4;
            }

            // Horizontal reduction: acc[0]+acc[1]+acc[2]+acc[3]
            let lo     = _mm256_castpd256_pd128(acc);
            let hi     = _mm256_extractf128_pd(acc, 1);
            let sum128 = _mm_add_pd(lo, hi);
            let halved = _mm_hadd_pd(sum128, sum128);
            let result = _mm_cvtsd_f64(halved);

            (result, base)
        };

        // Scalar tail for nnz % 4 != 0
        while base < end {
            result += data[base] * x[indices[base]];
            base += 1;
        }

        y[i] = result;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn spmv_avx2_dispatch(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: feature detection above guarantees avx2+fma are available.
        unsafe { spmv_avx2_gather(indptr, indices, data, x, y) }
    } else {
        spmv_csr(indptr, indices, data, x, y)
    }
}

// ─── Correctness verification ─────────────────────────────────────────────────

#[cfg(test)]
fn verify_sell_c(n: usize, c: usize) {
    let (indptr, indices, data) = make_ring_lap(n, 7);
    // Non-uniform x to catch permutation bugs
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();

    let mut y_csr = vec![0.0_f64; n];
    spmv_csr(&indptr, &indices, &data, &x, &mut y_csr);

    let sell = SellCsigma::from_csr(&indptr, &indices, &data, n, c);
    let mut y_sell = vec![0.0_f64; n];
    sell.spmv_scalar(&x, &mut y_sell);

    for i in 0..n {
        assert!(
            (y_csr[i] - y_sell[i]).abs() < 1e-12,
            "verify_sell_c(n={n}, c={c}): mismatch at row {i}: csr={} sell={}",
            y_csr[i],
            y_sell[i],
        );
    }
}

#[cfg(test)]
fn verify_avx2(n: usize) {
    let (indptr, indices, data) = make_ring_lap(n, 7);
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();

    let mut y_csr = vec![0.0_f64; n];
    spmv_csr(&indptr, &indices, &data, &x, &mut y_csr);

    let mut y_avx2 = vec![0.0_f64; n];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    spmv_avx2_dispatch(&indptr, &indices, &data, &x, &mut y_avx2);
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    spmv_csr(&indptr, &indices, &data, &x, &mut y_avx2);

    for i in 0..n {
        assert!(
            (y_csr[i] - y_avx2[i]).abs() < 1e-12,
            "verify_avx2(n={n}): mismatch at row {i}: csr={} avx2={}",
            y_csr[i],
            y_avx2[i],
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn verify_sell_c_correctness() {
        verify_sell_c(200, 4);
        verify_sell_c(2000, 8);
    }

    #[test]
    fn verify_avx2_correctness() {
        verify_avx2(200);
        verify_avx2(2000);
    }
}

// ─── Benchmark groups ──────────────────────────────────────────────────────────

fn bench_spmv_csr_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_csr_scaling");
    for &n in &[200_usize, 2000, 5000, 10000, 50000] {
        let (indptr, indices, data) = make_ring_lap(n, 7);
        let x = vec![1.0_f64 / (n as f64).sqrt(); n];
        let mut y = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                spmv_csr(
                    black_box(&indptr),
                    black_box(&indices),
                    black_box(&data),
                    black_box(&x),
                    black_box(&mut y),
                )
            });
        });
    }
    group.finish();
}

fn bench_spmv_sell_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_sell_c");
    for &n in &[200_usize, 2000, 5000, 10000, 50000] {
        let (indptr, indices, data) = make_ring_lap(n, 7);
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        for &chunk_c in &[4_usize, 8] {
            let sell = SellCsigma::from_csr(&indptr, &indices, &data, n, chunk_c);
            let mut y = vec![0.0_f64; n];
            group.bench_with_input(
                BenchmarkId::new(format!("C{chunk_c}"), n),
                &n,
                |b, _| {
                    b.iter(|| sell.spmv_scalar(black_box(&x), black_box(&mut y)))
                },
            );
        }
    }
    group.finish();
}

fn bench_spmv_sell_c_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_sell_c_conversion");
    for &n in &[200_usize, 2000, 5000, 10000, 50000] {
        let (indptr, indices, data) = make_ring_lap(n, 7);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                SellCsigma::from_csr(
                    black_box(&indptr),
                    black_box(&indices),
                    black_box(&data),
                    black_box(n),
                    black_box(8),
                )
            });
        });
    }
    group.finish();
}

fn bench_spmv_avx2(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_avx2");
    for &n in &[200_usize, 2000, 5000, 10000, 50000] {
        let (indptr, indices, data) = make_ring_lap(n, 7);
        let x = vec![1.0_f64 / (n as f64).sqrt(); n];
        let mut y = vec![0.0_f64; n];

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                spmv_csr(
                    black_box(&indptr),
                    black_box(&indices),
                    black_box(&data),
                    black_box(&x),
                    black_box(&mut y),
                )
            });
        });

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("avx2_gather", n), &n, |b, _| {
            b.iter(|| {
                spmv_avx2_dispatch(
                    black_box(&indptr),
                    black_box(&indices),
                    black_box(&data),
                    black_box(&x),
                    black_box(&mut y),
                )
            });
        });
    }
    group.finish();
}

// ─── blobs_5000 fixture loader ─────────────────────────────────────────────────

fn load_blobs5000_laplacian() -> sprs::CsMatI<f64, usize> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/blobs_5000/comp_b_laplacian.npz");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("cannot open blobs_5000 fixture {:?}: {}", path, e));
    let mut npz = NpzReader::new(file).expect("NpzReader failed on blobs_5000");

    let data: Vec<f64> = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("data")
        .expect("data key not found")
        .into_iter()
        .collect();

    let indices: Vec<usize> = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices")
        .expect("indices key not found")
        .iter()
        .map(|&x| x as usize)
        .collect();

    let indptr: Vec<usize> = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr")
        .expect("indptr key not found")
        .iter()
        .map(|&x| x as usize)
        .collect();

    let shape: Vec<usize> = npz
        .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix1>("shape")
        .expect("shape key not found")
        .iter()
        .map(|&x| x as usize)
        .collect();

    sprs::CsMatI::try_new((shape[0], shape[1]), indptr, indices, data)
        .expect("blobs_5000 CSR structure invalid")
}

fn bench_pipeline_blobs5000(c: &mut Criterion) {
    let laplacian = load_blobs5000_laplacian();
    let seed = 42_u64;

    // Warm-up: one call outside iter to prime instruction caches and BLAS state.
    let _ = solve_eigenproblem_pub(black_box(&laplacian), black_box(2), black_box(seed));

    let mut group = c.benchmark_group("pipeline_blobs5000");
    group.sample_size(10); // Criterion minimum; eigenproblem is slow so we keep it at the floor
    group.bench_function("blobs5000_solver", |b| {
        b.iter(|| {
            solve_eigenproblem_pub(
                black_box(&laplacian),
                black_box(2),
                black_box(seed),
            )
        });
    });
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_spmv_csr_scaling,
    bench_spmv_sell_c,
    bench_spmv_sell_c_conversion,
    bench_spmv_avx2,
    bench_pipeline_blobs5000
);
criterion_main!(benches);
