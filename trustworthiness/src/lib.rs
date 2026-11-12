//! Trustworthiness of a low-dimensional embedding.
//!
//! [`trustworthiness`] measures how well the k-nearest-neighbour structure of the
//! original data `X` is preserved in an embedding `Y`. [`trustworthiness_subsampled`]
//! evaluates the same score on a random subset of query rows.
//!
//! # Feature flags
//!
//! | Flag | Purpose |
//! |------|---------|
//! | `cli` | Builds the `trustworthiness` and `tw_profiler` binaries, which read `.npy` files. |
//! | `profiling` | Prints per-step `[timing:<step>] <nanoseconds>` lines to stderr. |

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::HashSet;

#[cfg(target_arch = "x86_64")]
mod kernels;

#[cfg(target_arch = "x86_64")]
use kernels::dist_sq_2d_avx2_batch;
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub use kernels::dist_sq_avx2_looped;

/// Squared Euclidean distance between two rows, summed in index order: the scalar
/// path of both distance steps.
fn dist_sq_scalar(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&p, &q)| (p - q) * (p - q))
        .sum()
}

/// Shared inner kernel for [`trustworthiness`] and [`trustworthiness_subsampled`].
///
/// `query_indices` selects which rows participate as outer "query" points; values
/// must lie in `[0, n)` and should be unique (duplicates would double-count their
/// penalty contribution). The per-row computation always runs against the full
/// `n`-point population. Only the leading factor of the normalization denominator
/// uses `m = query_indices.len()` — the `(2n − 3k − 1)` factor still uses the
/// full population `n`.
pub(crate) fn trustworthiness_inner(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
    query_indices: Option<&[usize]>,
) -> f64 {
    use std::cell::RefCell;
    let n = x.nrows();
    assert_eq!(
        y.nrows(),
        n,
        "trustworthiness: x and y must have the same number of rows"
    );
    assert!(k > 0, "trustworthiness: k must be > 0");
    assert!(
        k < n / 2,
        "trustworthiness: k must be < n/2 (got k={k}, n={n}, n/2={}); \
        this constraint is required by the normalization denominator and matches sklearn's ValueError",
        n / 2
    );
    let m_query = query_indices.map_or(n, |q| q.len());

    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        X_DIST_NS.store(0, Ordering::Relaxed);
        X_SORT_NS.store(0, Ordering::Relaxed);
        Y_DIST_NS.store(0, Ordering::Relaxed);
        PENALTY_NS.store(0, Ordering::Relaxed);
    }

    // SIMD dispatch needs C-contiguous rows; other layouts take the scalar path.
    #[cfg(target_arch = "x86_64")]
    let use_avx2_x = x.ncols() >= 10
        && x.is_standard_layout()
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2_x = false;

    #[cfg(target_arch = "x86_64")]
    let use_avx2_y =
        { y.ncols() == 2 && y.is_standard_layout() && is_x86_feature_detected!("avx2") };
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2_y = false;

    thread_local! {
        static COMB_DIST_X:  RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static COMB_INDICES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    thread_local! {
        static COMB_DIST_Y:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static COMB_INDICES_Y: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    #[cfg(feature = "profiling")]
    static X_DIST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static X_SORT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static Y_DIST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "profiling")]
    static PENALTY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let process_row = |i: usize| -> f64 {
        let xi = x.row(i);
        let yi = y.row(i);

        COMB_DIST_X.with(|dist_x_cell| {
            COMB_INDICES.with(|indices_cell| {
                let mut dist_x = dist_x_cell.borrow_mut();
                let mut indices = indices_cell.borrow_mut();

                // ── x_dist step ──────────────────────────────────────────────
                #[cfg(feature = "profiling")]
                let t_x_dist = std::time::Instant::now();

                dist_x.clear();
                dist_x.resize(n, 0.0f64);
                for j in 0..n {
                    let xj = x.row(j);
                    if use_avx2_x {
                        #[cfg(target_arch = "x86_64")]
                        {
                            let si = xi.as_slice().expect("x row must be contiguous");
                            let sj = xj.as_slice().expect("x row must be contiguous");
                            // SAFETY: use_avx2_x guarantees AVX2+FMA at run time; both
                            // rows are slices of equal length d_x >= 10.
                            dist_x[j] = unsafe { dist_sq_avx2_looped(si, sj) };
                        }
                    } else {
                        dist_x[j] = dist_sq_scalar(xi, xj);
                    }
                }

                #[cfg(feature = "profiling")]
                X_DIST_NS.fetch_add(
                    t_x_dist.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );

                // ── x_sort step ──────────────────────────────────────────────
                #[cfg(feature = "profiling")]
                let t_x_sort = std::time::Instant::now();

                indices.clear();
                indices.extend(0..n);
                indices.select_nth_unstable_by(k, |&a, &b| {
                    dist_x[a].total_cmp(&dist_x[b]).then(a.cmp(&b))
                });

                let knn_x_set: HashSet<usize> =
                    indices[..=k].iter().filter(|&&m| m != i).copied().collect();

                #[cfg(feature = "profiling")]
                X_SORT_NS.fetch_add(
                    t_x_sort.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );

                // ── y_dist step (flat_simd, replaces BinaryHeap) ─────────────
                #[cfg(feature = "profiling")]
                let t_y_dist = std::time::Instant::now();

                COMB_DIST_Y.with(|dy_cell| {
                    COMB_INDICES_Y.with(|iy_cell| {
                        let mut dist_y = dy_cell.borrow_mut();
                        let mut indices_y = iy_cell.borrow_mut();

                        dist_y.clear();
                        dist_y.resize(n, 0.0f64);

                        // Fill Y distances: AVX2 2D batch or scalar fallback.
                        if use_avx2_y {
                            #[cfg(target_arch = "x86_64")]
                            {
                                let y_flat = y
                                    .as_slice()
                                    .expect("y must be standard layout for AVX2 dispatch");
                                let yi_slice = &y_flat[i * 2..(i + 1) * 2];
                                // SAFETY: y_flat has n*d_y elements; yi_slice has 2 elements;
                                // dist_y has n elements; use_avx2_y guarantees d_y==2,
                                // standard_layout, and runtime AVX2.
                                unsafe {
                                    dist_sq_2d_avx2_batch(yi_slice, y_flat, n, &mut dist_y);
                                }
                            }
                        } else {
                            for j in 0..n {
                                dist_y[j] = dist_sq_scalar(yi, y.row(j));
                            }
                        }

                        dist_y[i] = f64::INFINITY; // self-exclusion

                        indices_y.clear();
                        indices_y.extend(0..n);
                        indices_y.select_nth_unstable_by(k, |&a, &b| {
                            dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b))
                        });

                        #[cfg(feature = "profiling")]
                        Y_DIST_NS.fetch_add(
                            t_y_dist.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );

                        // ── penalty step ──────────────────────────────────────
                        #[cfg(feature = "profiling")]
                        let t_penalty = std::time::Instant::now();

                        let mut row_penalty = 0u64;
                        for &j in &indices_y[..k] {
                            if !knn_x_set.contains(&j) {
                                let dj = dist_x[j];
                                let rank: usize = (0..n)
                                    .filter(|&m| dist_x[m] < dj || (dist_x[m] == dj && m < j))
                                    .count();
                                row_penalty += (rank - k) as u64;
                            }
                        }

                        #[cfg(feature = "profiling")]
                        PENALTY_NS.fetch_add(
                            t_penalty.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );

                        row_penalty as f64
                    })
                })
            })
        })
    };

    let penalty_sum: f64 = match query_indices {
        None => (0..n).into_par_iter().map(&process_row).sum(),
        Some(qi) => qi.par_iter().copied().map(&process_row).sum(),
    };

    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        eprintln!("[timing:x_dist] {}", X_DIST_NS.load(Ordering::Relaxed));
        eprintln!("[timing:x_sort] {}", X_SORT_NS.load(Ordering::Relaxed));
        eprintln!("[timing:y_dist] {}", Y_DIST_NS.load(Ordering::Relaxed));
        eprintln!("[timing:penalty] {}", PENALTY_NS.load(Ordering::Relaxed));
    }

    let denom = m_query as f64 * k as f64 * (2 * n).saturating_sub(3 * k + 1) as f64;
    1.0 - penalty_sum * 2.0 / denom
}

/// Computes the trustworthiness metric T(k) — a measure of how well the k-nearest-neighbor
/// structure of X (high-dimensional) is preserved in Y (embedding).
///
/// Uses partial rank (introselect) for O(n)-average X-NN detection, thread-local
/// scratch buffers to eliminate per-row allocations, and AVX2+FMA SIMD dispatch
/// for X-distance computation when dimensionality >= 10 and `x` is C-contiguous;
/// other layouts use the scalar path.
///
/// # Formula
/// `T(k) = 1 − (2 / (n·k·(2n−3k−1))) · Σᵢ Σ_{j ∈ U_i(k)} (r(i,j) − k)`
///
/// where `r(i,j)` is the 0-indexed rank of j in the distance ordering from i in X
/// (self = rank 0), and `U_i(k)` is the set of j in k-NN(i, Y) but NOT in k-NN(i, X).
///
/// # Panics
/// Panics if `k == 0`, `k >= n / 2`, or if `x` and `y` have different row counts.
/// The `k < n/2` restriction is required by the normalization denominator
/// `G_k = n·k·(2n−3k−1)`: when `k ≥ n/2` this denominator produces T > 1.
/// The guard uses integer floor division (`n / 2`), matching sklearn's exact boundary
/// condition (`n_neighbors >= n // 2` raises `ValueError` in sklearn).
pub fn trustworthiness(x: ArrayView2<f64>, y: ArrayView2<f64>, k: usize) -> f64 {
    trustworthiness_inner(x, y, k, None)
}

/// Sub-sampled trustworthiness: evaluates only `m` randomly chosen query rows
/// against the full `n`-point population.
///
/// Identical to [`trustworthiness`] except that the outer iteration is restricted
/// to `m` query rows sampled uniformly without replacement using `seed`. The
/// per-row computation (full pairwise X-distances, full Y k-NN, rank penalty
/// scan) still runs against all `n` points — only the outer iterator and the
/// leading denominator factor change.
///
/// # Formula
/// `T(k, m) = 1 − (2 / (m·k·(2n−3k−1))) · Σ_{i ∈ Q} Σ_{j ∈ U_i(k)} (r(i,j) − k)`
///
/// where `Q ⊂ [0, n)` is the set of `m` sampled query rows. The
/// `(2n − 3k − 1)` factor uses the full population `n`, so when `m == n` the
/// result is identical to `trustworthiness(x, y, k)`.
///
/// # Determinism
/// The sample is reproducible: equal `seed` values produce equal row sets.
/// When `m == n`, the function bypasses sampling entirely and uses the full
/// `0..n` range, guaranteeing parity with `trustworthiness`.
///
/// # Panics
/// Panics if `k == 0`, `k >= n / 2`, `m == 0`, `m > n`, or if `x` and `y` have
/// different row counts.
pub fn trustworthiness_subsampled(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
    m: usize,
    seed: u64,
) -> f64 {
    let n = x.nrows();
    assert!(m > 0, "trustworthiness_subsampled: m must be > 0");
    assert!(
        m <= n,
        "trustworthiness_subsampled: m must be <= n (got m={m}, n={n})"
    );

    if m == n {
        return trustworthiness_inner(x, y, k, None);
    }

    let query_indices = sample_query_indices(n, m, seed);
    trustworthiness_inner(x, y, k, Some(&query_indices))
}

/// `m` distinct row indices in `[0, n)`, sampled uniformly without replacement from `seed`.
fn sample_query_indices(n: usize, m: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    rand::seq::index::sample(&mut rng, n, m).into_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_tw_01_perfect_preservation() {
        // 20-point grid; each point embedded to its own location
        let x = ndarray::Array2::from_shape_fn((20, 4), |(i, d)| (i * 4 + d) as f64);
        let y = x.slice(ndarray::s![.., ..2]).to_owned();
        let t = trustworthiness(x.view(), y.view(), 5);
        assert!((t - 1.0).abs() < 1e-10, "perfect preservation: T={t}");
    }

    #[test]
    fn t_tw_02_result_in_unit_interval() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        let x = ndarray::Array2::from_shape_fn((20, 10), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((20, 2), |_| rng.random::<f64>());
        let t = trustworthiness(x.view(), y.view(), 5);
        assert!((0.0..=1.0).contains(&t), "T out of [0,1]: {t}");
    }

    /// Hand-verifiable 4-point, k=1 example with exactly one violation of rank penalty 1.
    ///
    /// n=4, k=1: denominator = 4·1·(8−3−1) = 16.
    ///
    /// X = [[0,0],[1,0],[0,1.5],[0,100]]:
    ///   KNN(0,X,1)={1} (rank_x[2]=2), KNN(1,X,1)={0}, KNN(2,X,1)={0}, KNN(3,X,1)={2}
    ///
    /// Y = [[0,0],[1,0],[0,0.5],[0,100]] (point 2 moved closer to 0):
    ///   KNN(0,Y,1)={2} → violation: rank_x[2]=2, penalty = 2−1 = 1
    ///   KNN(1,Y,1)={0} → 0 ∈ KNN(1,X,1): no penalty
    ///   KNN(2,Y,1)={0} → 0 ∈ KNN(2,X,1): no penalty
    ///   KNN(3,Y,1)={2} → 2 ∈ KNN(3,X,1): no penalty
    ///
    /// T = 1 − 2·1/16 = 7/8 = 0.875
    #[test]
    fn t_tw_03_formula_hand_check() {
        let x = ndarray::Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 100.0],
        )
        .unwrap();
        let y = ndarray::Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 100.0],
        )
        .unwrap();
        let t = trustworthiness(x.view(), y.view(), 1);
        let expected = 7.0 / 8.0;
        assert!(
            (t - expected).abs() < 1e-10,
            "hand-check: T={t:.10}, expected {expected:.10} (7/8)"
        );
    }

    #[test]
    fn t_tw_04_k_less_than_half_n() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let n = 20usize;
        let k = n / 2 - 1; // 9, just below the boundary
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        let t = trustworthiness(x.view(), y.view(), k);
        assert!(t.is_finite(), "T should be finite at k=n/2-1: {t}");
        assert!((0.0..=1.0).contains(&t), "T out of [0,1] at k=n/2-1: {t}");
    }

    #[test]
    #[should_panic(expected = "k must be < n/2")]
    fn t_tw_05_panics_on_k_gte_half_n() {
        let n = 20usize;
        let k = n / 2; // exactly n/2 — must panic
        let x = ndarray::Array2::zeros((n, 4));
        let y = ndarray::Array2::zeros((n, 2));
        let _ = trustworthiness(x.view(), y.view(), k);
    }

    #[test]
    fn t_tw_07_partial_rank_matches_full_sort() {
        use rand::{Rng, SeedableRng};
        use std::collections::HashSet;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let n = 30;
        let k = 5;
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());

        for i in 0..n {
            let xi = x.row(i);
            let dist_x: Vec<f64> = (0..n)
                .map(|j| {
                    xi.iter()
                        .zip(x.row(j).iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum()
                })
                .collect();

            // Full-sort baseline: knn_x_set via sort + rank scatter
            let mut sorted: Vec<(f64, usize)> = dist_x.iter().copied().zip(0..n).collect();
            sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let knn_baseline: HashSet<usize> = sorted[1..=k].iter().map(|&(_, j)| j).collect();

            // Partial-rank approach: select_nth_unstable_by
            let mut indices: Vec<usize> = (0..n).collect();
            indices.select_nth_unstable_by(k, |&a, &b| {
                dist_x[a].total_cmp(&dist_x[b]).then(a.cmp(&b))
            });
            let knn_partial: HashSet<usize> =
                indices[..=k].iter().filter(|&&m| m != i).copied().collect();

            assert_eq!(knn_baseline, knn_partial, "knn_x_set mismatch at row {i}");

            // Verify rank values for all non-knn points
            let mut rank_x = vec![0usize; n];
            for (rank, &(_, j)) in sorted.iter().enumerate() {
                rank_x[j] = rank;
            }
            for j in 0..n {
                if j == i || knn_baseline.contains(&j) {
                    continue;
                }
                let dj = dist_x[j];
                let scan_rank: usize = (0..n)
                    .filter(|&m| dist_x[m] < dj || (dist_x[m] == dj && m < j))
                    .count();
                assert_eq!(
                    rank_x[j], scan_rank,
                    "rank mismatch for point {j} from row {i}: sort={}, scan={scan_rank}",
                    rank_x[j]
                );
            }
        }
    }

    /// Brute-force O(n²) reference over every row.
    fn trustworthiness_brute_force(
        x: ndarray::ArrayView2<f64>,
        y: ndarray::ArrayView2<f64>,
        k: usize,
    ) -> f64 {
        let all_rows: Vec<usize> = (0..x.nrows()).collect();
        trustworthiness_brute_force_subsampled(x, y, k, &all_rows)
    }

    #[test]
    fn t_tw_08_combined_matches_baseline() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(123);
        let n = 50;
        let x = ndarray::Array2::from_shape_fn((n, 6), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());

        for k in [3, 7] {
            let t = trustworthiness(x.view(), y.view(), k);
            let t_ref = trustworthiness_brute_force(x.view(), y.view(), k);
            assert!(t.is_finite(), "T(k={k}) must be finite, got {t}");
            assert!((0.0..=1.0).contains(&t), "T(k={k}) out of [0,1]: {t}");
            assert!(
                (t - t_ref).abs() < 1e-12,
                "T(k={k}) combined={t} diverges from brute-force reference={t_ref}"
            );
        }
    }

    #[test]
    fn t_tw_10_self_exclusion_never_in_knn() {
        // Verify that the self-exclusion (dist_y[i] = INFINITY) prevents point i
        // from appearing in its own k-NN result. Catching a regression where the
        // INFINITY assignment is accidentally dropped.
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let n = 20;
        let k = 3;
        let x = ndarray::Array2::from_shape_fn((n, 4), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        // Run trustworthiness — internally computes k-NN of Y for each row.
        // We verify correctness by comparing against brute-force which also applies self-exclusion.
        let t = trustworthiness(x.view(), y.view(), k);
        let t_ref = trustworthiness_brute_force(x.view(), y.view(), k);
        assert!(
            (t - t_ref).abs() < 1e-12,
            "self-exclusion regression: combined={t} diverges from brute-force={t_ref}"
        );
        // Additionally, verify directly via brute-force that no point is its own neighbor.
        for i in 0..n {
            let yi = y.row(i);
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = yi
                        .iter()
                        .zip(y.row(j).iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            let knn: Vec<usize> = dists[..k].iter().map(|&(_, j)| j).collect();
            assert!(!knn.contains(&i), "point {i} appeared in its own k-NN");
        }
    }

    #[test]
    fn t_tw_11_column_major_x_takes_the_scalar_path() {
        // d_x = 12 meets the SIMD width threshold: a column-major x must fall back to
        // the scalar path instead of panicking, and score like its C-order copy.
        use ndarray::ShapeBuilder;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(4242);
        let n = 40;
        let k = 5;
        let x_f = ndarray::Array2::from_shape_fn((n, 12).f(), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        assert!(!x_f.is_standard_layout());

        let t = trustworthiness(x_f.view(), y.view(), k);
        let t_ref = trustworthiness_brute_force(x_f.view(), y.view(), k);
        let t_c_order = trustworthiness(x_f.as_standard_layout().view(), y.view(), k);
        assert!(
            (t - t_ref).abs() < 1e-12,
            "column-major x: T={t} diverges from brute-force reference={t_ref}"
        );
        assert!(
            (t - t_c_order).abs() < 1e-12,
            "column-major x: T={t} diverges from its C-order copy={t_c_order}"
        );
    }

    /// Brute-force O(n²) reference over the given query rows: full-sort X distances,
    /// sort-based Y-kNN.
    fn trustworthiness_brute_force_subsampled(
        x: ndarray::ArrayView2<f64>,
        y: ndarray::ArrayView2<f64>,
        k: usize,
        query_indices: &[usize],
    ) -> f64 {
        use std::collections::HashSet;
        let n = x.nrows();
        let m = query_indices.len();
        let penalty_sum: f64 = query_indices
            .iter()
            .map(|&i| {
                let xi = x.row(i);
                let yi = y.row(i);
                let mut dist_x: Vec<(f64, usize)> = (0..n)
                    .map(|j| {
                        let d: f64 = xi
                            .iter()
                            .zip(x.row(j).iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        (d, j)
                    })
                    .collect();
                dist_x.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let knn_x: HashSet<usize> = dist_x[1..=k].iter().map(|&(_, j)| j).collect();
                let mut rank_x = vec![0usize; n];
                for (rank, &(_, j)) in dist_x.iter().enumerate() {
                    rank_x[j] = rank;
                }
                let mut dist_y: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d: f64 = yi
                            .iter()
                            .zip(y.row(j).iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        (d, j)
                    })
                    .collect();
                dist_y.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let knn_y: HashSet<usize> = dist_y[..k].iter().map(|&(_, j)| j).collect();
                let mut row_penalty = 0u64;
                for j in &knn_y {
                    if !knn_x.contains(j) {
                        row_penalty += (rank_x[*j] - k) as u64;
                    }
                }
                row_penalty as f64
            })
            .sum();
        let denom = m as f64 * k as f64 * (2 * n).saturating_sub(3 * k + 1) as f64;
        1.0 - penalty_sum * 2.0 / denom
    }

    #[test]
    fn t_tws_01_m_equals_n_matches_full_trustworthiness() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(2026);
        let n = 60;
        let k = 5;
        let x = ndarray::Array2::from_shape_fn((n, 8), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        let t_full = trustworthiness(x.view(), y.view(), k);
        let t_sub = trustworthiness_subsampled(x.view(), y.view(), k, n, 0);
        assert!(
            (t_full - t_sub).abs() < 1e-10,
            "m==n parity: full={t_full}, subsampled={t_sub}"
        );
    }

    #[test]
    fn t_tws_02_m_less_than_n_in_unit_interval() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(33);
        let n = 200;
        let k = 5;
        let x = ndarray::Array2::from_shape_fn((n, 10), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());

        // Hard-coded query indices cross-checked against brute force: guards
        // against constant-output regressions and is independent of the
        // production sampling implementation.
        let query_indices: Vec<usize> = vec![3, 17, 29, 42, 58, 71, 89, 104, 127, 145, 161, 188];
        let t = trustworthiness_inner(x.view(), y.view(), k, Some(&query_indices));
        let t_ref = trustworthiness_brute_force_subsampled(x.view(), y.view(), k, &query_indices);
        assert!(
            (t - t_ref).abs() < 1e-12,
            "subsampled kernel vs brute force mismatch: inner={t}, brute={t_ref}"
        );
        assert!(t.is_finite(), "T must be finite, got {t}");
        assert!(t > 0.0 && t <= 1.0, "T out of (0,1]: {t}");
    }

    #[test]
    fn t_tws_03_same_seed_is_deterministic() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let n = 100;
        let k = 4;
        let m = 30;
        let x = ndarray::Array2::from_shape_fn((n, 6), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        let a = trustworthiness_subsampled(x.view(), y.view(), k, m, 12345);
        let b = trustworthiness_subsampled(x.view(), y.view(), k, m, 12345);
        assert_eq!(a, b, "same seed must produce identical result");
    }

    #[test]
    fn t_tws_04_different_seeds_sample_different_query_rows() {
        // The metric value of two samples can coincide (penalty sums are integers), so
        // the seed contract is checked on the sampled rows themselves.
        assert_ne!(
            sample_query_indices(100, 20, 1),
            sample_query_indices(100, 20, 2),
            "different seeds must produce different sampled index sets"
        );
    }

    #[test]
    #[should_panic(expected = "m must be <= n")]
    fn t_tws_05_panics_when_m_exceeds_n() {
        let n = 20usize;
        let x = ndarray::Array2::zeros((n, 4));
        let y = ndarray::Array2::zeros((n, 2));
        let _ = trustworthiness_subsampled(x.view(), y.view(), 3, n + 1, 0);
    }

    #[test]
    #[should_panic(expected = "m must be > 0")]
    fn t_tws_06_panics_when_m_zero() {
        let n = 20usize;
        let x = ndarray::Array2::zeros((n, 4));
        let y = ndarray::Array2::zeros((n, 2));
        let _ = trustworthiness_subsampled(x.view(), y.view(), 3, 0, 0);
    }

    #[test]
    fn t_tws_07_perfect_preservation_subsampled() {
        let n = 40;
        let x = ndarray::Array2::from_shape_fn((n, 4), |(i, d)| (i * 4 + d) as f64);
        let y = x.slice(ndarray::s![.., ..2]).to_owned();
        let t = trustworthiness_subsampled(x.view(), y.view(), 5, 10, 0);
        assert!(
            (t - 1.0).abs() < 1e-12,
            "perfect preservation: T={t}, expected 1.0"
        );
    }

    #[test]
    fn t_tws_08_brute_force_reference_matches() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7777);
        let n = 80;
        let k = 4;
        let x = ndarray::Array2::from_shape_fn((n, 6), |_| rng.random::<f64>());
        let y = ndarray::Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());

        // Hard-coded query indices decouple this correctness test from the
        // production sampling implementation (RNG choice, call order).
        let query_indices: Vec<usize> = vec![
            2, 7, 13, 19, 23, 28, 34, 41, 46, 52, 55, 58, 61, 64, 67, 70, 73, 75, 77, 79,
        ];

        let t_kernel = trustworthiness_inner(x.view(), y.view(), k, Some(&query_indices));
        let t_ref = trustworthiness_brute_force_subsampled(x.view(), y.view(), k, &query_indices);
        assert!(
            (t_kernel - t_ref).abs() < 1e-12,
            "subsampled brute-force mismatch: kernel={t_kernel}, ref={t_ref}"
        );
    }
}
