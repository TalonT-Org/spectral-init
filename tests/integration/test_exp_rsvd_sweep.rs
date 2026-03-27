//! RQ1 oversampling sweep tests.
//!
//! `accurate_sweep` — loops over p values using the high-accuracy rSVD variant (nbiter=6).
//! `production_sweep` — runs a single p value (set via SPECTRAL_RSVD_OVERSAMPLING) using
//!   the production rSVD variant (nbiter=2, reads env-var seam internally).
//!
//! Both tests emit `SWEEP_ROW,<fixture>,<n>,<p>,<k>,<residual>,<wall_time_s>,<passes>`
//! lines to stdout for collection by the outer shell scripts.

#[path = "../common/mod.rs"]
mod common;

const FIXTURES: &[&str] = &[
    "near_dupes_100",
    "blobs_connected_200",
    "moons_200",
    "circles_300",
    "blobs_500",
    "blobs_connected_2000",
];

const N_COMPONENTS: usize = 2;
const SEED: u64 = 42;
const P_VALUES_ACCURATE: &[usize] = &[5, 10, 15, 20, 25, 30, 50, 100];

#[test]
fn accurate_sweep() {
    for &fixture in FIXTURES {
        let path = common::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = common::load_sparse_csr(&path);
        let n = laplacian.rows();

        for &p in P_VALUES_ACCURATE {
            let k_sub = N_COMPONENTS + 1 + p; // rank=3, so k_sub = 3 + p
            let t0 = std::time::Instant::now();
            let (eigs, vecs) =
                spectral_init::rsvd_solve_accurate(&laplacian, N_COMPONENTS, SEED, k_sub);
            let wall_time = t0.elapsed().as_secs_f64();

            // max residual over non-trivial pairs (indices 1..=n_components)
            let max_res = (1..=N_COMPONENTS)
                .map(|i| common::residual_spmv(&laplacian, vecs.column(i), eigs[i]))
                .fold(0.0_f64, f64::max);

            let passes = max_res < 1e-2;
            let k = k_sub;
            println!("SWEEP_ROW,{fixture},{n},{p},{k},{max_res:.6e},{wall_time:.3},{passes}");
        }
    }
}

#[test]
fn production_sweep() {
    // Read p from env-var once for CSV reporting; rsvd_solve reads it internally.
    let env_p: Option<usize> = std::env::var("SPECTRAL_RSVD_OVERSAMPLING")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    for &fixture in FIXTURES {
        let path = common::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = common::load_sparse_csr(&path);
        let n = laplacian.rows();
        let rank = N_COMPONENTS + 1; // 3

        // Mirror rsvd_k_sub_effective logic for CSV reporting.
        let (p, k) = if let Some(p_override) = env_p {
            let k = (rank + p_override).min(n);
            (p_override, k)
        } else {
            let oversampling = (n / 10).max(rank.max(5)).min(n.saturating_sub(rank));
            let k = (rank + oversampling).min(n);
            (oversampling, k)
        };

        let t0 = std::time::Instant::now();
        // rsvd_solve (production path: nbiter=2) reads SPECTRAL_RSVD_OVERSAMPLING internally.
        let (eigs, vecs) = spectral_init::rsvd_solve(&laplacian, N_COMPONENTS, SEED);
        let wall_time = t0.elapsed().as_secs_f64();

        let max_res = (1..=N_COMPONENTS)
            .map(|i| common::residual_spmv(&laplacian, vecs.column(i), eigs[i]))
            .fold(0.0_f64, f64::max);

        let passes = max_res < 1e-2;
        println!("SWEEP_ROW,{fixture},{n},{p},{k},{max_res:.6e},{wall_time:.3},{passes}");
    }
}
