//! rSVD power-iteration (nbiter) sweep tests — experiment groupB.
//!
//! `direction_a_2il_vs_direct_l` — small grid on blobs_connected_2000; compares 2I-L
//!   vs direct-L projection methods across nbiter ∈ {2,4,6} × p ∈ {30,100}.
//!   Emits `SWEEP_ROW_A,{fixture},{method},{nbiter},{p},{residual},{wall_time_us}`.
//!
//! `direction_b_sweep` — full grid on both fixtures across
//!   nbiter ∈ {2,3,4,5,6,8,10} × p ∈ {20,30,50,100}.
//!   Emits `SWEEP_ROW_B,{fixture},{method},{nbiter},{p},{residual},{ortho_error},{wall_time_us},{gap},{passes}`.
//!
//! `direction_c_subspace` — subspace quality for passing (residual < 1e-2) combos from
//!   the direction_b grid; reference is rsvd_solve_accurate at nbiter=10, p=100.
//!   Emits `QUALITY_ROW_C,{fixture},{method},{nbiter},{p},{gram_det},{sign_error}`.
//!
//! Run with --test-threads=1 for reproducible wall-time measurements and safe env-var
//! manipulation in the 2il dispatch path.

#[path = "../common/mod.rs"]
mod common;

use ndarray::s;
use spectral_init::metrics::{
    max_eigenpair_residual, orthogonality_error, sign_agnostic_max_error, spectral_gap,
    subspace_gram_det_kd, RSVD_QUALITY_THRESHOLD,
};

const SEED: u64 = 42;
const FIXTURES_AB: &[&str] = &["blobs_connected_2000", "blobs_5000"];
const METHODS: &[&str] = &["2il", "direct_l"];
const NBITER_B: &[usize] = &[2, 3, 4, 5, 6, 8, 10];
const P_B: &[usize] = &[20, 30, 50, 100];

fn load_fixture(fixture: &str) -> (sprs::CsMat<f64>, usize, usize) {
    let laplacian = common::load_sparse_csr(&common::fixture_path(fixture, "comp_b_laplacian.npz"));
    let n = laplacian.rows();
    let ev_path = common::fixture_path(fixture, "comp_d_eigensolver.npz");
    let mut npz = ndarray_npy::NpzReader::new(
        std::fs::File::open(&ev_path)
            .unwrap_or_else(|e| panic!("cannot open {:?}: {}", ev_path, e)),
    )
    .unwrap();
    let k: i32 = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("k")
        .unwrap_or_else(|e| panic!("key 'k' not found in {:?}: {}", ev_path, e))
        .into_scalar();
    let nc = (k as usize) - 1;
    (laplacian, n, nc)
}

fn run_solver(
    lap: &sprs::CsMat<f64>,
    nc: usize,
    n: usize,
    method: &str,
    nbiter: usize,
    p: usize,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    let k_sub = (nc + 1 + p).min(n);
    match method {
        "2il" => {
            // SAFETY: all callers run with --test-threads=1; no concurrent env readers.
            unsafe {
                std::env::set_var("SPECTRAL_RSVD_NBITER", nbiter.to_string());
                std::env::set_var("SPECTRAL_RSVD_OVERSAMPLING", p.to_string());
            }
            let result = spectral_init::rsvd_solve(lap, nc, SEED);
            unsafe {
                std::env::remove_var("SPECTRAL_RSVD_NBITER");
                std::env::remove_var("SPECTRAL_RSVD_OVERSAMPLING");
            }
            result
        }
        "direct_l" => spectral_init::rsvd_solve_accurate(lap, nc, SEED, k_sub, nbiter),
        _ => panic!("unknown method: {method}"),
    }
}

#[test]
fn direction_a_2il_vs_direct_l() {
    let fixture = "blobs_connected_2000";
    let (lap, n, nc) = load_fixture(fixture);

    for &method in METHODS {
        for &nbiter in &[2usize, 4, 6] {
            for &p in &[30usize, 100] {
                let t0 = std::time::Instant::now();
                let (eigs, vecs) = run_solver(&lap, nc, n, method, nbiter, p);
                let wall_time_us = t0.elapsed().as_micros() as u64;

                let nt_eigs = eigs.slice(s![1..]).to_owned();
                let nt_vecs = vecs.slice(s![.., 1..]).to_owned();
                let residual = max_eigenpair_residual(&lap, &nt_eigs, &nt_vecs);

                println!(
                    "SWEEP_ROW_A,{fixture},{method},{nbiter},{p},{residual:.6e},{wall_time_us}"
                );
            }
        }
    }
}

#[test]
fn direction_b_sweep() {
    // Load both fixtures once outside all loops to avoid re-parsing per combo.
    let fixture_data: Vec<_> = FIXTURES_AB
        .iter()
        .map(|&f| (f, load_fixture(f)))
        .collect();

    for (fixture, (lap, n, nc)) in &fixture_data {
        for &method in METHODS {
            for &nbiter in NBITER_B {
                for &p in P_B {
                    let t0 = std::time::Instant::now();
                    let (eigs, vecs) = run_solver(lap, *nc, *n, method, nbiter, p);
                    let wall_time_us = t0.elapsed().as_micros() as u64;

                    let nt_eigs = eigs.slice(s![1..]).to_owned();
                    let nt_vecs = vecs.slice(s![.., 1..]).to_owned();
                    let residual = max_eigenpair_residual(lap, &nt_eigs, &nt_vecs);
                    let ortho_error = orthogonality_error(&nt_vecs);
                    let gap = spectral_gap(&eigs);
                    let passes_gate = residual < RSVD_QUALITY_THRESHOLD;

                    println!(
                        "SWEEP_ROW_B,{fixture},{method},{nbiter},{p},{residual:.6e},{ortho_error:.6e},{wall_time_us},{gap:.6e},{passes_gate}"
                    );
                }
            }
        }
    }
}

#[test]
fn direction_c_subspace() {
    let fixture_data: Vec<_> = FIXTURES_AB
        .iter()
        .map(|&f| (f, load_fixture(f)))
        .collect();

    for (fixture, (lap, n, nc)) in &fixture_data {
        // Reference: high-accuracy solve at nbiter=10, p=100.
        let k_sub_ref = (nc + 1 + 100).min(*n);
        let (_, ref_vecs) = spectral_init::rsvd_solve_accurate(lap, *nc, SEED, k_sub_ref, 10);
        let ref_nt_vecs = ref_vecs.slice(s![.., 1..]).to_owned();
        let ref_f32: ndarray::Array2<f32> = ref_nt_vecs.mapv(|x| x as f32);

        for &method in METHODS {
            for &nbiter in NBITER_B {
                for &p in P_B {
                    let (eigs, vecs) = run_solver(lap, *nc, *n, method, nbiter, p);
                    let nt_eigs = eigs.slice(s![1..]).to_owned();
                    let nt_vecs = vecs.slice(s![.., 1..]).to_owned();
                    let residual = max_eigenpair_residual(lap, &nt_eigs, &nt_vecs);

                    if residual >= RSVD_QUALITY_THRESHOLD {
                        continue; // skip non-passing combos
                    }

                    let gram_det = subspace_gram_det_kd(nt_vecs.view(), ref_nt_vecs.view());
                    let vecs_f32: ndarray::Array2<f32> = nt_vecs.mapv(|x| x as f32);
                    let sign_error = sign_agnostic_max_error(&vecs_f32, &ref_f32);

                    println!(
                        "QUALITY_ROW_C,{fixture},{method},{nbiter},{p},{gram_det:.6e},{sign_error:.6e}"
                    );
                }
            }
        }
    }
}
