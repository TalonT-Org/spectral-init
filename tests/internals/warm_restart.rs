//! Warm-restart behaviour of the LOBPCG solver on ill-conditioned Laplacians.
//!
//! The restart counts and residuals reached by these chaotic iterations can move at
//! roundoff level between platforms and builds, so the assertions here pin only the
//! chain's gates.

use crate::{fixtures, internals};
use spectral_init::__internal::{
    CsrOperator, LOBPCG_QUALITY_THRESHOLD, lobpcg_solve, lobpcg_solve_with_restarts,
};

#[test]
fn test_path_2000_warm_restart_majority_converged() {
    // With the 1000-iteration cap on warm-restart passes, path_2000 should converge
    // on a majority of seeds.
    let lap = fixtures::path_laplacian(2000);
    let sqrt_deg = fixtures::path_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    let n_components = 2;

    let mut n_converged = 0usize;
    for seed in [42u64, 43, 44] {
        if let Some(((eigs, vecs), _)) = lobpcg_solve(&op, n_components, seed, false, &sqrt_deg) {
            let residuals = (0..eigs.len())
                .map(|i| internals::eigenpair_residual(&op, vecs.column(i), eigs[i]))
                .collect::<Vec<_>>();
            if residuals.iter().all(|&r| r < 1e-5) {
                n_converged += 1;
            }
        }
    }
    assert!(
        n_converged >= 2,
        "path_2000 should converge on >= 2/3 seeds with raised restart cap; got {n_converged}/3"
    );
}

#[test]
fn test_ring_2000_warm_restart_meets_the_level_1_gate() {
    // ring_2000 seed=42: the first LOBPCG pass ends at max_residual 3.25e-4, sixteen times
    // the chain's Level 1 gate. The warm restarts bring it to 1.09e-5 (the four passes end
    // at 3.25e-4, 1.71e-5, 1.25e-5, 1.09e-5), which the gate accepts; a solver that stopped
    // after the first pass would be escalated to Level 2.
    let lap = fixtures::ring_laplacian(2000);
    let sqrt_deg = fixtures::ring_sqrt_deg(2000);
    let op = CsrOperator(&lap);

    let ((eigs, vecs), restart_count) = lobpcg_solve(&op, 2, 42, false, &sqrt_deg)
        .expect("lobpcg_solve returned None for ring_2000 seed=42");
    assert!(
        restart_count > 0,
        "ring_2000 seed=42 must need a warm restart; got restart_count={restart_count}"
    );

    let max_res = internals::max_operator_residual(&op, &eigs, &vecs);
    assert!(
        max_res < LOBPCG_QUALITY_THRESHOLD,
        "ring_2000 seed=42 must pass the Level 1 gate ({LOBPCG_QUALITY_THRESHOLD:.0e}) after \
         its warm restarts; got max_residual={max_res:.2e}"
    );
}

#[test]
fn test_ring_2000_first_pass_alone_misses_the_level_1_gate() {
    // The counterpart of the test above: with no warm restarts allowed, the same solve
    // returns its single unconverged pass (restart_count 1), which the gate rejects.
    let lap = fixtures::ring_laplacian(2000);
    let sqrt_deg = fixtures::ring_sqrt_deg(2000);
    let op = CsrOperator(&lap);

    let ((eigs, vecs), restart_count) = lobpcg_solve_with_restarts(&op, 2, 42, false, &sqrt_deg, 0)
        .expect("lobpcg_solve_with_restarts returned None for ring_2000 seed=42");
    assert_eq!(
        restart_count, 1,
        "one unconverged pass must be reported as restart_count=1"
    );

    let max_res = internals::max_operator_residual(&op, &eigs, &vecs);
    assert!(
        max_res >= LOBPCG_QUALITY_THRESHOLD,
        "ring_2000 seed=42 must miss the Level 1 gate ({LOBPCG_QUALITY_THRESHOLD:.0e}) \
         without warm restarts; got max_residual={max_res:.2e}"
    );
}

#[test]
fn test_path_2000_warm_restart_fires() {
    // path_2000 is a known hard input: high condition number, small spectral gap.
    // restart_count counts every unconverged pass, including the last one, so this
    // shows that a first pass leaves path_2000 unconverged on at least one of seeds
    // 42/43/44. It does not show that the restarts run; the majority-converged test
    // above and test_ring_2000_warm_restart_meets_the_level_1_gate fail without them.
    // NOTE: This assertion is FP-sensitive — LLVM FP reassociation at opt-level=2 can
    // change Gram matrix conditioning, affecting whether Cholesky triggers a restart.
    // Uses total_restarts > 0 across 3 seeds to reduce single-seed fragility.
    let lap = fixtures::path_laplacian(2000);
    let sqrt_deg = fixtures::path_sqrt_deg(2000);
    let op = CsrOperator(&lap);
    let mut total_restarts = 0usize;
    for seed in [42u64, 43, 44] {
        if let Some((_, rc)) = lobpcg_solve(&op, 2, seed, false, &sqrt_deg) {
            total_restarts += rc;
        }
    }
    assert!(
        total_restarts > 0,
        "path_2000: expected warm restart to fire on at least one of seeds 42/43/44 \
         at opt-level=2; got total_restarts={total_restarts}"
    );
}
