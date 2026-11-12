//! Escalation and the dense memory budget: which level answers, and which error is
//! returned when none can.

use crate::fixtures;
use ndarray::{Array1, Array2};
use spectral_init::__internal::{
    QualityGates, SolveReport, SolverLevel, SolverParams, solve_eigenproblem,
};
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{ComputeMode, SpectralError, SpectralInitConfig, spectral_init};

/// Size of the ring Laplacian the solver tests run on.
const N: usize = 100;

/// Bytes of the dense `N × N` matrix Levels 0 and 5 need.
const DENSE_BYTES: u64 = (N * N * 8) as u64;

/// Bytes of the dense `N × k` sketch Level 4 needs for two components: rank 3 plus the
/// default oversampling of `N / 10`.
const SKETCH_BYTES: u64 = (N * (3 + N / 10) * 8) as u64;

/// Gates that no iterative result can pass (`residual < 0.0` never holds).
fn failing_iterative_gates() -> QualityGates {
    QualityGates {
        lobpcg: 0.0,
        sinv: 0.0,
        rsvd: 0.0,
        ..QualityGates::default()
    }
}

/// Gates that Levels 1–3 cannot pass and any Level 4 (rSVD) result does.
fn rsvd_only_gates() -> QualityGates {
    QualityGates {
        rsvd: f64::INFINITY,
        ..failing_iterative_gates()
    }
}

/// Asserts that `result` is `DenseBudgetExceeded` reporting exactly these sizes.
fn assert_budget_exceeded<T: std::fmt::Debug>(
    result: Result<T, SpectralError>,
    required: u64,
    budget: u64,
) {
    match result {
        Err(SpectralError::DenseBudgetExceeded {
            required_bytes,
            budget_bytes,
            ..
        }) => assert_eq!((required_bytes, budget_bytes), (required, budget)),
        other => panic!("expected DenseBudgetExceeded, got {other:?}"),
    }
}

fn solve(params: &SolverParams) -> Result<SolveReport, SpectralError> {
    let laplacian = fixtures::ring_laplacian(N);
    let sqrt_deg: Array1<f64> = fixtures::ring_sqrt_deg(N);
    solve_eigenproblem(
        &laplacian,
        2,
        42,
        &sqrt_deg,
        ComputeMode::PythonCompat,
        params,
    )
    .map(|(_eigenpairs, report)| report)
}

#[test]
fn forced_dense_answers_when_every_other_level_fails() {
    let params = SolverParams {
        dense_n_threshold: 0,
        gates: failing_iterative_gates(),
        dense_memory_budget: DENSE_BYTES,
        ..SolverParams::default()
    };
    let report = solve(&params).expect("Level 5 fits the budget exactly");
    assert_eq!(report.level, SolverLevel::ForcedDense);
    assert!(report.residual < params.gates.dense);
    assert!(report.level_elapsed[SolverLevel::DenseEvd as usize].is_none());
    for (level, elapsed) in report.level_elapsed.iter().enumerate().skip(1) {
        assert!(elapsed.is_some(), "level {level} must have run");
    }
}

#[test]
fn forced_dense_over_budget_is_an_error() {
    let params = SolverParams {
        dense_n_threshold: 0,
        gates: failing_iterative_gates(),
        dense_memory_budget: DENSE_BYTES - 1,
        ..SolverParams::default()
    };
    assert_budget_exceeded(solve(&params), DENSE_BYTES, DENSE_BYTES - 1);
}

#[test]
fn forced_dense_result_must_pass_the_dense_gate() {
    let params = SolverParams {
        dense_n_threshold: 0,
        gates: QualityGates {
            dense: 0.0,
            ..failing_iterative_gates()
        },
        dense_memory_budget: DENSE_BYTES,
        ..SolverParams::default()
    };
    assert_err!(
        solve(&params),
        SpectralError::ConvergenceFailure { n: N, .. }
    );
}

#[test]
fn level_0_over_budget_escalates_to_an_iterative_level() {
    // n = 100 is below the dense threshold, so Level 0 is tried first.
    let within = SolverParams::default();
    assert_eq!(solve(&within).unwrap().level, SolverLevel::DenseEvd);

    let over = SolverParams {
        dense_memory_budget: DENSE_BYTES - 1,
        ..SolverParams::default()
    };
    let report = solve(&over).expect("an iterative level converges on the ring");
    assert!(
        report.level > SolverLevel::DenseEvd && report.level < SolverLevel::ForcedDense,
        "expected an iterative level, got {:?}",
        report.level
    );
}

#[test]
fn level_4_runs_within_budget_and_is_skipped_over_it() {
    // Levels 1–3 are made to fail; any Level 4 result is accepted.
    let within = SolverParams {
        dense_n_threshold: 0,
        gates: rsvd_only_gates(),
        dense_memory_budget: SKETCH_BYTES,
        ..SolverParams::default()
    };
    assert_eq!(solve(&within).unwrap().level, SolverLevel::Rsvd);

    // One byte less: Level 4 is skipped, and Level 5 needs even more.
    let over = SolverParams {
        dense_memory_budget: SKETCH_BYTES - 1,
        ..within
    };
    assert_budget_exceeded(solve(&over), DENSE_BYTES, SKETCH_BYTES - 1);
}

// ── the budget of `SpectralInitConfig` reaches every dense allocation ────────

fn params_with_config_budget(budget: u64) -> SolverParams {
    let config = SpectralInitConfig::new(2).with_dense_memory_budget(budget);
    let params = SolverParams::from_config(&config);
    assert_eq!(params.dense_memory_budget, budget);
    params
}

#[test]
fn config_budget_reaches_level_0() {
    let report = solve(&params_with_config_budget(DENSE_BYTES - 1)).unwrap();
    assert_ne!(report.level, SolverLevel::DenseEvd);
    let report = solve(&params_with_config_budget(DENSE_BYTES)).unwrap();
    assert_eq!(report.level, SolverLevel::DenseEvd);
}

#[test]
fn config_budget_reaches_level_4() {
    let with_budget = |budget| SolverParams {
        dense_n_threshold: 0,
        gates: rsvd_only_gates(),
        ..params_with_config_budget(budget)
    };
    assert_eq!(
        solve(&with_budget(SKETCH_BYTES)).unwrap().level,
        SolverLevel::Rsvd
    );
    assert!(matches!(
        solve(&with_budget(SKETCH_BYTES - 1)),
        Err(SpectralError::DenseBudgetExceeded { .. })
    ));
}

#[test]
fn config_budget_reaches_level_5() {
    let with_budget = |budget| SolverParams {
        dense_n_threshold: 0,
        gates: failing_iterative_gates(),
        ..params_with_config_budget(budget)
    };
    assert_eq!(
        solve(&with_budget(DENSE_BYTES)).unwrap().level,
        SolverLevel::ForcedDense
    );
    assert_budget_exceeded(
        solve(&with_budget(DENSE_BYTES - 1)),
        DENSE_BYTES,
        DENSE_BYTES - 1,
    );
}

/// Six separate triangles with well-separated data: more components than
/// `2 · n_components`, so the component centroids get a spectral embedding of their own.
fn six_triangles() -> (CsMatI<f32, u32, usize>, Array2<f32>) {
    let mut tri: TriMatI<f32, u32> = TriMatI::new((18, 18));
    for component in 0..6usize {
        let base = 3 * component;
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            tri.add_triplet(base + a, base + b, 1.0);
            tri.add_triplet(base + b, base + a, 1.0);
        }
    }
    let data = Array2::from_shape_fn((18, 2), |(node, feature)| {
        (node / 3) as f32 * 0.5 + feature as f32 * 0.1
    });
    (tri.to_csr(), data)
}

#[test]
fn config_budget_reaches_the_centroid_affinity() {
    let (graph, data) = six_triangles();
    // The affinity between six component centroids is a dense 6 × 6 matrix: 288 bytes.
    let config = SpectralInitConfig::new(2).with_dense_memory_budget(288);
    let coords = spectral_init(graph.view(), Some(data.view()), &config)
        .expect("the affinity matrix fits the budget exactly");
    assert_eq!(coords.dim(), (18, 2));

    let config = config.with_dense_memory_budget(287);
    assert_budget_exceeded(
        spectral_init(graph.view(), Some(data.view()), &config),
        288,
        287,
    );
}
