//! No valid Laplacian makes the solver chain panic: with the dense first level switched
//! off, every later level runs and the chain still returns `Ok` or `Err`.

use crate::assertions::validate_graph;
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use spectral_init::__internal::{
    SolverParams, build_normalized_laplacian, compute_degrees, solve_eigenproblem,
};
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{ComputeMode, SpectralInitConfig};

/// Failing cases persist to `no_panic.proptest-regressions`, next to this file.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        max_shrink_iters: 256,
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..Default::default()
    }
}

/// A symmetric, positively weighted graph (self-loops allowed) and a solve request.
#[derive(Debug, Clone)]
struct Case {
    n: usize,
    edges: Vec<(usize, usize, f32)>,
    n_components: usize,
    seed: u64,
    rust_native: bool,
    dense_memory_budget: u64,
}

impl Case {
    fn graph(&self) -> CsMatI<f32, u32, usize> {
        let mut tri: TriMatI<f32, u32> = TriMatI::new((self.n, self.n));
        for &(row, col, weight) in &self.edges {
            tri.add_triplet(row, col, weight);
            if row != col {
                tri.add_triplet(col, row, weight);
            }
        }
        tri.to_csr()
    }

    fn compute_mode(&self) -> ComputeMode {
        if self.rust_native {
            ComputeMode::RustNative
        } else {
            ComputeMode::PythonCompat
        }
    }
}

fn case() -> impl Strategy<Value = Case> {
    (2usize..=14).prop_flat_map(|n| {
        (
            proptest::collection::vec((0..n, 0..n, 0.05f32..4.0), 0..=40),
            // Up to and beyond the largest solvable request (n - 1).
            0..=n + 1,
            any::<u64>(),
            any::<bool>(),
            prop_oneof![
                Just(0u64),
                Just(256u64),
                Just(SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET),
            ],
        )
            .prop_map(
                move |(edges, n_components, seed, rust_native, dense_memory_budget)| Case {
                    n,
                    edges,
                    n_components,
                    seed,
                    rust_native,
                    dense_memory_budget,
                },
            )
    })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn solver_chain_never_panics(case in case()) {
        let graph = case.graph();
        prop_assert_eq!(validate_graph(&graph), Ok(()));

        let mode = case.compute_mode();
        let (_degrees, sqrt_deg) = compute_degrees(graph.view(), mode);
        let inv_sqrt_deg: Vec<f64> = sqrt_deg
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        let laplacian = build_normalized_laplacian(graph.view(), &inv_sqrt_deg);

        let params = SolverParams {
            dense_n_threshold: 0,
            dense_memory_budget: case.dense_memory_budget,
            ..SolverParams::default()
        };
        let result = solve_eigenproblem(
            &laplacian,
            case.n_components,
            case.seed,
            &sqrt_deg,
            mode,
            &params,
        );
        if let Ok(((eigenvalues, eigenvectors), _report)) = result {
            prop_assert_eq!(eigenvalues.len(), case.n_components + 1);
            prop_assert_eq!(eigenvectors.dim(), (case.n, case.n_components + 1));
            prop_assert!(eigenvalues.iter().all(|&v| (0.0..=2.0).contains(&v)));
            prop_assert!(eigenvectors.iter().all(|v| v.is_finite()));
        }
    }
}
