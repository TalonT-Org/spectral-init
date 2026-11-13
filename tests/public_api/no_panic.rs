//! No input makes `spectral_init` panic: every call returns `Ok` or `Err`.

use crate::assertions::property_config;
use proptest::prelude::*;
use spectral_init::ndarray::Array2;
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{ComputeMode, SpectralInitConfig, spectral_init};

/// An arbitrary input to `spectral_init`.
#[derive(Debug, Clone)]
struct Case {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, f32)>,
    mirror: bool,
    csc: bool,
    data: Option<(usize, usize, Vec<f32>)>,
    n_components: usize,
    seed: u64,
    rust_native: bool,
    dense_memory_budget: u64,
}

impl Case {
    fn graph(&self) -> CsMatI<f32, u32, usize> {
        let mut tri: TriMatI<f32, u32> = TriMatI::new((self.rows, self.cols));
        for &(row, col, weight) in &self.entries {
            if row < self.rows && col < self.cols {
                tri.add_triplet(row, col, weight);
            }
            if self.mirror && row != col && col < self.rows && row < self.cols {
                tri.add_triplet(col, row, weight);
            }
        }
        if self.csc { tri.to_csc() } else { tri.to_csr() }
    }

    fn data(&self) -> Option<Array2<f32>> {
        self.data.as_ref().map(|(rows, cols, values)| {
            Array2::from_shape_vec((*rows, *cols), values.clone()).expect("rows * cols values")
        })
    }

    fn spectral_config(&self) -> SpectralInitConfig {
        SpectralInitConfig::new(self.n_components)
            .with_seed(self.seed)
            .with_compute_mode(if self.rust_native {
                ComputeMode::RustNative
            } else {
                ComputeMode::PythonCompat
            })
            .with_dense_memory_budget(self.dense_memory_budget)
    }
}

/// Weights a well-formed graph carries.
fn clean_weight() -> BoxedStrategy<f32> {
    (0.05f32..4.0).boxed()
}

/// Weights of every kind, valid or not.
fn any_weight() -> BoxedStrategy<f32> {
    prop_oneof![
        6 => 0.05f32..4.0,
        1 => Just(0.0f32),
        1 => Just(-0.0f32),
        1 => Just(-1.5f32),
        1 => Just(f32::NAN),
        1 => Just(f32::INFINITY),
        1 => Just(f32::MAX),
        1 => Just(f32::MIN_POSITIVE),
    ]
    .boxed()
}

fn data_value() -> BoxedStrategy<f32> {
    prop_oneof![
        8 => -100.0f32..100.0,
        1 => Just(f32::NAN),
        1 => Just(f32::NEG_INFINITY),
    ]
    .boxed()
}

fn budget() -> BoxedStrategy<u64> {
    prop_oneof![
        Just(0u64),
        Just(64u64),
        Just(512u64),
        Just(4096u64),
        Just(SpectralInitConfig::DEFAULT_DENSE_MEMORY_BUDGET),
    ]
    .boxed()
}

fn case() -> impl Strategy<Value = Case> {
    // `clean` cases are square, symmetric and positively weighted, so that the solvers
    // run; the others exercise the validation boundary.
    (0usize..=9, 0usize..=9, any::<bool>()).prop_flat_map(|(rows, other, clean)| {
        let cols = if clean || other % 3 != 0 { rows } else { other };
        let weight = if clean { clean_weight() } else { any_weight() };
        let entries = proptest::collection::vec((0..rows.max(1), 0..cols.max(1), weight), 0..=28);
        let data = proptest::option::of(
            (prop_oneof![3 => Just(rows), 1 => 0usize..=10], 0usize..=3).prop_flat_map(
                |(data_rows, data_cols)| {
                    (
                        Just(data_rows),
                        Just(data_cols),
                        proptest::collection::vec(data_value(), data_rows * data_cols),
                    )
                },
            ),
        );
        (
            entries,
            any::<bool>(),
            any::<bool>(),
            data,
            0usize..=4,
            any::<u64>(),
            any::<bool>(),
            budget(),
        )
            .prop_map(
                move |(entries, mirror, csc, data, n_components, seed, rust_native, budget)| Case {
                    rows,
                    cols,
                    entries,
                    mirror: clean || mirror,
                    csc,
                    data,
                    n_components,
                    seed,
                    rust_native,
                    dense_memory_budget: budget,
                },
            )
    })
}

proptest! {
    #![proptest_config(property_config(512, 256))]

    #[test]
    fn spectral_init_never_panics(case in case()) {
        let graph = case.graph();
        let data = case.data();
        let result = spectral_init(
            graph.view(),
            data.as_ref().map(|d| d.view()),
            &case.spectral_config(),
        );
        if let Ok(coords) = result {
            prop_assert_eq!(coords.dim(), (case.rows, case.n_components));
            prop_assert!(coords.is_standard_layout());
            prop_assert!(coords.iter().all(|v| v.is_finite()));
        }
    }
}
