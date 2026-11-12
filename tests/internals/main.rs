//! Integration tests of the crate's internals, reached through the `__internal` feature.

#![cfg(feature = "__internal")]

#[macro_use]
#[path = "../common/assertions.rs"]
mod assertions;
#[path = "../common/diagnostics.rs"]
mod diagnostics;
#[path = "../common/fixtures.rs"]
mod fixtures;
#[path = "../common/internals.rs"]
mod internals;

mod accuracy_report;
mod adversarial_graphs;
mod comp_a_degrees;
mod comp_b_laplacian;
mod comp_c_components;
mod comp_d_dense_evd;
mod comp_d_rsvd;
mod comp_e_selection;
mod comp_f_lobpcg;
mod comp_g_sinv;
mod diagnostic_metrics;
mod e2e_residual;
mod escalation;
mod laplacian;
mod no_panic;
mod operator_fixtures;
mod property_tests;
mod psd_contract;
mod scaling_fixtures;
mod simd;
mod solver_invariants;
mod warm_restart;
