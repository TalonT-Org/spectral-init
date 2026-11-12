//! Integration tests of the public API, built with default features.

#[macro_use]
#[path = "../common/assertions.rs"]
mod assertions;
#[path = "../common/fixtures.rs"]
mod fixtures;

mod determinism;
mod e2e_validation;
mod layout;
mod no_panic;
mod npz_loading;
mod pipeline;
mod traits;
mod umap_rs;
mod validation;
