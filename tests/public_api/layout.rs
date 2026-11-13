//! The returned array is always in standard (C-contiguous, row-major) layout.

use crate::fixtures;
use spectral_init::ndarray::Array2;
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralInitConfig, spectral_init};

fn assert_standard_layout(coords: &Array2<f32>, expected_dim: (usize, usize)) {
    assert_eq!(coords.dim(), expected_dim);
    assert!(
        coords.is_standard_layout(),
        "array is not in standard layout"
    );
    let slice = coords.as_slice().expect("as_slice() must succeed");
    // Row-major: element (i, j) sits at i * n_components + j.
    let n_components = expected_dim.1;
    for ((i, j), &value) in coords.indexed_iter() {
        assert_eq!(slice[i * n_components + j], value);
    }
}

#[test]
fn connected_graph_output_is_standard_layout() {
    let graph = fixtures::make_ring(40);
    for n_components in [2usize, 3] {
        let coords = spectral_init(
            graph.view(),
            None,
            &SpectralInitConfig::new(n_components).with_seed(42),
        )
        .expect("ring graph");
        assert_standard_layout(&coords, (40, n_components));
    }
}

#[test]
fn disconnected_graph_output_is_standard_layout() {
    // Two rings of 12 nodes each.
    let mut tri: TriMatI<f32, u32> = TriMatI::new((24, 24));
    for base in [0usize, 12] {
        for i in 0..12 {
            let (a, b) = (base + i, base + (i + 1) % 12);
            tri.add_triplet(a, b, 1.0);
            tri.add_triplet(b, a, 1.0);
        }
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();
    for n_components in [2usize, 3] {
        let coords = spectral_init(
            graph.view(),
            None,
            &SpectralInitConfig::new(n_components).with_seed(42),
        )
        .expect("two-ring graph");
        assert_standard_layout(&coords, (24, n_components));
    }
}
