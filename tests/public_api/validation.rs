//! Boundary validation: every defect maps to its own error variant, in a fixed order of
//! precedence, and equivalent encodings of a valid graph give the same result.

use crate::fixtures;
use spectral_init::ndarray::Array2;
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{ComputeMode, SpectralError, SpectralInitConfig, spectral_init};

type Graph = CsMatI<f32, u32, usize>;

fn triplets(shape: (usize, usize), entries: &[(usize, usize, f32)]) -> TriMatI<f32, u32> {
    let mut tri = TriMatI::new(shape);
    for &(row, col, weight) in entries {
        tri.add_triplet(row, col, weight);
    }
    tri
}

fn csr(shape: (usize, usize), entries: &[(usize, usize, f32)]) -> Graph {
    triplets(shape, entries).to_csr()
}

fn csc(shape: (usize, usize), entries: &[(usize, usize, f32)]) -> Graph {
    triplets(shape, entries).to_csc()
}

/// Entries of the symmetric matrix with the given upper-triangle edges.
fn symmetric(edges: &[(usize, usize, f32)]) -> Vec<(usize, usize, f32)> {
    edges
        .iter()
        .flat_map(|&(i, j, w)| [(i, j, w), (j, i, w)])
        .collect()
}

/// A valid 5-node path graph 0-1-2-3-4.
fn path_5() -> Vec<(usize, usize, f32)> {
    symmetric(&[(0, 1, 1.0), (1, 2, 0.5), (2, 3, 0.25), (3, 4, 2.0)])
}

fn run(graph: &Graph, n_components: usize) -> Result<Array2<f32>, SpectralError> {
    spectral_init(graph.view(), None, &SpectralInitConfig::new(n_components))
}

/// Outcome of validating `graph` alone: a data argument with no rows turns "the graph is
/// valid" into `DataShapeMismatch`, the last-but-one check, without running a solver.
fn validate_graph(graph: &Graph) -> Result<(), SpectralError> {
    let no_rows = Array2::<f32>::zeros((0, 1));
    match spectral_init(
        graph.view(),
        Some(no_rows.view()),
        &SpectralInitConfig::new(1),
    ) {
        Err(SpectralError::DataShapeMismatch { .. }) => Ok(()),
        Err(other) => Err(other),
        Ok(_) => panic!("a data argument without rows must be rejected"),
    }
}

// ── one defect, one variant ──────────────────────────────────────────────────

#[test]
fn zero_components_is_an_invalid_parameter() {
    let graph = csr((5, 5), &path_5());
    assert!(matches!(
        run(&graph, 0),
        Err(SpectralError::InvalidParameter {
            name: "n_components",
            ..
        })
    ));
}

#[test]
fn non_square_matrix_is_rejected() {
    let graph = csr((3, 4), &[(0, 1, 1.0), (1, 0, 1.0)]);
    assert_err!(
        run(&graph, 2),
        SpectralError::NotSquare {
            rows: 3,
            cols: 4,
            ..
        }
    );
}

#[test]
fn too_few_nodes_covers_the_empty_graph() {
    let empty = csr((0, 0), &[]);
    assert_err!(
        run(&empty, 2),
        SpectralError::TooFewNodes {
            n: 0,
            n_components: 2,
            ..
        }
    );

    // n must exceed n_components: n == n_components is still too few.
    let pair = csr((2, 2), &symmetric(&[(0, 1, 1.0)]));
    assert_err!(
        run(&pair, 2),
        SpectralError::TooFewNodes {
            n: 2,
            n_components: 2,
            ..
        }
    );
}

#[test]
fn non_finite_weights_are_rejected() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut entries = path_5();
        entries.extend([(1, 3, bad), (3, 1, bad)]);
        let graph = csr((5, 5), &entries);
        // Row 1 comes first in CSR order.
        assert_err!(
            run(&graph, 2),
            SpectralError::NonFiniteWeight { row: 1, col: 3, .. },
            "weight {bad}"
        );
    }
}

#[test]
fn negative_weights_are_rejected() {
    let mut entries = path_5();
    entries.extend([(0, 4, -0.5), (4, 0, -0.5)]);
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NegativeWeight { row: 0, col: 4, .. }
    );
}

#[test]
fn negative_zero_is_an_absent_edge_not_a_negative_weight() {
    let mut entries = path_5();
    entries.push((0, 4, -0.0));
    let graph = csr((5, 5), &entries);
    assert!(run(&graph, 2).is_ok());
}

#[test]
fn missing_mirror_entry_is_not_symmetric() {
    let mut entries = path_5();
    entries.push((0, 3, 0.7));
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NotSymmetric { row: 0, col: 3, .. }
    );
}

#[test]
fn unequal_mirror_weights_are_not_symmetric() {
    let mut entries = path_5();
    entries.extend([(0, 3, 0.7), (3, 0, 0.6)]);
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NotSymmetric { row: 0, col: 3, .. }
    );
}

#[test]
fn overflowing_degree_is_rejected() {
    // Node 1 has two edges of weight f32::MAX: their single-precision sum is infinite.
    let entries = symmetric(&[(0, 1, f32::MAX), (1, 2, f32::MAX), (2, 3, 1.0)]);
    let graph = csr((4, 4), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::DegreeOverflow { node: 1, .. }
    );
}

#[test]
fn data_with_the_wrong_row_count_is_rejected() {
    let graph = csr((5, 5), &path_5());
    let data = Array2::<f32>::zeros((4, 3));
    assert_err!(
        spectral_init(graph.view(), Some(data.view()), &SpectralInitConfig::new(2)),
        SpectralError::DataShapeMismatch {
            expected_rows: 5,
            actual_rows: 4,
            ..
        }
    );
}

#[test]
fn non_finite_data_is_rejected_even_when_data_is_unused() {
    // A connected graph never reads `data`, but it is validated all the same.
    let graph = csr((5, 5), &path_5());
    let mut data = Array2::<f32>::zeros((5, 3));
    data[[2, 1]] = f32::NAN;
    assert_err!(
        spectral_init(graph.view(), Some(data.view()), &SpectralInitConfig::new(2)),
        SpectralError::NonFiniteData { row: 2, col: 1, .. }
    );
}

// ── precedence ───────────────────────────────────────────────────────────────

#[test]
fn invalid_parameter_precedes_not_square() {
    let graph = csr((3, 4), &[(0, 1, f32::NAN)]);
    assert!(matches!(
        run(&graph, 0),
        Err(SpectralError::InvalidParameter { .. })
    ));
}

#[test]
fn not_square_precedes_too_few_nodes_and_weight_checks() {
    // 2 × 3 with n_components = 5 and a NaN weight.
    let graph = csr((2, 3), &[(0, 1, f32::NAN)]);
    assert_err!(
        run(&graph, 5),
        SpectralError::NotSquare {
            rows: 2,
            cols: 3,
            ..
        }
    );
}

#[test]
fn too_few_nodes_precedes_weight_checks() {
    let graph = csr((2, 2), &[(0, 1, f32::NAN), (1, 0, -1.0)]);
    assert_err!(
        run(&graph, 2),
        SpectralError::TooFewNodes {
            n: 2,
            n_components: 2,
            ..
        }
    );
}

#[test]
fn non_finite_precedes_negative_wherever_it_is_stored() {
    // The negative weight is stored first (row 0); the NaN sits in the last row.
    let mut entries = path_5();
    entries.extend([(0, 2, -1.0), (2, 0, -1.0), (4, 4, f32::NAN)]);
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NonFiniteWeight { row: 4, col: 4, .. }
    );
}

#[test]
fn nan_on_both_sides_is_non_finite_not_asymmetric() {
    let mut entries = path_5();
    entries.extend([(0, 3, f32::NAN), (3, 0, f32::NAN)]);
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NonFiniteWeight { row: 0, col: 3, .. }
    );
}

#[test]
fn negative_precedes_not_symmetric() {
    let mut entries = path_5();
    entries.extend([(0, 3, 0.7), (4, 2, -0.1)]);
    let graph = csr((5, 5), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NegativeWeight { row: 4, col: 2, .. }
    );
}

#[test]
fn not_symmetric_precedes_degree_overflow() {
    let mut entries = symmetric(&[(0, 1, f32::MAX), (1, 2, f32::MAX), (2, 3, 1.0)]);
    entries.push((3, 0, 0.5));
    let graph = csr((4, 4), &entries);
    assert_err!(
        run(&graph, 2),
        SpectralError::NotSymmetric { row: 3, col: 0, .. }
    );
}

#[test]
fn degree_overflow_precedes_data_checks() {
    let entries = symmetric(&[(0, 1, f32::MAX), (1, 2, f32::MAX), (2, 3, 1.0)]);
    let graph = csr((4, 4), &entries);
    let data = Array2::<f32>::from_elem((9, 2), f32::NAN);
    assert_err!(
        spectral_init(graph.view(), Some(data.view()), &SpectralInitConfig::new(2)),
        SpectralError::DegreeOverflow { node: 1, .. }
    );
}

#[test]
fn data_shape_precedes_data_values() {
    let graph = csr((5, 5), &path_5());
    let data = Array2::<f32>::from_elem((4, 2), f32::INFINITY);
    assert_err!(
        spectral_init(graph.view(), Some(data.view()), &SpectralInitConfig::new(2)),
        SpectralError::DataShapeMismatch {
            expected_rows: 5,
            actual_rows: 4,
            ..
        }
    );
}

// ── caller's orientation ─────────────────────────────────────────────────────

#[test]
fn csc_errors_use_the_callers_row_and_column() {
    let mut entries = path_5();
    entries.push((0, 3, 0.7));
    assert_err!(
        run(&csc((5, 5), &entries), 2),
        SpectralError::NotSymmetric { row: 0, col: 3, .. }
    );

    let mut entries = path_5();
    entries.push((4, 1, f32::NAN));
    assert_err!(
        run(&csc((5, 5), &entries), 2),
        SpectralError::NonFiniteWeight { row: 4, col: 1, .. }
    );
}

// ── equivalent encodings ─────────────────────────────────────────────────────

/// A 9-node connected graph with uneven weights, as upper-triangle edges.
fn connected_edges() -> Vec<(usize, usize, f32)> {
    let mut edges: Vec<(usize, usize, f32)> = (0..9usize)
        .map(|i| (i, (i + 1) % 9, 0.3 + 0.07 * i as f32))
        .collect();
    edges.extend([(0, 4, 0.45), (2, 7, 0.81), (3, 8, 0.12)]);
    edges
}

/// Three separate triangles: a disconnected graph.
fn disconnected_edges() -> Vec<(usize, usize, f32)> {
    (0..3usize)
        .flat_map(|c| {
            let base = 3 * c;
            [
                (base, base + 1, 0.9 - 0.1 * c as f32),
                (base + 1, base + 2, 0.5 + 0.2 * c as f32),
                (base, base + 2, 0.35),
            ]
        })
        .collect()
}

fn outputs(graph: &Graph) -> Vec<Array2<f32>> {
    [ComputeMode::PythonCompat, ComputeMode::RustNative]
        .into_iter()
        .map(|mode| {
            let config = SpectralInitConfig::new(2)
                .with_seed(7)
                .with_compute_mode(mode);
            spectral_init(graph.view(), None, &config).expect("valid graph")
        })
        .collect()
}

#[test]
fn csc_storage_gives_the_same_result_as_csr() {
    for edges in [connected_edges(), disconnected_edges()] {
        let entries = symmetric(&edges);
        let as_csr = csr((9, 9), &entries);
        let as_csc = csc((9, 9), &entries);
        assert!(as_csr.is_csr() && as_csc.is_csc());
        assert_eq!(outputs(&as_csr), outputs(&as_csc));
    }
}

#[test]
fn stored_zeros_give_the_same_result_as_no_entry() {
    for edges in [connected_edges(), disconnected_edges()] {
        let entries = symmetric(&edges);
        let canonical = csr((9, 9), &entries);

        // Explicit zeros: a mirrored pair, a one-sided entry and a diagonal one.
        let mut with_zeros = entries.clone();
        with_zeros.extend([(1, 5, 0.0), (5, 1, 0.0), (6, 2, 0.0), (4, 4, 0.0)]);
        let padded = csr((9, 9), &with_zeros);
        assert!(padded.nnz() > canonical.nnz());

        assert_eq!(outputs(&canonical), outputs(&padded));
    }
}

#[test]
fn stored_zeros_do_not_join_components() {
    // Two triangles joined only by a stored zero stay two components, each placed at
    // its own position. Were the zero an edge, the six nodes would form one component
    // and get a different layout.
    let mut entries = symmetric(&[
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (3, 5, 1.0),
    ]);
    let separate = csr((6, 6), &entries);
    entries.extend([(2, 3, 0.0), (3, 2, 0.0)]);
    let zero_bridged = csr((6, 6), &entries);
    assert_eq!(outputs(&separate), outputs(&zero_bridged));
}

#[test]
fn data_is_ignored_when_there_are_few_components() {
    // Three components and two dimensions: 3 <= 2 * 2, so `data` is not consulted.
    let graph = csr((9, 9), &symmetric(&disconnected_edges()));
    let data = Array2::from_shape_fn((9, 4), |(i, j)| (i * 31 + j * 7) as f32);
    let config = SpectralInitConfig::new(2).with_seed(3);
    assert_eq!(
        spectral_init(graph.view(), Some(data.view()), &config),
        spectral_init(graph.view(), None, &config)
    );
}

#[test]
fn many_components_require_data() {
    // Five isolated pairs: 5 components > 2 * 2.
    let edges: Vec<(usize, usize, f32)> = (0..5usize).map(|c| (2 * c, 2 * c + 1, 1.0)).collect();
    let graph = csr((10, 10), &symmetric(&edges));
    let config = SpectralInitConfig::new(2);
    assert_err!(
        spectral_init(graph.view(), None, &config),
        SpectralError::DataRequired {
            connected_components: 5,
            n_components: 2,
            ..
        }
    );

    let data = Array2::from_shape_fn((10, 2), |(i, j)| (i / 2 + j) as f32);
    let coords =
        spectral_init(graph.view(), Some(data.view()), &config).expect("data was supplied");
    assert_eq!(coords.dim(), (10, 2));
    assert!(coords.iter().all(|v| v.is_finite()));
}

// ── tolerance calibration ────────────────────────────────────────────────────

const DATASETS: &[&str] = &[
    "blobs_50",
    "blobs_500",
    "blobs_5000",
    "blobs_connected_200",
    "blobs_connected_2000",
    "circles_300",
    "disconnected_200",
    "moons_200",
    "near_dupes_100",
];

fn fixture_graphs() -> impl Iterator<Item = (String, Graph)> {
    DATASETS.iter().flat_map(|dataset| {
        ["step5a_pruned.npz", "step5a_pruned_exact.npz"].map(|file| {
            let graph = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(dataset, file));
            (format!("{dataset}/{file}"), graph)
        })
    })
}

/// Scales the first off-diagonal weight of `graph` by `factor`, leaving its mirror alone.
fn perturb_one_weight(graph: &Graph, factor: f32) -> Graph {
    let (position, _) = graph
        .iter()
        .enumerate()
        .find(|(_, (weight, (row, col)))| row != col && **weight != 0.0)
        .expect("graph has an off-diagonal edge");
    let mut data = graph.data().to_vec();
    data[position] *= factor;
    CsMatI::new(
        graph.shape(),
        graph.indptr().to_owned().into_raw_storage(),
        graph.indices().to_vec(),
        data,
    )
}

#[test]
fn every_fixture_graph_passes_validation() {
    let mut count = 0;
    for (name, graph) in fixture_graphs() {
        assert_eq!(validate_graph(&graph), Ok(()), "{name}");
        count += 1;
    }
    assert_eq!(count, 18);
}

/// `umap-rs` symmetrizes its graph in single precision; for mix ratios other than 0, 0.5
/// and 1 the two halves of an edge can differ by a rounding error.
#[test]
fn umap_rs_graphs_pass_validation_for_every_mix_ratio() {
    for dataset in DATASETS {
        for set_op_mix_ratio in [0.3, 0.5, 0.7, 1.0] {
            let graph = crate::umap_rs::fuzzy_graph(dataset, set_op_mix_ratio);
            assert_eq!(
                validate_graph(&graph),
                Ok(()),
                "{dataset}, set_op_mix_ratio = {set_op_mix_ratio}"
            );
        }
    }
}

#[test]
fn a_one_part_in_a_thousand_asymmetry_fails_validation() {
    for (name, graph) in fixture_graphs() {
        let perturbed = perturb_one_weight(&graph, 1.0 + 1e-3);
        assert!(
            matches!(
                validate_graph(&perturbed),
                Err(SpectralError::NotSymmetric { .. })
            ),
            "{name}"
        );
    }
}

#[test]
fn a_one_ulp_asymmetry_passes_validation() {
    for (name, graph) in fixture_graphs() {
        let (position, _) = graph
            .iter()
            .enumerate()
            .find(|(_, (weight, (row, col)))| row != col && **weight != 0.0)
            .expect("graph has an off-diagonal edge");
        let mut data = graph.data().to_vec();
        data[position] = f32::from_bits(data[position].to_bits() + 1);
        let nudged: Graph = CsMatI::new(
            graph.shape(),
            graph.indptr().to_owned().into_raw_storage(),
            graph.indices().to_vec(),
            data,
        );
        assert_eq!(validate_graph(&nudged), Ok(()), "{name}");
    }
}
