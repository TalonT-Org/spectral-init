//! Accuracy and parity measurements of the solver chain against the umap-learn reference
//! fixtures, shared by the assessment tests and the `init_evidence` binary.

use crate::diagnostics::{
    DEGENERATE_GAP_THRESHOLD, check_eigenvalue_bounds, eigenvalue_abs_errors,
    eigenvalue_condition_number, orthogonality_error, separation_ratio, sign_agnostic_max_error,
    spectral_gap, subspace_gram_det_kd,
};
use crate::fixtures;
use crate::internals::solve_with_unit_hint;
use ndarray::{Array1, Array2, s};
use spectral_init::__internal::{
    DENSE_EVD_QUALITY_THRESHOLD, LOBPCG_QUALITY_THRESHOLD, RSVD_QUALITY_THRESHOLD,
    SINV_LOBPCG_QUALITY_THRESHOLD, SolverLevel, SolverParams, embed_disconnected, find_components,
    max_eigenpair_residual, select_eigenvectors,
};
use spectral_init::ComputeMode;
use sprs::CsMatI;

/// A reference fixture dataset under `tests/fixtures/`.
#[derive(Debug, Clone, Copy)]
pub struct Dataset {
    pub name: &'static str,
    /// Number of points.
    pub n: usize,
    /// Whether the fuzzy graph is connected.
    pub connected: bool,
    /// Number of connected components of the fuzzy graph.
    pub n_graph_components: usize,
}

/// The nine reference datasets.
pub const DATASETS: &[Dataset] = &[
    Dataset::new("blobs_50", 50, 3),
    Dataset::new("blobs_500", 500, 4),
    Dataset::new("blobs_5000", 5000, 2),
    Dataset::new("blobs_connected_200", 200, 1),
    Dataset::new("blobs_connected_2000", 2000, 1),
    Dataset::new("circles_300", 300, 1),
    Dataset::new("disconnected_200", 200, 4),
    Dataset::new("moons_200", 200, 1),
    Dataset::new("near_dupes_100", 100, 1),
];

impl Dataset {
    const fn new(name: &'static str, n: usize, n_graph_components: usize) -> Self {
        Self {
            name,
            n,
            connected: n_graph_components == 1,
            n_graph_components,
        }
    }

    /// Number of embedding dimensions the reference was computed for: one less than the
    /// number of reference eigenvalues.
    pub fn n_embedding_dims(&self) -> usize {
        let eigenvalues = self.reference_eigenvalues();
        assert!(
            !eigenvalues.is_empty(),
            "{}: ref eigenvalues array is empty — fixture may be corrupt",
            self.name
        );
        eigenvalues.len() - 1
    }

    fn reference_eigenvalues(&self) -> Array1<f64> {
        fixtures::load_dense(
            &fixtures::fixture_path(self.name, "comp_d_eigensolver.npz"),
            "eigenvalues",
        )
    }
}

/// Residual threshold the chain applies at `level`.
pub fn residual_threshold(level: SolverLevel) -> f64 {
    match level {
        SolverLevel::DenseEvd | SolverLevel::ForcedDense => DENSE_EVD_QUALITY_THRESHOLD,
        SolverLevel::Lobpcg | SolverLevel::LobpcgRegularized => LOBPCG_QUALITY_THRESHOLD,
        SolverLevel::SinvLobpcg => SINV_LOBPCG_QUALITY_THRESHOLD,
        SolverLevel::Rsvd => RSVD_QUALITY_THRESHOLD,
    }
}

/// Eigenpairs the solver chain computes for the reference Laplacian of a connected
/// dataset.
pub struct ReferenceSolve {
    pub laplacian: CsMatI<f64, usize>,
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    /// Level of the chain that produced the eigenpairs.
    pub level: SolverLevel,
    pub n_embedding_dims: usize,
}

/// Runs the solver chain (seed 42, `PythonCompat`, default parameters) on the reference
/// Laplacian of `dataset`.
pub fn solve_reference_laplacian(dataset: &Dataset) -> ReferenceSolve {
    let n_embedding_dims = dataset.n_embedding_dims();
    let laplacian = fixtures::load_sparse_csr(&fixtures::fixture_path(
        dataset.name,
        "comp_b_laplacian.npz",
    ));
    let ((eigenvalues, eigenvectors), report) = solve_with_unit_hint(
        &laplacian,
        n_embedding_dims,
        42,
        ComputeMode::PythonCompat,
        &SolverParams::default(),
    );
    ReferenceSolve {
        laplacian,
        eigenvalues,
        eigenvectors,
        level: report.level,
        n_embedding_dims,
    }
}

/// How well computed eigenpairs satisfy the eigenproblem.
pub struct Accuracy {
    pub max_residual: f64,
    pub orthogonality_error: f64,
    /// Every eigenvalue lies in `[-bounds_tol, 2 + bounds_tol]`.
    pub eigenvalues_in_range: bool,
    pub eigenvalues_sorted: bool,
    pub spectral_gap: f64,
    /// `NaN` when the second eigenvalue is not positive.
    pub condition_number: f64,
}

pub fn measure_accuracy(solve: &ReferenceSolve, bounds_tol: f64) -> Accuracy {
    let eigenvalues = &solve.eigenvalues;
    let (eigenvalues_in_range, eigenvalues_sorted) =
        check_eigenvalue_bounds(eigenvalues, bounds_tol);
    Accuracy {
        max_residual: max_eigenpair_residual(&solve.laplacian, eigenvalues, &solve.eigenvectors),
        orthogonality_error: orthogonality_error(&solve.eigenvectors),
        eigenvalues_in_range,
        eigenvalues_sorted,
        spectral_gap: spectral_gap(eigenvalues),
        condition_number: if eigenvalues.len() > 1 && eigenvalues[1] > 0.0 {
            eigenvalue_condition_number(eigenvalues)
        } else {
            f64::NAN
        },
    }
}

/// How closely computed eigenpairs match the umap-learn reference.
pub struct Parity {
    pub max_eigenvalue_abs_error: f64,
    /// Largest element difference of the scaled, pre-noise coordinates, up to the sign of
    /// each column.
    pub sign_agnostic_max_error: f64,
    /// Gram determinant between the computed and the reference non-trivial eigenvectors:
    /// 1 when they span the same subspace.
    pub subspace_gram_det: f64,
    /// Whether two reference eigenvalues are closer than `DEGENERATE_GAP_THRESHOLD`, in
    /// which case only the subspace, not the individual vectors, is determined.
    pub degenerate: bool,
}

/// Scale eigenvectors to pre-noise f32 (matching Python's 10/max_abs expansion).
fn compute_pre_noise(selected: &Array2<f64>) -> Array2<f32> {
    let max_abs = selected.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 {
        return Array2::<f32>::zeros(selected.raw_dim());
    }
    let expansion = 10.0 / max_abs;
    selected.mapv(|v| (v * expansion) as f32)
}

pub fn measure_parity(dataset: &Dataset, solve: &ReferenceSolve) -> Parity {
    let eigensolver_path = fixtures::fixture_path(dataset.name, "comp_d_eigensolver.npz");
    let ref_eigenvalues = dataset.reference_eigenvalues();
    let ref_eigenvectors: Array2<f64> = fixtures::load_dense(&eigensolver_path, "eigenvectors");
    let ref_pre_noise: Array2<f32> = fixtures::load_dense(
        &fixtures::fixture_path(dataset.name, "comp_f_scaling.npz"),
        "pre_noise",
    );

    let abs_errors = eigenvalue_abs_errors(&solve.eigenvalues, &ref_eigenvalues);
    let max_eigenvalue_abs_error = abs_errors.iter().cloned().fold(0.0f64, f64::max);

    let selected = select_eigenvectors(
        solve
            .eigenvalues
            .as_slice_memory_order()
            .expect("eigenvalues contiguous"),
        &solve.eigenvectors,
        solve.n_embedding_dims,
    );
    let sign_agnostic_max_error =
        sign_agnostic_max_error(&compute_pre_noise(&selected), &ref_pre_noise);

    let evals_slice = ref_eigenvalues
        .as_slice_memory_order()
        .expect("ref eigenvalues contiguous");
    let degenerate = solve.n_embedding_dims >= 2
        && evals_slice[1..]
            .windows(2)
            .any(|w| w[1] - w[0] < DEGENERATE_GAP_THRESHOLD);

    Parity {
        max_eigenvalue_abs_error,
        sign_agnostic_max_error,
        subspace_gram_det: subspace_gram_det_kd(
            solve.eigenvectors.slice(s![.., 1..]),
            ref_eigenvectors.slice(s![.., 1..]),
        ),
        degenerate,
    }
}

/// Layout of a disconnected dataset.
pub struct DisconnectedLayout {
    /// Number of connected components found in the fuzzy graph.
    pub n_graph_components: usize,
    /// Smallest distance between component centroids over the largest spread within a
    /// component, with components as labelled by umap-learn.
    pub separation_ratio: f64,
}

/// Embeds the pruned fuzzy graph of a disconnected dataset (seed 42, `PythonCompat`, no
/// data) and measures how well its components are separated.
pub fn measure_disconnected(dataset: &Dataset) -> DisconnectedLayout {
    let graph = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(
        dataset.name,
        "step5a_pruned.npz",
    ));
    let (labels, n_graph_components) = find_components(graph.view());
    let embedding = embed_disconnected(
        graph.view(),
        &labels,
        n_graph_components,
        dataset.n_embedding_dims(),
        42,
        None,
        ComputeMode::PythonCompat,
        &SolverParams::default(),
    )
    .unwrap_or_else(|e| panic!("{}: embed_disconnected failed: {e}", dataset.name));

    let reference_labels: Array1<i32> = fixtures::load_dense(
        &fixtures::fixture_path(dataset.name, "comp_c_components.npz"),
        "labels",
    );
    let reference_labels: Vec<usize> = reference_labels.iter().map(|&x| x as usize).collect();

    DisconnectedLayout {
        n_graph_components,
        separation_ratio: separation_ratio(embedding.view(), &reference_labels),
    }
}
