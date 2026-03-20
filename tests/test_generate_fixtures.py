# tests/test_generate_fixtures.py
"""Tests for the KNN pipeline steps in generate_fixtures.py."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse


SCRIPT = Path(__file__).parent / "generate_fixtures.py"


@pytest.fixture(scope="session")
def blobs_50_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the blobs_50 fixture pipeline once per test session; shared by steps 3–5a tests."""
    td = tmp_path_factory.mktemp("blobs_50")
    _run(["blobs_50"], str(td))
    return td


@pytest.fixture(scope="session")
def disconnected_200_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the disconnected_200 fixture pipeline once per test session."""
    td = tmp_path_factory.mktemp("disconnected_200")
    _run(["disconnected_200"], str(td))
    return td


def _run(datasets: list[str], outdir: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPT), "--output-dir", outdir, "--datasets"] + datasets
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CMD: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_step0_raw_data_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step0_raw_data.npz")
        assert set(d.files) >= {"X", "n_samples", "n_features"}
        assert d["X"].dtype == np.float64
        assert d["X"].shape == (50, 2)
        assert d["n_samples"] == np.int32(50)
        assert d["n_features"] == np.int32(2)


def test_step1_knn_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step1_knn.npz")
        assert set(d.files) >= {"knn_indices", "knn_dists", "n_neighbors"}
        assert d["knn_indices"].dtype == np.int32
        assert d["knn_indices"].shape == (50, 15)
        assert d["knn_dists"].dtype == np.float32
        assert d["knn_dists"].shape == (50, 15)
        assert (d["knn_indices"] >= 0).all() and (d["knn_indices"] < 50).all()
        assert (d["knn_dists"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_step2_smooth_knn_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step2_smooth_knn.npz")
        assert set(d.files) >= {"sigmas", "rhos", "n_neighbors"}
        assert d["sigmas"].dtype == np.float32
        assert d["sigmas"].shape == (50,)
        assert d["rhos"].dtype == np.float32
        assert d["rhos"].shape == (50,)
        assert (d["sigmas"] > 0).all()
        assert (d["rhos"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_pipeline_is_deterministic():
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        _run(["blobs_50"], td_a)
        _run(["blobs_50"], td_b)
        for fname in (
            "step0_raw_data.npz", "step1_knn.npz", "step2_smooth_knn.npz",
            "step3_membership.npz", "step4_symmetrized.npz", "step5a_pruned.npz",
            "comp_a_degrees.npz", "comp_b_laplacian.npz", "comp_c_components.npz",
        ):
            npz_a = np.load(Path(td_a) / "blobs_50" / fname)
            npz_b = np.load(Path(td_b) / "blobs_50" / fname)
            assert set(npz_a.files) == set(npz_b.files), f"{fname}: different keys"
            for key in npz_a.files:
                np.testing.assert_array_equal(
                    npz_a[key], npz_b[key], err_msg=f"{fname}[{key!r}] differs between runs"
                )


def test_all_7_datasets_generate():
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(SCRIPT), "--output-dir", td]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CMD: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        expected_datasets = [
            "blobs_50", "blobs_500", "moons_200", "blobs_5000",
            "circles_300", "near_dupes_100", "disconnected_200",
        ]
        for name in expected_datasets:
            for fname in (
                "step0_raw_data.npz", "step1_knn.npz", "step2_smooth_knn.npz",
                "step3_membership.npz", "step4_symmetrized.npz", "step5a_pruned.npz",
                "comp_a_degrees.npz", "comp_b_laplacian.npz", "comp_c_components.npz",
            ):
                assert (Path(td) / name / fname).exists(), f"Missing {name}/{fname}"


def test_step3_membership_output(blobs_50_outdir: Path):
    """step3: shape (n,n), values in (0,1], NOT symmetric (directed graph)."""
    A = scipy.sparse.load_npz(blobs_50_outdir / "blobs_50" / "step3_membership.npz")
    assert A.shape == (50, 50)
    assert A.nnz > 0
    assert (A.data > 0).all(), "all nonzeros must be positive"
    assert (A.data <= 1.0 + 1e-10).all(), "all nonzeros must be <= 1.0"
    diff = (A - A.T).tocsr()
    assert diff.nnz > 0, "step3 must NOT be symmetric (directed graph)"


def test_step4_symmetrized_output(blobs_50_outdir: Path):
    """step4: shape (n,n), values in (0,1], symmetric."""
    A = scipy.sparse.load_npz(blobs_50_outdir / "blobs_50" / "step4_symmetrized.npz")
    assert A.shape == (50, 50)
    assert A.nnz > 0
    assert (A.data > 0).all(), "all nonzeros must be positive"
    assert (A.data <= 1.0 + 1e-10).all(), "all nonzeros must be <= 1.0"
    diff = abs(A - A.T)
    assert diff.max() < 1e-10, "step4 must be symmetric (||A - A^T|| == 0)"


def test_step5a_pruned_output(blobs_50_outdir: Path):
    """step5a: fewer nnz than step4, min nonzero >= threshold, still symmetric."""
    A4 = scipy.sparse.load_npz(blobs_50_outdir / "blobs_50" / "step4_symmetrized.npz")
    A5 = scipy.sparse.load_npz(blobs_50_outdir / "blobs_50" / "step5a_pruned.npz")
    assert A5.shape == (50, 50)
    n_epochs = 500  # n=50 <= 10000
    threshold = A4.data.max() / float(n_epochs)
    # blobs_50 has 15 neighbors with varied membership strengths; many edges fall
    # below max/500 ≈ 0.002, so pruning is guaranteed for this dataset.
    assert A5.nnz < A4.nnz, "step5a must have fewer nonzeros than step4 (edges pruned)"
    assert A5.data.min() >= threshold, "all surviving edges must be >= threshold"
    diff = abs(A5 - A5.T)
    assert diff.max() < 1e-10, "step5a must still be symmetric"


def test_comp_a_degrees_output(blobs_50_outdir: Path):
    d = np.load(blobs_50_outdir / "blobs_50" / "comp_a_degrees.npz")
    assert set(d.files) >= {"degrees", "sqrt_deg"}
    degrees = d["degrees"]
    sqrt_deg = d["sqrt_deg"]
    assert degrees.dtype == np.float64
    assert degrees.shape == (50,)
    assert sqrt_deg.dtype == np.float64
    assert sqrt_deg.shape == (50,)
    assert (degrees >= 0).all(), "degrees must be non-negative"
    np.testing.assert_allclose(sqrt_deg, np.sqrt(degrees), atol=1e-14)


def test_comp_b_laplacian_output(blobs_50_outdir: Path):
    L = scipy.sparse.load_npz(blobs_50_outdir / "blobs_50" / "comp_b_laplacian.npz")
    n = 50
    assert L.shape == (n, n)
    assert L.dtype == np.float64
    diag = np.asarray(L.diagonal())
    np.testing.assert_allclose(diag, 1.0, atol=1e-12, err_msg="Laplacian diagonal must be 1.0")
    diff = abs(L - L.T)
    assert diff.max() < 1e-14, f"Laplacian is not symmetric: max diff = {diff.max()}"
    L_dense = L.toarray()
    eigvals = np.linalg.eigvalsh(L_dense)
    assert eigvals.min() >= -1e-6, f"Eigenvalue below 0: {eigvals.min()}"
    assert eigvals.max() <= 2.0 + 1e-10, f"Eigenvalue above 2: {eigvals.max()}"


def test_comp_c_components_connected(blobs_50_outdir: Path):
    d = np.load(blobs_50_outdir / "blobs_50" / "comp_c_components.npz")
    assert set(d.files) >= {"n_components", "labels"}
    assert int(d["n_components"]) == 1, "blobs_50 must be a single connected component"
    labels = d["labels"]
    assert labels.dtype == np.int32
    assert labels.shape == (50,)
    assert (labels == 0).all(), "all nodes in a single component must have label 0"


def test_comp_c_components_disconnected(disconnected_200_outdir: Path):
    d = np.load(disconnected_200_outdir / "disconnected_200" / "comp_c_components.npz")
    n_components = int(d["n_components"])
    assert n_components > 1, f"disconnected_200 must have >1 component, got {n_components}"
    labels = d["labels"]
    assert labels.dtype == np.int32
    assert labels.shape == (200,)
    assert labels.min() >= 0
    assert labels.max() < n_components


def test_step3_membership_no_explicit_zeros():
    """step3_membership.data must contain no explicit zero entries after generation."""
    datasets = [
        "blobs_50", "blobs_500", "moons_200", "blobs_5000",
        "circles_300", "near_dupes_100", "disconnected_200",
    ]
    with tempfile.TemporaryDirectory() as td:
        _run(datasets, td)
        for name in datasets:
            A = scipy.sparse.load_npz(
                str(Path(td) / name / "step3_membership.npz")
            )
            assert np.all(A.data > 0), (
                f"[{name}] step3_membership contains explicit zeros "
                f"(count={np.sum(A.data == 0)})"
            )


# ---------------------------------------------------------------------------
# Exact KNN path tests (Tests 1–14)
# ---------------------------------------------------------------------------

def test_step1_knn_exact_output():
    """Test 1: exact-only run produces step1_knn_exact.npz; step1_knn.npz absent."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        p = Path(td) / "blobs_50"
        d = np.load(p / "step1_knn_exact.npz")
        assert set(d.files) >= {"knn_indices", "knn_dists", "n_neighbors"}
        assert d["knn_indices"].dtype == np.int32
        assert d["knn_indices"].shape == (50, 15)
        assert d["knn_dists"].dtype == np.float32
        assert d["knn_dists"].shape == (50, 15)
        assert (d["knn_indices"] >= 0).all() and (d["knn_indices"] < 50).all()
        # no self-references
        for i in range(50):
            assert i not in d["knn_indices"][i], f"self-reference at row {i}"
        assert (d["knn_dists"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)
        assert not (p / "step1_knn.npz").exists(), "approx step1 should not exist in exact-only run"


def test_step1_knn_exact_distances_match_pairwise():
    """Test 2: exact KNN distances match sklearn pairwise_distances ground truth."""
    sys.path.insert(0, str(SCRIPT.parent))
    from generate_fixtures import generate_step1_knn_exact
    from sklearn.metrics import pairwise_distances
    from fixture_utils import DATASETS

    X = None
    for name, gen_fn, kwargs in DATASETS:
        if name == "blobs_50":
            X, _ = gen_fn(**kwargs)
            break
    assert X is not None

    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        knn_indices, knn_dists = generate_step1_knn_exact(X, outdir, 15)
        pdist = pairwise_distances(X, metric="euclidean")
        for i in range(len(X)):
            for j in range(15):
                expected = pdist[i, knn_indices[i, j]]
                np.testing.assert_allclose(
                    knn_dists[i, j], expected, atol=1e-5,
                    err_msg=f"Distance mismatch at row {i}, neighbor {j}",
                )


def test_knn_method_exact_generates_exact_files():
    """Test 3: --knn-method exact generates all _exact step files; no approx step1."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        p = Path(td) / "blobs_50"
        for fname in (
            "step1_knn_exact.npz", "step2_smooth_knn_exact.npz",
            "step3_membership_exact.npz", "step4_symmetrized_exact.npz",
            "step5a_pruned_exact.npz",
        ):
            assert (p / fname).exists(), f"Missing {fname}"
        assert not (p / "step1_knn.npz").exists(), "approx step1 should not exist"


def test_knn_method_approx_default():
    """Test 4: default (approx) run produces step1_knn.npz; no _exact files."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        p = Path(td) / "blobs_50"
        assert (p / "step1_knn.npz").exists()
        assert not (p / "step1_knn_exact.npz").exists()


def test_knn_method_both_generates_all_files():
    """Test 5: --knn-method both generates both approx and exact step1/5a files."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "both"])
        p = Path(td) / "blobs_50"
        assert (p / "step1_knn.npz").exists()
        assert (p / "step1_knn_exact.npz").exists()
        assert (p / "step5a_pruned.npz").exists()
        assert (p / "step5a_pruned_exact.npz").exists()


def test_exact_step3_membership_is_directed():
    """Test 6: step3_membership_exact.npz must be directed (asymmetric)."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        A = scipy.sparse.load_npz(str(Path(td) / "blobs_50" / "step3_membership_exact.npz"))
        diff = (A - A.T).tocsr()
        assert diff.nnz > 0, "step3_membership_exact must be asymmetric (directed)"


def test_exact_step4_symmetrized_is_symmetric():
    """Test 7: step4_symmetrized_exact.npz must be symmetric."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        A = scipy.sparse.load_npz(str(Path(td) / "blobs_50" / "step4_symmetrized_exact.npz"))
        assert abs(A - A.T).max() < 1e-10, "step4_symmetrized_exact must be symmetric"


def test_exact_step5a_pruned_fewer_edges():
    """Test 8: step5a_pruned_exact must have fewer edges than step4_symmetrized_exact."""
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        p = Path(td) / "blobs_50"
        A4 = scipy.sparse.load_npz(str(p / "step4_symmetrized_exact.npz"))
        A5 = scipy.sparse.load_npz(str(p / "step5a_pruned_exact.npz"))
        assert A5.nnz < A4.nnz, "step5a_pruned_exact must have fewer nnz than step4_symmetrized_exact"


def test_exact_path_deterministic():
    """Test 9: exact path is deterministic (byte-identical across two runs)."""
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        _run(["blobs_50"], td_a, ["--knn-method", "exact"])
        _run(["blobs_50"], td_b, ["--knn-method", "exact"])
        f_a = (Path(td_a) / "blobs_50" / "step1_knn_exact.npz").read_bytes()
        f_b = (Path(td_b) / "blobs_50" / "step1_knn_exact.npz").read_bytes()
        assert f_a == f_b, "step1_knn_exact.npz must be byte-identical across runs"


def test_knn_method_manifest_records_method():
    """Test 10: manifest records knn_method and step_files_exact for exact run."""
    import json
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, ["--knn-method", "exact"])
        manifest = json.loads((Path(td) / "manifest.json").read_text())
        entry = next(e for e in manifest["datasets"] if e["name"] == "blobs_50")
        assert entry["knn_method"] == "exact"
        assert "step1_knn_exact.npz" in entry["step_files_exact"]


def test_suffix_parameter_step2():
    """Test 11: generate_step2_smooth_knn with suffix='_exact' writes correct file."""
    sys.path.insert(0, str(SCRIPT.parent))
    from generate_fixtures import generate_step2_smooth_knn

    rng = np.random.RandomState(0)
    knn_dists = rng.rand(50, 15).astype(np.float32) * 2.0
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        generate_step2_smooth_knn(knn_dists, outdir, 15, suffix="_exact")
        assert (outdir / "step2_smooth_knn_exact.npz").exists()
        assert not (outdir / "step2_smooth_knn.npz").exists()


def test_suffix_parameter_step3():
    """Test 12: generate_step3_membership with suffix='_exact' writes correct file."""
    sys.path.insert(0, str(SCRIPT.parent))
    from generate_fixtures import generate_step2_smooth_knn, generate_step3_membership

    rng = np.random.RandomState(0)
    knn_indices = np.argsort(rng.rand(50, 50), axis=1)[:, 1:16].astype(np.int32)
    knn_dists = rng.rand(50, 15).astype(np.float32) * 2.0
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        sigmas, rhos = generate_step2_smooth_knn(knn_dists, outdir, 15, suffix="_exact")
        generate_step3_membership(knn_indices, knn_dists, sigmas, rhos, outdir, 50, suffix="_exact")
        assert (outdir / "step3_membership_exact.npz").exists()
        assert not (outdir / "step3_membership.npz").exists()


def test_suffix_parameter_step4():
    """Test 13: generate_step4_symmetrized with suffix='_exact' writes correct file."""
    sys.path.insert(0, str(SCRIPT.parent))
    from generate_fixtures import generate_step4_symmetrized

    n = 20
    rng = np.random.RandomState(0)
    rows = rng.randint(0, n, 100)
    cols = rng.randint(0, n, 100)
    vals = rng.rand(100).astype(np.float32) * 0.5
    directed = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        generate_step4_symmetrized(directed, outdir, suffix="_exact")
        assert (outdir / "step4_symmetrized_exact.npz").exists()
        assert not (outdir / "step4_symmetrized.npz").exists()


def test_suffix_parameter_step5a():
    """Test 14: generate_step5a_prune with suffix='_exact' writes correct file."""
    sys.path.insert(0, str(SCRIPT.parent))
    from generate_fixtures import generate_step4_symmetrized, generate_step5a_prune

    n = 20
    rng = np.random.RandomState(0)
    rows = rng.randint(0, n, 100)
    cols = rng.randint(0, n, 100)
    vals = rng.rand(100).astype(np.float32) * 0.5
    directed = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        symmetric = generate_step4_symmetrized(directed, outdir, suffix="_exact")
        generate_step5a_prune(symmetric, outdir, n, suffix="_exact")
        assert (outdir / "step5a_pruned_exact.npz").exists()
        assert not (outdir / "step5a_pruned.npz").exists()
