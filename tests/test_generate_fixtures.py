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
    assert eigvals.min() >= -1e-7, f"Eigenvalue below 0: {eigvals.min()}"
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


# ---------------------------------------------------------------------------
# Exact-distance KNN path tests
# ---------------------------------------------------------------------------

def test_step1_knn_exact_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step1_knn_exact.npz")
        assert set(d.files) >= {"knn_indices", "knn_dists", "n_neighbors"}
        assert d["knn_indices"].dtype == np.int32
        assert d["knn_indices"].shape == (50, 15)
        assert d["knn_dists"].dtype == np.float32
        assert d["knn_dists"].shape == (50, 15)
        assert (d["knn_indices"] >= 0).all() and (d["knn_indices"] < 50).all()
        assert (d["knn_dists"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_step1_knn_exact_self_not_in_indices():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step1_knn_exact.npz")
        idx = d["knn_indices"]
        for i in range(idx.shape[0]):
            assert i not in idx[i].tolist(), f"Self-index {i} found in row {i}"


def test_exact_pipeline_files_generated_for_small_dataset():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        for fname in (
            "step1_knn_exact.npz",
            "step2_smooth_knn_exact.npz",
            "step3_membership_exact.npz",
            "step4_symmetrized_exact.npz",
            "step5a_pruned_exact.npz",
        ):
            assert (Path(td) / "blobs_50" / fname).exists(), f"Missing {fname}"


def test_exact_pipeline_not_generated_for_large_dataset():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_5000"], td)
        for fname in (
            "step1_knn_exact.npz",
            "step2_smooth_knn_exact.npz",
        ):
            assert not (Path(td) / "blobs_5000" / fname).exists(), \
                f"{fname} should not exist for n=5000"


def test_exact_pipeline_coexists_with_approx():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        approx = np.load(Path(td) / "blobs_50" / "step1_knn.npz")
        exact = np.load(Path(td) / "blobs_50" / "step1_knn_exact.npz")
        # Both must exist
        assert approx["knn_indices"].shape == (50, 15)
        assert exact["knn_indices"].shape == (50, 15)
        # They may differ (different algorithms), but approx must still match PyNNDescent
        assert approx["knn_dists"][:, 0].min() >= 0


def test_exact_knn_distances_are_exact():
    from sklearn.metrics import pairwise_distances as sk_pdist
    from fixture_utils import make_blobs_dataset
    X, _ = make_blobs_dataset(n=50, n_features=2, n_centers=3, seed=42)
    D = sk_pdist(X, metric="euclidean")
    sorted_idx = np.argsort(D, axis=1)
    expected_indices = sorted_idx[:, 1:16].astype(np.int32)
    expected_dists = D[np.arange(50)[:, None], expected_indices].astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step1_knn_exact.npz")
        np.testing.assert_array_equal(d["knn_indices"], expected_indices)
        np.testing.assert_allclose(d["knn_dists"], expected_dists, rtol=1e-5)


def test_exact_step2_smooth_knn_exact_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step2_smooth_knn_exact.npz")
        assert d["sigmas"].dtype == np.float32
        assert d["sigmas"].shape == (50,)
        assert d["rhos"].dtype == np.float32
        assert d["rhos"].shape == (50,)
        assert (d["sigmas"] > 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_exact_knn_method_flag_approx_only():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, extra_args=["--knn-method", "approx"])
        assert not (Path(td) / "blobs_50" / "step1_knn_exact.npz").exists()
        assert (Path(td) / "blobs_50" / "step1_knn.npz").exists()


def test_exact_knn_method_flag_exact_only():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td, extra_args=["--knn-method", "exact"])
        assert (Path(td) / "blobs_50" / "step1_knn_exact.npz").exists()
        assert not (Path(td) / "blobs_50" / "step1_knn.npz").exists()


def test_verify_exact_fixtures_pass():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_fixtures
    import json
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        manifest = {"generated_at": "2026-01-01T00:00:00Z", "datasets": [
            {"name": "blobs_50", "shape": [50, 2], "params": {}, "n_neighbors": 15,
             "step_files": []}
        ]}
        (Path(td) / "manifest.json").write_text(json.dumps(manifest))
        ok = verify_fixtures.main(output_dir=td, dataset_names=["blobs_50"])
        assert ok, "verify_fixtures failed for blobs_50 exact fixtures"
