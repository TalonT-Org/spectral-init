import sys
import numpy as np
import pytest
from pathlib import Path
from conftest import run_fixture_pipeline

CONNECTED_DATASETS = [
    "moons_200", "circles_300", "near_dupes_100",
    "blobs_connected_200", "blobs_connected_2000",
]
DISCONNECTED_DATASETS = ["blobs_50", "blobs_500", "blobs_5000", "disconnected_200"]
ALL_DATASETS = CONNECTED_DATASETS + DISCONNECTED_DATASETS


# ── Session fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def comp_d_e_f_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate comp_d/e/f artifacts for all datasets once per session.

    Note: pytest-xdist (``-n auto``) is not supported; run without ``-n`` flag.
    """
    td = tmp_path_factory.mktemp("comp_d_e_f")
    run_fixture_pipeline(ALL_DATASETS, str(td))
    return td


# ── helpers ───────────────────────────────────────────────────────────────────

def load(base: Path, name: str, fname: str):
    return np.load(base / name / fname, allow_pickle=False)


# ── Component D ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalues_range(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    lam = d["eigenvalues"]
    assert np.all(lam >= -1e-12) and np.all(lam <= 2.0 + 1e-12)

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalues_sorted(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    lam = d["eigenvalues"]
    assert np.all(np.diff(lam) >= -1e-12)   # non-decreasing

@pytest.mark.parametrize("name", CONNECTED_DATASETS)
def test_comp_d_first_eigenvalue_near_zero_connected(name, comp_d_e_f_outdir):
    """For connected graphs, smallest eigenvalue of normalized Laplacian is 0."""
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert d["eigenvalues"][0] < 1e-6


@pytest.mark.parametrize("name", DISCONNECTED_DATASETS)
def test_comp_d_n_components_near_zero_eigenvalues(name, comp_d_e_f_outdir):
    """Disconnected graphs have exactly n_components near-zero eigenvalues."""
    c = load(comp_d_e_f_outdir, name, "comp_c_components.npz")
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    n_components = int(c["n_components"])
    near_zero = int(np.sum(d["eigenvalues"] < 1e-6))
    assert near_zero == n_components, (
        f"{name}: expected {n_components} near-zero eigenvalues (one per component), "
        f"got {near_zero}"
    )

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_residuals_small(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert np.all(d["residuals"] < 1e-3)

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvectors_orthonormal(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    V = d["eigenvectors"]
    VTV = V.T @ V
    assert np.allclose(VTV, np.eye(VTV.shape[0]), atol=1e-10)

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_k_scalar(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    expected_k = e["embedding"].shape[1] + 1   # k = n_components + 1
    assert d["k"].ndim == 0 and d["k"] == expected_k


# ── Component E ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_e_order_skips_trivial(name, comp_d_e_f_outdir):
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    order = e["order"]
    trivial_col = np.argsort(d["eigenvalues"])[0]
    assert trivial_col not in order

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_e_embedding_shape(name, comp_d_e_f_outdir):
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    raw = np.load(comp_d_e_f_outdir / name / "step0_raw_data.npz")
    n = int(raw["n_samples"])
    assert e["embedding"].shape == (n, 2)   # n_components=2


# ── Component F ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_f_pre_noise_max_abs_10(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert abs(np.abs(f["pre_noise"]).max() - 10.0) < 1e-5   # f32 precision

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_f_final_equals_pre_plus_noise(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert np.allclose(f["final"], f["pre_noise"] + f["noise"], atol=1e-7)

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_f_noise_statistics(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    noise = f["noise"].astype(np.float64)
    tol = 5 * 0.0001 / np.sqrt(noise.size)   # 5 std-errors of the estimate
    assert abs(noise.mean()) < tol
    assert abs(noise.std() - 0.0001) < tol

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_f_dtypes(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert f["pre_noise"].dtype == np.float32
    assert f["noise"].dtype == np.float32
    assert f["final"].dtype == np.float32
    assert f["expansion"].dtype == np.float64


# ── Connectivity assertions for new connected datasets ────────────────────────

@pytest.mark.parametrize("name", ["blobs_connected_200", "blobs_connected_2000"])
def test_connected_blobs_single_component(name, comp_d_e_f_outdir):
    """Both connected blob datasets must have n_components == 1 after pruning."""
    d = load(comp_d_e_f_outdir, name, "comp_c_components.npz")
    assert int(d["n_components"]) == 1, (
        f"{name}: expected 1 connected component, got {int(d['n_components'])}"
    )


def test_blobs_connected_2000_min_n(comp_d_e_f_outdir):
    """blobs_connected_2000 must have at least 2000 samples."""
    d = np.load(comp_d_e_f_outdir / "blobs_connected_2000" / "step0_raw_data.npz")
    assert int(d["n_samples"]) >= 2000


# ── Solver metadata (REQ-META-001/002/003) ────────────────────────────────────

# --- REQ-META-001: solver_name ---

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_solver_name_present(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert "solver_name" in d.files

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_solver_name_value(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    val = d["solver_name"].item()
    assert val in {b"eigsh", b"eigh"}, f"unexpected solver_name: {val!r}"

# --- REQ-META-002: converged ---

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_converged_present(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert "converged" in d.files

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_converged_dtype(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert d["converged"].dtype == np.bool_, (
        f"converged dtype {d['converged'].dtype!r} is not bool"
    )

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_metadata_consistency(name, comp_d_e_f_outdir):
    """converged must be True iff solver_name is b'eigsh'."""
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    converged = d["converged"].item()
    solver = d["solver_name"].item()
    assert converged == (solver == b"eigsh"), (
        f"converged={converged} inconsistent with solver_name={solver!r}"
    )

# --- REQ-META-003: eigenvalue_gaps ---

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalue_gaps_present(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert "eigenvalue_gaps" in d.files

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalue_gaps_dtype_shape(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    gaps = d["eigenvalue_gaps"]
    k = int(d["k"])
    assert gaps.dtype == np.float64, f"eigenvalue_gaps dtype {gaps.dtype!r} != float64"
    assert gaps.shape == (k - 1,), f"eigenvalue_gaps shape {gaps.shape} != ({k-1},)"

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalue_gaps_match_diff(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert np.allclose(d["eigenvalue_gaps"], np.diff(d["eigenvalues"]), atol=1e-15)

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_eigenvalue_gaps_nonneg(name, comp_d_e_f_outdir):
    """Gaps must be >= 0 because eigenvalues are sorted ascending."""
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert np.all(d["eigenvalue_gaps"] >= -1e-15), (
        f"negative gap found: {d['eigenvalue_gaps'].min():.2e}"
    )

# --- REQ-VER-003: comparison method reporting ---

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_comp_d_e_gap_aware_comparison_reports_method(name, comp_d_e_f_outdir):
    """Gap-aware comparison must report the method used for each column/cluster."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from verify_fixtures import _compare_eigenvectors_gap_aware
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    order = e["order"].tolist()
    V = d["eigenvectors"]
    gaps = d["eigenvalue_gaps"]
    # Compare the embedding against the eigenvectors from which it was selected
    # (comp_e selects a subset; we verify those selected columns only)
    selected_V = V[:, order]
    selected_gaps = np.array([
        gaps[i] for i in order[:-1]  # gaps between selected columns
    ]) if len(order) > 1 else np.array([])
    errors, method_log = _compare_eigenvectors_gap_aware(
        selected_V, e["embedding"], selected_gaps
    )
    assert errors == [], f"Eigenvector comparison failed:\n" + "\n".join(errors)
    assert len(method_log) > 0, "method_log must report which comparison was used"
    for entry in method_log:
        assert "sign-flip" in entry or "subspace" in entry, (
            f"method_log entry does not name a comparison method: {entry!r}"
        )


# ── Byte-identical repro ──────────────────────────────────────────────────────

def test_byte_identical_reruns(tmp_path):
    """Running the generator twice produces byte-identical comp_f_scaling.npz."""
    run_fixture_pipeline(["blobs_50"], str(tmp_path / "run1"))
    first = (tmp_path / "run1" / "blobs_50" / "comp_f_scaling.npz").read_bytes()
    run_fixture_pipeline(["blobs_50"], str(tmp_path / "run2"))
    second = (tmp_path / "run2" / "blobs_50" / "comp_f_scaling.npz").read_bytes()
    assert first == second


# ---------------------------------------------------------------------------
# Exact-path verify_fixtures integration tests (Tests 15–16)
# ---------------------------------------------------------------------------

def test_verify_exact_path_passes_for_blobs_50(tmp_path):
    """Test 15: verify_exact_path returns empty failures list for a generated exact run."""
    run_fixture_pipeline(["blobs_50"], str(tmp_path), ["--knn-method", "exact"])
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_fixtures
    failures = verify_fixtures.verify_exact_path(tmp_path / "blobs_50", n_samples=50, n_neighbors=15)
    assert failures == [], f"verify_exact_path reported failures:\n" + "\n".join(failures)


def test_verify_detects_exact_files_via_main(tmp_path):
    """Test 16: verify_fixtures.main returns True (all pass) for a 'both' run."""
    run_fixture_pipeline(["blobs_50"], str(tmp_path), ["--knn-method", "both"])
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_fixtures
    result = verify_fixtures.main(tmp_path, ["blobs_50"])
    assert result is True, "verify_fixtures.main should return True when all checks pass"
