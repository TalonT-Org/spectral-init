import subprocess, sys
import numpy as np
import pytest
from pathlib import Path

DATASETS = ["blobs_50", "disconnected_200", "near_dupes_100"]
SCRIPT = Path(__file__).parent / "generate_fixtures.py"


# ── Session fixture ────────────────────────────────────────────────────────────

def _run(datasets: list[str], outdir: str) -> None:
    cmd = [sys.executable, str(SCRIPT), "--output-dir", outdir, "--datasets"] + datasets
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CMD: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.fixture(scope="session")
def comp_d_e_f_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate comp_d/e/f artifacts for three representative datasets once per session.

    Note: pytest-xdist (``-n auto``) is not supported; run without ``-n`` flag.
    """
    td = tmp_path_factory.mktemp("comp_d_e_f")
    _run(DATASETS, str(td))
    return td


# ── helpers ───────────────────────────────────────────────────────────────────

def load(base: Path, name: str, fname: str):
    return np.load(base / name / fname, allow_pickle=False)


# ── Component D ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DATASETS)
def test_comp_d_eigenvalues_range(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    lam = d["eigenvalues"]
    assert np.all(lam >= -1e-12) and np.all(lam <= 2.0 + 1e-12)

@pytest.mark.parametrize("name", DATASETS)
def test_comp_d_eigenvalues_sorted(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    lam = d["eigenvalues"]
    assert np.all(np.diff(lam) >= -1e-12)   # non-decreasing

@pytest.mark.parametrize("name", ["blobs_50", "near_dupes_100"])
def test_comp_d_first_eigenvalue_near_zero_connected(name, comp_d_e_f_outdir):
    """For connected graphs, smallest eigenvalue of normalized Laplacian is 0."""
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert d["eigenvalues"][0] < 1e-6

@pytest.mark.parametrize("name", DATASETS)
def test_comp_d_residuals_small(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    assert np.all(d["residuals"] < 1e-3)

@pytest.mark.parametrize("name", DATASETS)
def test_comp_d_eigenvectors_orthonormal(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    V = d["eigenvectors"]
    VTV = V.T @ V
    assert np.allclose(VTV, np.eye(VTV.shape[0]), atol=1e-10)

@pytest.mark.parametrize("name", DATASETS)
def test_comp_d_k_scalar(name, comp_d_e_f_outdir):
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    expected_k = e["embedding"].shape[1] + 1   # k = n_components + 1
    assert d["k"].ndim == 0 and d["k"] == expected_k


# ── Component E ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DATASETS)
def test_comp_e_order_skips_trivial(name, comp_d_e_f_outdir):
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    d = load(comp_d_e_f_outdir, name, "comp_d_eigensolver.npz")
    order = e["order"]
    trivial_col = np.argsort(d["eigenvalues"])[0]
    assert trivial_col not in order

@pytest.mark.parametrize("name", DATASETS)
def test_comp_e_embedding_shape(name, comp_d_e_f_outdir):
    e = load(comp_d_e_f_outdir, name, "comp_e_selection.npz")
    raw = np.load(comp_d_e_f_outdir / name / "step0_raw_data.npz")
    n = int(raw["n_samples"])
    assert e["embedding"].shape == (n, 2)   # n_components=2


# ── Component F ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DATASETS)
def test_comp_f_pre_noise_max_abs_10(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert abs(np.abs(f["pre_noise"]).max() - 10.0) < 1e-5   # f32 precision

@pytest.mark.parametrize("name", DATASETS)
def test_comp_f_final_equals_pre_plus_noise(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert np.allclose(f["final"], f["pre_noise"] + f["noise"], atol=1e-7)

@pytest.mark.parametrize("name", DATASETS)
def test_comp_f_noise_statistics(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    noise = f["noise"].astype(np.float64)
    tol = 5 * 0.0001 / np.sqrt(noise.size)   # 5 std-errors of the estimate
    assert abs(noise.mean()) < tol
    assert abs(noise.std() - 0.0001) < tol

@pytest.mark.parametrize("name", DATASETS)
def test_comp_f_dtypes(name, comp_d_e_f_outdir):
    f = load(comp_d_e_f_outdir, name, "comp_f_scaling.npz")
    assert f["pre_noise"].dtype == np.float32
    assert f["noise"].dtype == np.float32
    assert f["final"].dtype == np.float32
    assert f["expansion"].dtype == np.float64


# ── Byte-identical repro ──────────────────────────────────────────────────────

def test_byte_identical_reruns(tmp_path):
    """Running the generator twice produces byte-identical comp_f_scaling.npz."""
    _run(["blobs_50"], str(tmp_path / "run1"))
    first = (tmp_path / "run1" / "blobs_50" / "comp_f_scaling.npz").read_bytes()
    _run(["blobs_50"], str(tmp_path / "run2"))
    second = (tmp_path / "run2" / "blobs_50" / "comp_f_scaling.npz").read_bytes()
    assert first == second
