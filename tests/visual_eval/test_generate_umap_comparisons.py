"""Unit tests for generate_umap_comparisons.py Tier 2 additions."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_load_pendigits():
    from tests.visual_eval.generate_umap_comparisons import load_dataset
    X, labels, name = load_dataset("pendigits")
    assert X.shape == (1797, 64)
    assert X.dtype == np.float64
    assert labels.shape == (1797,)
    assert labels.dtype == np.int32
    assert name == "pendigits"
    assert set(labels) == set(range(10))


def test_singlecell_raises():
    from tests.visual_eval.generate_umap_comparisons import load_dataset
    with pytest.raises(NotImplementedError, match="scanpy"):
        load_dataset("singlecell")


@pytest.mark.parametrize("tier,expected", [
    ("1", ["blobs_1000", "circles_1000", "swiss_roll_2000", "two_moons_1000", "blobs_hd_2000"]),
    ("2", ["mnist_10k", "fashion_mnist_10k", "pendigits"]),
])
def test_tier_selection(tier, expected):
    from tests.visual_eval.generate_umap_comparisons import TIER1_DATASETS, TIER2_DATASETS
    assert (TIER1_DATASETS if tier == "1" else TIER2_DATASETS) == expected


def test_load_mnist_deterministic(monkeypatch):
    """fetch_openml is mocked; verifies deterministic subsampling."""
    import tests.visual_eval.generate_umap_comparisons as m
    n_total = 70000
    fake_data = np.arange(n_total * 4, dtype=np.float32).reshape(n_total, 4)
    fake_target = np.tile(np.arange(10), n_total // 10).astype(str)
    fake_bundle = SimpleNamespace(data=fake_data, target=fake_target)
    monkeypatch.setattr(m, "_fetch_openml", lambda *a, **kw: fake_bundle)
    X1, y1, name1 = m.load_mnist()
    X2, y2, name2 = m.load_mnist()
    assert name1 == "mnist_10k"
    assert X1.shape == (10000, 4)
    np.testing.assert_array_equal(X1, X2)   # same seed → same subsample
    np.testing.assert_array_equal(y1, y2)


def test_plot_large_dataset_uses_small_scatter(tmp_path):
    """_make_baseline_plot applies s=0.5 and alpha=0.3 when n > 5000."""
    from tests.visual_eval.generate_umap_comparisons import _make_baseline_plot
    n = 10000
    rng = np.random.RandomState(0)
    init = rng.randn(n, 2)
    final = rng.randn(n, 2).astype(np.float32)
    labels = (np.arange(n) % 10).astype(np.int32)
    eigs = np.linspace(0, 1, 10)
    metrics = {"trustworthiness": 0.9, "silhouette": 0.5,
               "n_components": 1, "spectral_gap": 0.1, "condition_number": 10.0}
    _make_baseline_plot("test_large", init, final, labels, eigs, metrics, tmp_path)
    assert (tmp_path / "test_large_baseline.png").exists()


def test_unknown_dataset_exits(capsys):
    from tests.visual_eval.generate_umap_comparisons import main
    with pytest.raises(SystemExit):
        sys.argv = ["prog", "--phase", "baseline", "--dataset", "nonexistent_xyz"]
        main()


def test_all_tier_datasets():
    from tests.visual_eval.generate_umap_comparisons import TIER1_DATASETS, TIER2_DATASETS, ALL_DATASETS
    assert ALL_DATASETS == TIER1_DATASETS + TIER2_DATASETS
    assert len(ALL_DATASETS) == len(TIER1_DATASETS) + len(TIER2_DATASETS)
