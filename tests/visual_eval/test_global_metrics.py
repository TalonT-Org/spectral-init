"""Smoke tests for global_metrics.py — Category D global structure metrics."""
import numpy as np


def _make_high_low_blobs(n_clusters=5, n_per_cluster=100, n_dims=20, seed=42):
    """Well-separated blobs in high-dim, PCA projection, and random permutation.

    Returns
    -------
    X_high       : (n_total, n_dims) float64 — high-dimensional features
    X_low_pca    : (n_total, 2) float64 — PCA 2D projection (structured)
    X_low_random : (n_total, 2) float64 — row-permuted X_low_pca (unstructured)
    labels       : (n_total,) int32 — cluster IDs 0..n_clusters-1
    """
    from sklearn.datasets import make_blobs
    from sklearn.decomposition import PCA

    X_high, labels = make_blobs(
        n_samples=n_clusters * n_per_cluster,
        n_features=n_dims,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=seed,
    )
    X_high = X_high.astype(np.float64)
    labels = labels.astype(np.int32)

    pca = PCA(n_components=2, random_state=seed)
    X_low_pca = pca.fit_transform(X_high)

    rng = np.random.RandomState(seed + 1)
    X_low_random = rng.permutation(X_low_pca)

    return X_high, X_low_pca, X_low_random, labels


# --- API and range tests ---

def test_random_triplet_accuracy_returns_float_in_range():
    X_high, X_low_pca, _, _ = _make_high_low_blobs()
    from global_metrics import random_triplet_accuracy
    result = random_triplet_accuracy(X_high, X_low_pca)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_shepard_correlation_returns_dict_with_correct_keys():
    X_high, X_low_pca, _, _ = _make_high_low_blobs()
    from global_metrics import shepard_correlation
    result = shepard_correlation(X_high, X_low_pca)
    assert isinstance(result, dict)
    assert "pearson" in result and "spearman" in result
    assert isinstance(result["pearson"], float) and isinstance(result["spearman"], float)
    assert -1.0 <= result["pearson"] <= 1.0
    assert -1.0 <= result["spearman"] <= 1.0


def test_centroid_distance_correlation_returns_float_in_range():
    X_high, X_low_pca, _, labels = _make_high_low_blobs()
    from global_metrics import centroid_distance_correlation
    result = centroid_distance_correlation(X_high, X_low_pca, labels)
    assert isinstance(result, float)
    assert result > 0.9, f"centroid_dist_corr too low for well-separated PCA blobs: {result:.4f}"


def test_knn_preservation_returns_float_in_unit_interval():
    # Use n_dims=2 so PCA is a near-identity transform and knn preservation is high.
    X_high, X_low_pca, _, _ = _make_high_low_blobs(n_dims=2)
    from global_metrics import knn_preservation
    result = knn_preservation(X_high, X_low_pca, k=10)
    assert isinstance(result, float)
    assert result >= 0.7, f"knn_preservation too low for PCA-projected blobs: {result:.4f}"
    assert result <= 1.0


def test_compute_global_metrics_returns_required_keys():
    X_high, X_low_pca, _, labels = _make_high_low_blobs()
    from global_metrics import compute_global_metrics
    result = compute_global_metrics(X_high, X_low_pca, labels)
    for key in ("triplet_accuracy", "shepard_pearson", "shepard_spearman",
                "centroid_dist_corr", "knn_preservation"):
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float)


# --- Directionality tests (REQ-TEST-001 and REQ-TEST-002) ---

def test_pca_beats_random_directionality():
    """PCA 2D projection scores strictly higher than row-permuted random on all metrics."""
    X_high, X_low_pca, X_low_random, labels = _make_high_low_blobs()
    from global_metrics import compute_global_metrics
    good = compute_global_metrics(X_high, X_low_pca, labels)
    bad  = compute_global_metrics(X_high, X_low_random, labels)
    for key in ("triplet_accuracy", "shepard_pearson", "shepard_spearman",
                "centroid_dist_corr", "knn_preservation"):
        assert not np.isnan(good[key]), f"{key}: PCA result is NaN"
        assert not np.isnan(bad[key]), f"{key}: random result is NaN"
        assert good[key] > bad[key], (
            f"{key}: PCA={good[key]:.4f} not > random={bad[key]:.4f}"
        )


def test_identity_triplet_accuracy_near_one():
    """When high-dim and embedding are the same 2D array, triplet accuracy ≈ 1.0."""
    from sklearn.datasets import make_blobs
    from global_metrics import random_triplet_accuracy
    X_2d, _ = make_blobs(n_samples=300, n_features=2, centers=5,
                          cluster_std=0.3, random_state=0)
    X_2d = X_2d.astype(np.float64)
    result = random_triplet_accuracy(X_2d, X_2d, seed=42)
    assert result > 0.99, f"Identity mapping triplet accuracy too low: {result:.4f}"


# --- Reproducibility test ---

def test_triplet_accuracy_reproducible():
    X_high, X_low_pca, _, _ = _make_high_low_blobs()
    from global_metrics import random_triplet_accuracy
    r1 = random_triplet_accuracy(X_high, X_low_pca, seed=7)
    r2 = random_triplet_accuracy(X_high, X_low_pca, seed=7)
    assert r1 == r2
