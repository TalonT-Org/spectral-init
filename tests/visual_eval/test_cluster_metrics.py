"""Smoke tests for cluster_metrics.py — Category C cluster preservation metrics."""
import numpy as np


def _make_cluster_blobs(n_clusters=5, n_per_cluster=60, seed=42):
    """Well-separated blobs with known cluster labels.

    Returns
    -------
    embedding : (n_clusters * n_per_cluster, 2) float64 — UMAP-like 2-D coordinates
    true_labels : (n_clusters * n_per_cluster,) int — cluster IDs 0..n_clusters-1
    """
    from sklearn.datasets import make_blobs

    embedding, true_labels = make_blobs(
        n_samples=n_clusters * n_per_cluster,
        centers=n_clusters,
        cluster_std=0.3,
        center_box=(-20.0, 20.0),
        random_state=seed,
    )
    return embedding.astype(np.float64), true_labels.astype(np.int32)


def test_cluster_ari_returns_float_in_range():
    embedding, labels = _make_cluster_blobs()
    from cluster_metrics import cluster_ari

    result = cluster_ari(embedding, labels)
    assert isinstance(result, float)
    assert -0.5 <= result <= 1.0


def test_cluster_nmi_returns_float_in_range():
    embedding, labels = _make_cluster_blobs()
    from cluster_metrics import cluster_nmi

    result = cluster_nmi(embedding, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_celltype_purity_returns_float_in_unit_interval():
    embedding, labels = _make_cluster_blobs()
    from cluster_metrics import celltype_purity

    result = celltype_purity(embedding, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_perfect_alignment_gives_scores_near_one():
    """Well-separated blobs with correct cluster count → ARI≈1, NMI≈1, purity≈1."""
    embedding, true_labels = _make_cluster_blobs()
    from cluster_metrics import compute_cluster_metrics

    result = compute_cluster_metrics(embedding, true_labels)
    assert result["ari"] > 0.95, f"ARI too low: {result['ari']}"
    assert result["nmi"] > 0.95, f"NMI too low: {result['nmi']}"
    assert result["celltype_purity"] > 0.95, f"purity too low: {result['celltype_purity']}"


def test_random_permutation_gives_ari_near_zero():
    """Randomly permuted reference labels give ARI ≈ 0.0, NMI ≈ 0.0, and degraded purity."""
    embedding, true_labels = _make_cluster_blobs()
    rng = np.random.RandomState(0)
    permuted = rng.permutation(true_labels)
    from cluster_metrics import compute_cluster_metrics

    result = compute_cluster_metrics(embedding, permuted)
    assert abs(result["ari"]) < 0.1, f"ARI not near zero: {result['ari']}"
    assert result["nmi"] < 0.1, f"NMI not near zero: {result['nmi']}"
    assert result["celltype_purity"] < 0.4, f"purity not sufficiently degraded: {result['celltype_purity']}"


def test_compute_cluster_metrics_returns_required_keys():
    embedding, labels = _make_cluster_blobs()
    from cluster_metrics import compute_cluster_metrics

    result = compute_cluster_metrics(embedding, labels)
    for key in ("ari", "nmi", "celltype_purity"):
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float)
