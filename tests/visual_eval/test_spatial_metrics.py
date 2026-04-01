"""Smoke tests for spatial_metrics.py — Category B spatial correlation metrics."""
import numpy as np


def _make_spatial_blobs(n_per_cluster=80, seed=42):
    """3-cluster blobs with physically separated centers.

    Returns
    -------
    spatial_coords : (240, 2) float64 — physical x,y
    good_embedding : (240, 2) — mirrors spatial layout (small jitter)
    random_embedding : (240, 2) — no spatial structure
    labels : (240,) int — cluster IDs 0,1,2
    """
    from sklearn.datasets import make_blobs

    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])
    spatial_coords, labels = make_blobs(
        n_samples=n_per_cluster * 3,
        centers=centers,
        cluster_std=0.8,
        random_state=seed,
    )
    good_embedding = spatial_coords + rng.randn(*spatial_coords.shape) * 0.3
    random_embedding = rng.randn(*spatial_coords.shape) * 5.0
    return spatial_coords.astype(np.float64), good_embedding, random_embedding, labels


# ---------------------------------------------------------------------------
# Metric function API tests
# ---------------------------------------------------------------------------


def test_sna_returns_float_in_range():
    spatial, good_emb, _, labels = _make_spatial_blobs()
    from spatial_metrics import spatial_neighbor_agreement

    result = spatial_neighbor_agreement(spatial, good_emb, k=15)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_spatial_dist_corr_returns_float_in_range():
    spatial, good_emb, _, _ = _make_spatial_blobs()
    from spatial_metrics import spatial_distance_correlation

    result = spatial_distance_correlation(spatial, good_emb, sample_size=200, seed=0)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_morans_i_returns_float_in_range():
    spatial, good_emb, _, _ = _make_spatial_blobs()
    from spatial_metrics import morans_i

    result = morans_i(spatial, good_emb[:, 0], k=6)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_chaos_returns_nonnegative_float():
    spatial, _, _, labels = _make_spatial_blobs()
    from spatial_metrics import chaos_score

    result = chaos_score(spatial, labels)
    assert isinstance(result, float)
    assert result >= 0.0


def test_pas_returns_float_in_unit_interval():
    spatial, _, _, labels = _make_spatial_blobs()
    from spatial_metrics import pas_score

    result = pas_score(spatial, labels, k=10, threshold=0.6)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Directionality tests (REQ-TEST-002)
# ---------------------------------------------------------------------------


def test_sna_directionality():
    """Good embedding (mirrors spatial) beats random on SNA."""
    spatial, good_emb, rand_emb, _ = _make_spatial_blobs()
    from spatial_metrics import spatial_neighbor_agreement

    assert spatial_neighbor_agreement(spatial, good_emb) > spatial_neighbor_agreement(
        spatial, rand_emb
    )


def test_spatial_dist_corr_directionality():
    """Good embedding has higher Spearman rank correlation than random."""
    spatial, good_emb, rand_emb, _ = _make_spatial_blobs()
    from spatial_metrics import spatial_distance_correlation

    assert spatial_distance_correlation(
        spatial, good_emb, sample_size=200
    ) > spatial_distance_correlation(spatial, rand_emb, sample_size=200)


def test_morans_i_directionality():
    """Embedding that mirrors spatial has higher Moran's I than random."""
    spatial, good_emb, rand_emb, _ = _make_spatial_blobs()
    from spatial_metrics import morans_i

    assert morans_i(spatial, good_emb[:, 0]) > morans_i(spatial, rand_emb[:, 0])


def test_chaos_directionality():
    """True (spatially compact) labels give lower CHAOS than scrambled labels."""
    spatial, _, _, labels = _make_spatial_blobs()
    rng = np.random.RandomState(0)
    scrambled = rng.permutation(labels)
    from spatial_metrics import chaos_score

    assert chaos_score(spatial, labels) < chaos_score(spatial, scrambled)


def test_pas_directionality():
    """True labels give lower PAS than scrambled labels."""
    spatial, _, _, labels = _make_spatial_blobs()
    rng = np.random.RandomState(0)
    scrambled = rng.permutation(labels)
    from spatial_metrics import pas_score

    assert pas_score(spatial, labels) < pas_score(spatial, scrambled)


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


def test_compute_spatial_metrics_returns_required_keys():
    spatial, good_emb, _, labels = _make_spatial_blobs()
    from spatial_metrics import compute_spatial_metrics

    result = compute_spatial_metrics(spatial, good_emb, labels)
    for key in [
        "sna",
        "spatial_dist_corr",
        "morans_i_max",
        "morans_i_dim0",
        "morans_i_dim1",
        "chaos",
        "pas",
    ]:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float)


def test_morans_i_max_is_max_of_dims():
    spatial, good_emb, _, labels = _make_spatial_blobs()
    from spatial_metrics import compute_spatial_metrics

    result = compute_spatial_metrics(spatial, good_emb, labels)
    assert result["morans_i_max"] == max(result["morans_i_dim0"], result["morans_i_dim1"])


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_chaos_skips_singleton_clusters():
    """Clusters with fewer than 2 cells must not raise."""
    from spatial_metrics import chaos_score

    rng = np.random.RandomState(7)
    # Cluster 0 has 1 cell, cluster 1 has 50 cells
    spatial = rng.randn(51, 2)
    labels = np.array([0] + [1] * 50, dtype=np.int32)
    result = chaos_score(spatial, labels)
    assert isinstance(result, float) and result >= 0.0


def test_spatial_dist_corr_subsampling_reproducibility():
    """spatial_distance_correlation with same seed gives same result."""
    from spatial_metrics import spatial_distance_correlation

    rng = np.random.RandomState(1)
    spatial = rng.randn(300, 2)
    embedding = rng.randn(300, 2)
    r1 = spatial_distance_correlation(spatial, embedding, sample_size=100, seed=42)
    r2 = spatial_distance_correlation(spatial, embedding, sample_size=100, seed=42)
    assert r1 == r2
