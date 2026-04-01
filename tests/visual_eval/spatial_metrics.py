"""tests/visual_eval/spatial_metrics.py — Category B spatial correlation metrics.

Pure numpy/scipy/sklearn. No scanpy, anndata, or polars dependency.
Runnable as: python -m tests.visual_eval.spatial_metrics
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def spatial_neighbor_agreement(
    spatial_coords: np.ndarray,
    embedding: np.ndarray,
    k: int = 15,
) -> float:
    """Fraction of spatial k-NN neighbors preserved in the embedding (mean Jaccard).

    Parameters
    ----------
    spatial_coords:
        (n, 2) physical x,y coordinates.
    embedding:
        (n, d) low-dimensional embedding coordinates.
    k:
        Number of neighbors (excluding self).

    Returns
    -------
    float in [0, 1].
    """
    nn_spatial = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn_spatial.fit(spatial_coords)
    spatial_idx = nn_spatial.kneighbors(return_distance=False)[:, 1:]  # (n, k)

    nn_embed = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_embed.fit(embedding)
    embed_idx = nn_embed.kneighbors(return_distance=False)[:, 1:]  # (n, k)

    jaccards = np.empty(len(spatial_coords), dtype=np.float64)
    for i in range(len(spatial_coords)):
        s = set(spatial_idx[i])
        e = set(embed_idx[i])
        intersection = len(s & e)
        union = len(s | e)
        jaccards[i] = intersection / union if union > 0 else 0.0

    return float(np.mean(jaccards))


def spatial_distance_correlation(
    spatial_coords: np.ndarray,
    embedding: np.ndarray,
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Spearman rank correlation between pairwise spatial and embedding distances.

    Parameters
    ----------
    spatial_coords:
        (n, 2) physical coordinates.
    embedding:
        (n, d) embedding coordinates.
    sample_size:
        Max number of cells to subsample before computing pairwise distances.
    seed:
        Random seed for subsampling.

    Returns
    -------
    float in [-1, 1].
    """
    n = len(spatial_coords)
    rng = np.random.RandomState(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)

    d_spatial = pdist(spatial_coords[idx])
    d_embed = pdist(embedding[idx])
    rho, _ = spearmanr(d_spatial, d_embed)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def morans_i(
    spatial_coords: np.ndarray,
    values: np.ndarray,
    k: int = 6,
) -> float:
    """Moran's I spatial autocorrelation statistic using k-NN spatial weights.

    Parameters
    ----------
    spatial_coords:
        (n, 2) physical coordinates.
    values:
        (n,) scalar values at each cell.
    k:
        Number of spatial neighbors.

    Returns
    -------
    float in [-1, 1].
    """
    n = len(values)
    if n <= k:
        return 0.0
    z = values - np.mean(values)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn.fit(spatial_coords)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:]  # (n, k)

    w = float(n * k)  # sum of uniform binary weights
    neighbor_z_sum = z[neighbors].sum(axis=1)  # (n,)
    numerator = float(np.dot(z, neighbor_z_sum))
    denominator = float(np.sum(z ** 2))

    if denominator == 0.0:
        return 0.0
    return float((n / w) * (numerator / denominator))


def chaos_score(
    spatial_coords: np.ndarray,
    labels: np.ndarray,
) -> float:
    """CHAOS: mean nearest-neighbor distance within clusters (lower = more compact).

    Clusters with fewer than 2 cells are skipped.

    Parameters
    ----------
    spatial_coords:
        (n, 2) physical coordinates.
    labels:
        (n,) integer cluster IDs.

    Returns
    -------
    float >= 0.
    """
    unique_clusters = np.unique(labels)
    total_weighted = 0.0
    n_participating = 0

    for c in unique_clusters:
        mask = labels == c
        coords_c = spatial_coords[mask]
        n_c = len(coords_c)
        if n_c < 2:
            continue
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(coords_c)
        dists, _ = nn.kneighbors(coords_c)
        total_weighted += n_c * dists[:, 1].mean()
        n_participating += n_c

    if n_participating == 0:
        return 0.0
    return float(total_weighted / n_participating)


def pas_score(
    spatial_coords: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    threshold: float = 0.6,
) -> float:
    """PAS: fraction of cells whose spatial neighborhood is dominated by other labels.

    Parameters
    ----------
    spatial_coords:
        (n, 2) physical coordinates.
    labels:
        (n,) integer cluster IDs.
    k:
        Number of spatial neighbors (excluding self).
    threshold:
        A cell is anomalous if fewer than ``(1 - threshold)`` of its neighbors
        share its label.

    Returns
    -------
    float in [0, 1].
    """
    if len(spatial_coords) <= k:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn.fit(spatial_coords)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:]  # (n, k)

    same_label = labels[neighbors] == labels[:, None]  # (n, k)
    same_frac = same_label.mean(axis=1)  # (n,)
    n_anomalous = int(np.sum(same_frac < (1.0 - threshold)))
    return float(n_anomalous / len(labels))


def compute_spatial_metrics(
    spatial_coords: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    k_sna: int = 15,
    k_morans: int = 6,
    k_pas: int = 10,
) -> dict:
    """Compute all Category B spatial metrics and return a flat dict.

    Keys: sna, spatial_dist_corr, morans_i_max, morans_i_dim0, morans_i_dim1,
          chaos, pas.

    Parameters
    ----------
    spatial_coords:
        (n, 2) float64 physical coordinates.
    embedding:
        (n, 2) embedding coordinates.
    labels:
        (n,) integer cluster IDs.
    k_sna:
        k for spatial_neighbor_agreement.
    k_morans:
        k for morans_i.
    k_pas:
        k for pas_score.

    Returns
    -------
    dict with 7 float values.
    """
    mi_dim0 = morans_i(spatial_coords, embedding[:, 0], k=k_morans)
    mi_dim1 = morans_i(spatial_coords, embedding[:, 1], k=k_morans)
    return {
        "sna": spatial_neighbor_agreement(spatial_coords, embedding, k=k_sna),
        "spatial_dist_corr": spatial_distance_correlation(spatial_coords, embedding),
        "morans_i_max": max(mi_dim0, mi_dim1),
        "morans_i_dim0": mi_dim0,
        "morans_i_dim1": mi_dim1,
        "chaos": chaos_score(spatial_coords, labels),
        "pas": pas_score(spatial_coords, labels, k=k_pas),
    }


def _run_smoke_tests() -> None:
    """Run inline directionality and range checks without pytest."""
    from sklearn.datasets import make_blobs

    rng = np.random.RandomState(42)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])
    spatial_coords, labels = make_blobs(
        n_samples=240,
        centers=centers,
        cluster_std=0.8,
        random_state=42,
    )
    spatial_coords = spatial_coords.astype(np.float64)
    good_embedding = spatial_coords + rng.randn(*spatial_coords.shape) * 0.3
    random_embedding = rng.randn(*spatial_coords.shape) * 5.0

    # SNA range
    sna_good = spatial_neighbor_agreement(spatial_coords, good_embedding, k=15)
    assert isinstance(sna_good, float) and 0.0 <= sna_good <= 1.0, f"SNA range fail: {sna_good}"
    # SNA directionality
    sna_rand = spatial_neighbor_agreement(spatial_coords, random_embedding, k=15)
    assert sna_good > sna_rand, f"SNA directionality fail: {sna_good} <= {sna_rand}"

    # Distance correlation range
    dc_good = spatial_distance_correlation(spatial_coords, good_embedding, sample_size=200)
    assert isinstance(dc_good, float) and -1.0 <= dc_good <= 1.0, f"DC range fail: {dc_good}"
    # Distance correlation directionality
    dc_rand = spatial_distance_correlation(spatial_coords, random_embedding, sample_size=200)
    assert dc_good > dc_rand, f"DC directionality fail: {dc_good} <= {dc_rand}"

    # Moran's I range
    mi = morans_i(spatial_coords, good_embedding[:, 0], k=6)
    assert isinstance(mi, float) and -1.0 <= mi <= 1.0, f"Moran's I range fail: {mi}"
    # Moran's I directionality
    mi_rand = morans_i(spatial_coords, random_embedding[:, 0], k=6)
    assert mi > mi_rand, f"Moran's I directionality fail: {mi} <= {mi_rand}"

    # CHAOS nonnegative
    chaos_true = chaos_score(spatial_coords, labels)
    assert isinstance(chaos_true, float) and chaos_true >= 0.0, f"CHAOS range fail: {chaos_true}"
    # CHAOS directionality
    scrambled = rng.permutation(labels)
    chaos_scrambled = chaos_score(spatial_coords, scrambled)
    assert chaos_true < chaos_scrambled, f"CHAOS directionality fail: {chaos_true} >= {chaos_scrambled}"

    # PAS unit interval
    pas_true = pas_score(spatial_coords, labels, k=10)
    assert isinstance(pas_true, float) and 0.0 <= pas_true <= 1.0, f"PAS range fail: {pas_true}"
    # PAS directionality
    pas_scrambled = pas_score(spatial_coords, scrambled, k=10)
    assert pas_true < pas_scrambled, f"PAS directionality fail: {pas_true} >= {pas_scrambled}"

    # compute_spatial_metrics keys
    result = compute_spatial_metrics(spatial_coords, good_embedding, labels)
    required_keys = ["sna", "spatial_dist_corr", "morans_i_max", "morans_i_dim0", "morans_i_dim1", "chaos", "pas"]
    for key in required_keys:
        assert key in result and isinstance(result[key], float), f"Missing or wrong type for key: {key}"
    assert result["morans_i_max"] == max(result["morans_i_dim0"], result["morans_i_dim1"])

    # Singleton cluster edge case
    rng2 = np.random.RandomState(7)
    spatial_s = rng2.randn(51, 2)
    labels_s = np.array([0] + [1] * 50, dtype=np.int32)
    chaos_s = chaos_score(spatial_s, labels_s)
    assert isinstance(chaos_s, float) and chaos_s >= 0.0, f"Singleton CHAOS fail: {chaos_s}"


if __name__ == "__main__":
    import sys

    _run_smoke_tests()
    print("All smoke tests passed.")
    sys.exit(0)
