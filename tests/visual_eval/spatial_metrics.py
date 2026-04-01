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


