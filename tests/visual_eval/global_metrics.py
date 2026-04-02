"""tests/visual_eval/global_metrics.py — Category D global structure metrics.

Pure numpy/scipy/sklearn. No scanpy, anndata, or polars dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


def random_triplet_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_triplets: int = 50000,
    seed: int = 42,
) -> float:
    """Fraction of random triplets where the high-dim ordering matches the low-dim ordering.

    For each sampled triplet (anchor, pos, neg), checks whether the relative
    ordering of distances to pos/neg is preserved from high-dim to low-dim.

    Parameters
    ----------
    X_high : (n, d_high) float64 — high-dimensional coordinates
    X_low  : (n, 2) float64 — low-dimensional embedding
    n_triplets : number of candidate triplets to sample
    seed : random seed for reproducibility

    Returns
    -------
    float in [0.5, 1.0] — fraction of triplets with preserved ordering
    """
    rng = np.random.RandomState(seed)
    n = len(X_high)

    max_pts = 5000
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        Xh = X_high[idx]
        Xl = X_low[idx]
        n_sub = max_pts
    else:
        Xh = X_high
        Xl = X_low
        n_sub = n

    D_high = cdist(Xh, Xh)
    D_low = cdist(Xl, Xl)

    anchors = rng.randint(0, n_sub, n_triplets)
    pos = rng.randint(0, n_sub, n_triplets)
    neg = rng.randint(0, n_sub, n_triplets)

    valid = (anchors != pos) & (anchors != neg) & (pos != neg)
    anchors = anchors[valid]
    pos = pos[valid]
    neg = neg[valid]

    agree = (
        (D_high[anchors, pos] < D_high[anchors, neg])
        == (D_low[anchors, pos] < D_low[anchors, neg])
    )
    return float(agree.mean())


def shepard_correlation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    sample_n: int = 2000,
    seed: int = 42,
) -> dict:
    """Pearson and Spearman correlation between pairwise high-dim and low-dim distances.

    Parameters
    ----------
    X_high  : (n, d_high) float64
    X_low   : (n, 2) float64
    sample_n : subsample size when n > sample_n
    seed : random seed for subsampling

    Returns
    -------
    dict with keys "pearson" and "spearman", each a float in [-1, 1]
    """
    rng = np.random.RandomState(seed)
    n = len(X_high)

    if n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        Xh = X_high[idx]
        Xl = X_low[idx]
    else:
        Xh = X_high
        Xl = X_low

    d_high = pdist(Xh)
    d_low = pdist(Xl)

    if np.std(d_high) == 0.0 or np.std(d_low) == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan")}

    pearson_r, _ = pearsonr(d_high, d_low)
    spearman_r, _ = spearmanr(d_high, d_low)

    return {"pearson": float(pearson_r), "spearman": float(spearman_r)}


def centroid_distance_correlation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Spearman correlation between pairwise cluster centroid distances in high vs low dim.

    Parameters
    ----------
    X_high  : (n, d_high) float64
    X_low   : (n, 2) float64
    labels  : (n,) int — cluster assignments

    Returns
    -------
    float in [-1, 1]
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")
    centroids_high = np.array([X_high[labels == lbl].mean(axis=0) for lbl in unique_labels])
    centroids_low = np.array([X_low[labels == lbl].mean(axis=0) for lbl in unique_labels])

    d_high = pdist(centroids_high)
    d_low = pdist(centroids_low)

    rho, _ = spearmanr(d_high, d_low)
    return float(rho)


def knn_preservation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 15,
) -> float:
    """Mean fraction of k-nearest neighbors shared between high-dim and low-dim spaces.

    Parameters
    ----------
    X_high : (n, d_high) float64
    X_low  : (n, 2) float64
    k : number of neighbors to compare

    Returns
    -------
    float in [0, 1]
    """
    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k).fit(X_low)

    _, idx_high = nn_high.kneighbors(X_high)
    _, idx_low = nn_low.kneighbors(X_low)

    preservation = np.array([
        len(np.intersect1d(idx_high[i], idx_low[i])) / k
        for i in range(len(X_high))
    ])
    return float(np.mean(preservation))


def compute_global_metrics(
    X_high: np.ndarray,
    X_low: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Compute all Category D global structure metrics.

    Parameters
    ----------
    X_high : (n, d_high) float64
    X_low  : (n, 2) float64
    labels : (n,) int — cluster assignments
    k : number of neighbors for knn_preservation

    Returns
    -------
    dict with keys: triplet_accuracy, shepard_pearson, shepard_spearman,
                    centroid_dist_corr, knn_preservation
    """
    shep = shepard_correlation(X_high, X_low)
    return {
        "triplet_accuracy": random_triplet_accuracy(X_high, X_low),
        "shepard_pearson": shep["pearson"],
        "shepard_spearman": shep["spearman"],
        "centroid_dist_corr": centroid_distance_correlation(X_high, X_low, labels),
        "knn_preservation": knn_preservation(X_high, X_low, k=k),
    }
