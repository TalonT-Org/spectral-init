"""tests/visual_eval/cluster_metrics.py — Category C cluster preservation metrics.

Pure numpy/sklearn. No scanpy, anndata, or polars dependency.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _purity_from_assignments(
    kmeans_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> float:
    """Weighted mean per-cluster purity = (1/n) · Σ_c max_t |c ∩ t|."""
    n = len(reference_labels)
    total_max = 0
    for c in np.unique(kmeans_labels):
        mask = kmeans_labels == c
        _, counts = np.unique(reference_labels[mask], return_counts=True)
        total_max += int(counts.max())
    return float(total_max / n)


def cluster_ari(
    embedding: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int | None = None,
) -> float:
    """Adjusted Rand Index between KMeans clusters and reference labels.

    Parameters
    ----------
    embedding:
        (n, 2) UMAP coordinates.
    reference_labels:
        (n,) ground-truth cell-type IDs.
    n_clusters:
        Number of clusters. Defaults to the number of unique reference labels.

    Returns
    -------
    float in [-0.5, 1.0].
    """
    k = n_clusters if n_clusters is not None else len(np.unique(reference_labels))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = km.fit_predict(embedding)
    return float(adjusted_rand_score(reference_labels, kmeans_labels))


def cluster_nmi(
    embedding: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int | None = None,
) -> float:
    """Normalized Mutual Information between KMeans clusters and reference labels.

    Parameters
    ----------
    embedding:
        (n, 2) UMAP coordinates.
    reference_labels:
        (n,) ground-truth cell-type IDs.
    n_clusters:
        Number of clusters. Defaults to the number of unique reference labels.

    Returns
    -------
    float in [0, 1].
    """
    k = n_clusters if n_clusters is not None else len(np.unique(reference_labels))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = km.fit_predict(embedding)
    return float(normalized_mutual_info_score(reference_labels, kmeans_labels))


def celltype_purity(
    embedding: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int | None = None,
) -> float:
    """Weighted mean per-cluster purity of KMeans clusters w.r.t. reference labels.

    For each KMeans cluster, compute the fraction of cells belonging to the
    dominant reference label. Return the cluster-size-weighted mean across all
    clusters: (1/n) · Σ_c max_t |c ∩ t|.

    Parameters
    ----------
    embedding:
        (n, 2) UMAP coordinates.
    reference_labels:
        (n,) ground-truth cell-type IDs.
    n_clusters:
        Number of clusters. Defaults to the number of unique reference labels.

    Returns
    -------
    float in [0, 1].
    """
    k = n_clusters if n_clusters is not None else len(np.unique(reference_labels))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = km.fit_predict(embedding)
    return _purity_from_assignments(kmeans_labels, reference_labels)


def compute_cluster_metrics(
    embedding: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int | None = None,
    seed: int = 42,
) -> dict:
    """Compute all Category C cluster metrics with a single KMeans fit.

    Parameters
    ----------
    embedding:
        (n, 2) UMAP coordinates.
    reference_labels:
        (n,) ground-truth cell-type IDs.
    n_clusters:
        Number of clusters. Defaults to the number of unique reference labels.
    seed:
        Random seed for KMeans reproducibility.

    Returns
    -------
    dict with keys: ari, nmi, celltype_purity.
    """
    k = n_clusters if n_clusters is not None else len(np.unique(reference_labels))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans_labels = km.fit_predict(embedding)
    return {
        "ari": float(adjusted_rand_score(reference_labels, kmeans_labels)),
        "nmi": float(normalized_mutual_info_score(reference_labels, kmeans_labels)),
        "celltype_purity": _purity_from_assignments(kmeans_labels, reference_labels),
    }
