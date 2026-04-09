import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Experiment-wide constants
# ---------------------------------------------------------------------------

K = 15
SEEDS = list(range(10))
M_VALUES_10K = [250, 500, 1000, 2000, 5000, 7500]
M_VALUES_50K = [250, 500, 1000, 2000, 5000, 7500, 10000, 25000]

# ---------------------------------------------------------------------------
# Approach A: row-subsampled trustworthiness estimator
# ---------------------------------------------------------------------------

def trustworthiness_row_subsampled(X, Y, k, query_idx):
    """Approach A: m query rows, distances to ALL n points.

    Unbiased estimator of full-n trustworthiness.
    Denominator m * k * (2n - 3k - 1) matches the full-n formula when m == n.
    """
    n = X.shape[0]
    m = len(query_idx)
    dist_X = pairwise_distances(X[query_idx], X)
    for i, gi in enumerate(query_idx):
        dist_X[i, gi] = np.inf  # exclude self
    ranks_X = np.argsort(np.argsort(dist_X, axis=1), axis=1) + 1
    x_knn_mask = ranks_X <= k  # (m, n) boolean
    # Request k+1 neighbors because kneighbors(Y[query_idx]) includes self (distance=0)
    # at some position; filtering it gives k actual non-self neighbors matching sklearn.
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(Y)
    y_knn_all = nn.kneighbors(Y[query_idx], return_distance=False)  # (m, k+1)
    penalty = 0.0
    for i in range(m):
        gi = query_idx[i]
        y_knn = [j for j in y_knn_all[i] if j != gi][:k]
        for j_col in y_knn:
            if not x_knn_mask[i, j_col]:
                penalty += ranks_X[i, j_col] - k
    denom = m * k * (2 * n - 3 * k - 1)
    return 1.0 - 2.0 * penalty / denom

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_npy_pair(data_dir, prefix, n):
    """Load {data_dir}/{prefix}_n{n}_x.npy and {prefix}_n{n}_y.npy.

    Returns (X, Y) as numpy arrays.
    """
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"{prefix}_n{n}_x.npy")
    Y = np.load(data_dir / f"{prefix}_n{n}_y.npy")
    return X, Y


def save_result_json(path, result_dict):
    """Write result_dict to path as JSON, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result_dict, f)
