"""Run subsampling experiment: Approach A and B for all (dataset, n, m, seed).

Full run: 2 approaches × 14 m-values × 10 seeds × 2 datasets = 560 trials.
Dry run (--dry-run): seed=0, m=2000, both approaches, MERFISH n=10K only.

Run from experiment root:
    micromamba run -n subsampled-tw-tradeoff python scripts/run_subsampling.py
    micromamba run -n subsampled-tw-tradeoff python scripts/run_subsampling.py --dry-run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_tw
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    K, SEEDS, M_VALUES_10K, M_VALUES_50K,
    load_npy_pair, save_result_json, trustworthiness_row_subsampled,
)

EXPROOT = Path(__file__).parent.parent
BATCH_SIZE = 5000
_MEM_LIMIT = 8 * 1024 ** 3  # 8 GB

DATASETS = [
    ("merfish",  10_000, EXPROOT / "data" / "merfish"),
    ("merfish",  50_000, EXPROOT / "data" / "merfish"),
    ("gaussian", 10_000, EXPROOT / "data" / "gaussian"),
    ("gaussian", 50_000, EXPROOT / "data" / "gaussian"),
]


def _m_values(n: int) -> list:
    return M_VALUES_10K if n == 10_000 else M_VALUES_50K


def _approach_a_batched(X, Y, k, query_idx):
    """Memory-safe batch variant of Approach A.

    Processes query_idx in chunks of BATCH_SIZE to avoid materialising
    the full (m, n) float64 distance matrix when m*n*8 > MEM_LIMIT.
    """
    n_full = X.shape[0]
    m = len(query_idx)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(Y)
    y_knn_all = nn.kneighbors(Y[query_idx], return_distance=False)  # (m, k+1)
    penalty = 0.0
    for b0 in range(0, m, BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, m)
        bq = query_idx[b0:b1]
        dist_b = pairwise_distances(X[bq], X)           # (b_len, n_full)
        for li, gi in enumerate(bq):
            dist_b[li, gi] = np.inf                     # exclude self
        ranks_b = np.argsort(np.argsort(dist_b, axis=1), axis=1) + 1
        mask_b = ranks_b <= k
        for li in range(len(bq)):
            gi = bq[li]
            y_knn = [j for j in y_knn_all[b0 + li] if j != gi][:k]
            for j_col in y_knn:
                if not mask_b[li, j_col]:
                    penalty += ranks_b[li, j_col] - k
    return 1.0 - 2.0 * penalty / (m * k * (2 * n_full - 3 * k - 1))


def _load_exact_T(dataset: str, n: int) -> float:
    path = EXPROOT / "results" / "raw" / f"exact_{dataset}_{n}.json"
    if not path.exists():
        sys.exit(
            f"ERROR: exact baseline not found: {path}\n"
            "Run compute_exact.py first."
        )
    with open(path) as f:
        return json.load(f)["T_exact"]


def _run_trial(X, Y, n, approach, m, seed, T_exact, dataset):
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=m, replace=False)

    if approach == "A":
        mem_bytes = m * n * 8
        if mem_bytes > _MEM_LIMIT:
            print(
                f"WARNING: Approach A ({m}×{n}) ≈ {mem_bytes/1e9:.1f} GB > 8 GB. "
                f"Using batched computation (BATCH_SIZE={BATCH_SIZE}).",
                file=sys.stderr,
            )
            t0 = time.perf_counter()
            T_sub = _approach_a_batched(X, Y, K, idx)
        else:
            t0 = time.perf_counter()
            T_sub = trustworthiness_row_subsampled(X, Y, K, idx)
        wall_s = time.perf_counter() - t0
    else:  # approach == "B"
        t0 = time.perf_counter()
        T_sub = sklearn_tw(X[idx], Y[idx], n_neighbors=K)
        wall_s = time.perf_counter() - t0

    T_sub = float(T_sub)
    delta_T = T_sub - float(T_exact)
    print(
        f"[approach {approach}] dataset={dataset} n={n} m={m} seed={seed} "
        f"→ T_sub={T_sub:.4f}",
        file=sys.stderr,
    )
    return {
        "approach": approach, "dataset": dataset, "n": n, "m": m,
        "seed": seed, "k": K,
        "T_sub": T_sub, "T_exact": float(T_exact),
        "delta_T": delta_T, "abs_delta_T": abs(delta_T), "wall_s": wall_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal run: seed=0, m=2000, both approaches, MERFISH n=10K.")
    args = parser.parse_args()

    # Load datasets (skip missing with warning)
    loaded = {}
    for dataset, n, data_dir in DATASETS:
        try:
            loaded[(dataset, n)] = load_npy_pair(data_dir, dataset, n)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping {dataset} n={n}", file=sys.stderr)

    if args.dry_run:
        trials = [
            ("merfish", 10_000, "A", 2000, 0),
            ("merfish", 10_000, "B", 2000, 0),
        ]
    else:
        trials = [
            (ds, n, approach, m, seed)
            for ds, n, _ in DATASETS if (ds, n) in loaded
            for approach in ["A", "B"]
            for m in _m_values(n)
            for seed in SEEDS
        ]

    # Load exact baselines only for datasets that appear in the trial list and
    # were successfully loaded (abort on missing so stale runs are caught early).
    needed_pairs = {(ds, n) for ds, n, *_ in trials if (ds, n) in loaded}
    exact_cache = {
        (ds, n): _load_exact_T(ds, n)
        for ds, n in needed_pairs
    }

    for ds, n, approach, m, seed in trials:
        if (ds, n) not in loaded:
            continue
        X, Y = loaded[(ds, n)]
        result = _run_trial(X, Y, n, approach, m, seed, exact_cache[(ds, n)], ds)
        out = EXPROOT / "results" / "raw" / f"sub_{approach}_{ds}_{n}_m{m}_s{seed}.json"
        save_result_json(out, result)


if __name__ == "__main__":
    main()
