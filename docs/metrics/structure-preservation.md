# Category A: Structure Preservation Metrics

These 5 metrics measure how faithfully the embedding preserves local and global high-dimensional structure, independent of spatial coordinates. They form the primary quality signal for comparing Python UMAP against Rust spectral initialization.

---

### Trustworthiness

**Definition:** `T(k) = 1 - (2 / (n·k·(2n - 3k - 1))) · Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)`
where U_k(i) is the set of k-NN in the embedding that are _not_ in the k-NN in high-dimensional space, and r(i,j) is the rank of j in high-dimensional space from i.

**What it measures:** Local neighborhood preservation — penalizes "false neighbors" in the embedding (points that appear close in 2D but are far in high-D).

**Range and interpretation:** [0, 1]. Higher = better. T > 0.90 = good; T > 0.95 = excellent.

**Computational complexity:** O(n²). Strategy for 4.2M cells: stratified subsample of 20K–50K cells.

**Implementation:** `sklearn.manifold.trustworthiness(X, embedding, n_neighbors=15)`

**Thresholds:** PASS gate: `|T_python - T_rust| < 0.01` (absolute delta between Python and Rust embeddings at each scale).

---

### Continuity

**Definition:** `C(k) = 1 - (2 / (n·k·(2n - 3k - 1))) · Σ_i Σ_{j ∈ V_k(i)} (r̂(i,j) - k)`
where V_k(i) is the set of k-NN in high-dimensional space that are _not_ in the k-NN in the embedding, and r̂(i,j) is the rank of j in embedding space from i.

**What it measures:** Complementary to trustworthiness — penalizes "missing neighbors" (points that were close in high-D but are pushed far apart in the embedding). Together T and C bound the neighborhood distortion.

**Range and interpretation:** [0, 1]. Higher = better. Interpret alongside Trustworthiness.

**Computational complexity:** O(n²). Subsample identically to Trustworthiness.

**Implementation:** `pyDRMetrics.DRMetrics(X, embedding).C` or `zadu.ZADU([{"id": "c_metric", ...}], X).fit(embedding)`

**Thresholds:** No explicit benchmark gate. Report Continuity for Python and Rust; flag if delta > 0.01 (same tolerance as Trustworthiness).

---

### Silhouette Score

**Definition:** `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
where a(i) = mean intra-cluster distance for point i, b(i) = mean distance to the nearest different cluster.
Overall score = mean s(i) over all points.

**What it measures:** Cluster separation quality in the embedding — how well-separated and cohesive the resulting clusters are.

**Range and interpretation:** [-1, 1]. Values near +1 = well-separated clusters; near 0 = overlapping; near -1 = misclassification.

**Computational complexity:** O(n²). Subsample 50K cells for large datasets.

**Implementation:** `sklearn.metrics.silhouette_score(embedding, labels)`

**Thresholds:** PASS gate: `|S_python - S_rust| < 0.05` (absolute delta).

---

### Procrustes Disparity

**Definition:** `d = ||X_std - Y_std||²_F`
after optimal rotation, reflection, and scaling alignment (each matrix normalized to unit Frobenius norm). Computed as the third return value of `scipy.spatial.procrustes`.

**What it measures:** Geometric shape similarity between two embeddings — measures how different the overall embedding "shape" is after accounting for arbitrary rotation, reflection, and scale.

**Range and interpretation:** [0, 1]. Near 0 = nearly identical shapes; near 1 = completely different shapes. Target: < 0.05 for Rust vs Python comparison.

**Computational complexity:** O(n) after distance matrices are formed (SVD of n×2 matrices). Full dataset is feasible.

**Implementation:** `scipy.spatial.procrustes(emb_python, emb_rust)[2]` — returns (mtx1, mtx2, disparity).

**Thresholds:** PASS gate: `disparity < 0.05`.

---

### Pairwise Distance Correlation

**Definition:** `r = pearsonr(pdist(emb_python), pdist(emb_rust))`
where pdist computes the condensed pairwise distance matrix.

**What it measures:** Whether relative inter-point distances are preserved between the Python and Rust embeddings — a global fidelity check.

**Range and interpretation:** [-1, 1]. Near +1 = distances are proportionally preserved. Target: > 0.99.

**Computational complexity:** O(n²). Subsample 5K cells for large datasets.

**Implementation:** `scipy.stats.pearsonr(scipy.spatial.distance.pdist(emb_py), scipy.spatial.distance.pdist(emb_rust))`

**Thresholds:** PASS gate: `r > 0.99`.
