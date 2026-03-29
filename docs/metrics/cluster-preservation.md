# Category C: Cluster Preservation Metrics

These 3 metrics measure agreement between cluster labels derived from the original kNN graph and from the embedding's kNN graph.

---

### Adjusted Rand Index (ARI)

**Definition:** `ARI = (RI - E[RI]) / (max(RI) - E[RI])`
where RI is the Rand Index (fraction of pairs with consistent label assignments), and E[RI] is its expected value under random labeling.

**What it measures:** Agreement between original-graph Leiden cluster labels and embedding-space Leiden cluster labels, corrected for chance.

**Range and interpretation:** [-0.5, 1.0]. 1.0 = perfect agreement; 0.0 = random agreement; negative = worse than random. Target: > 0.90.

**Computational complexity:** O(n). Full dataset feasible.

**Implementation:** `sklearn.metrics.adjusted_rand_score(labels_original, labels_embedding)`

**Thresholds:** PASS gate: `ARI > 0.90` (applied to both Python and Rust embeddings independently).

---

### Normalized Mutual Information (NMI)

**Definition:** `NMI(U, V) = 2 · I(U; V) / (H(U) + H(V))`
where I(U;V) is mutual information between cluster assignments U and V, and H is entropy.

**What it measures:** Information-theoretic cluster agreement — how much knowing one clustering tells you about the other.

**Range and interpretation:** [0, 1]. 1.0 = perfect agreement; 0.0 = no shared information. Complements ARI.

**Computational complexity:** O(n). Full dataset feasible.

**Implementation:** `sklearn.metrics.normalized_mutual_info_score(labels_original, labels_embedding)`

**Thresholds:** No explicit benchmark gate. Report alongside ARI; value near 1 expected for high-quality embeddings.

---

### Cell-Type Purity

**Definition:** `Purity = (1/n) · Σ_c max_{t} |c ∩ t|`
where c ranges over Leiden clusters and t over annotated cell types. Weighted average: each cluster's dominant cell-type fraction, weighted by cluster size.

**What it measures:** How cleanly cell-type annotations align with embedding clusters — whether UMAP groups similar cell types together.

**Range and interpretation:** [0, 1]. Higher = better cluster purity.

**Computational complexity:** O(n). Full dataset feasible.

**Implementation:** Custom (group cells by Leiden cluster, compute per-cluster mode cell type fraction, weighted average).

**Thresholds:** No explicit benchmark gate. Report purity for Python and Rust embeddings; comparable values expected.
