# Category D: Global Structure Metrics

These 4 metrics assess whether the global relative geometry of the high-dimensional data is preserved in the 2D embedding — testing structure at a scale larger than local neighborhoods.

---

### Random Triplet Accuracy

**Definition:** `RTA = (1/|T|) · Σ_{(i,j,k) ∈ T} 1[ (D_high(i,j) < D_high(i,k)) == (D_low(i,j) < D_low(i,k)) ]`
where T is a set of random triplets sampled from a subsample of n_sub points.

**What it measures:** Whether distance orderings between triples of points are preserved — i.e., if j is closer to i than k in high-D, is it also closer in the embedding?

**Range and interpretation:** [0.5, 1.0]. 0.5 = random baseline; 1.0 = perfect. UMAP typically 0.55–0.70; PaCMAP/TriMap: 0.75–0.90.

**Computational complexity:** O(n_triplets). Subsample 5K cells, 50K triplets.

**Implementation:** Custom (see benchmark plan Section 8, C3 for reference implementation using `scipy.spatial.distance.squareform` + `pdist`).

**Thresholds:** No explicit benchmark gate. Report for Python and Rust; similar values expected.

---

### Shepard Diagram Correlation

**Definition:** `r_pearson = pearsonr(pdist(X_high_sample), pdist(X_low_sample))`
`r_spearman = spearmanr(pdist(X_high_sample), pdist(X_low_sample))`
Both reported. The Shepard diagram is the scatter plot of these two distance vectors.

**What it measures:** Overall fidelity of the pairwise distance mapping from high-D to embedding — the classical stress-based quality measure, without the Python-vs-Rust comparison framing.

**Range and interpretation:** [-1, 1]. Near +1 = high global distance fidelity. Pearson tests linear relationship; Spearman tests rank-order preservation.

**Computational complexity:** O(n²). Subsample 2K–5K cells.

**Implementation:** Custom (`scipy.stats.pearsonr`, `scipy.stats.spearmanr`, `scipy.spatial.distance.pdist`). See benchmark plan Section 8, D1 for reference implementation.

**Thresholds:** No explicit benchmark gate. Report both coefficients for each embedding; values > 0.7 indicate reasonable global structure.

---

### Centroid Distance Correlation

**Definition:** `ρ = spearmanr(pdist(centroids_high), pdist(centroids_low))`
where `centroids_high[c] = mean(X_high[labels == c])` and similarly for the embedding.

**What it measures:** Whether the relative inter-cluster distances in high-D are preserved in the embedding — a cheaper O(c²) alternative to full pairwise distance correlation, using only cluster centroids.

**Range and interpretation:** [-1, 1]. Near +1 = cluster topology preserved. Cheaper than Shepard but coarser.

**Computational complexity:** O(c²) where c = number of clusters. Typically c ≪ n; full dataset feasible.

**Implementation:** `scipy.stats.spearmanr(scipy.spatial.distance.pdist(centroids_high), scipy.spatial.distance.pdist(centroids_low))`. See benchmark plan Section 8, D2 for reference implementation.

**Thresholds:** No explicit benchmark gate. Values > 0.90 expected for embeddings that preserve global cluster topology.

---

### kNN Preservation Rate

**Definition:** `kNNPR(k) = (1/n) · Σ_i (|NN_high(i,k) ∩ NN_low(i,k)| / k)`
where NN_high(i,k) and NN_low(i,k) are the k-nearest neighbors of point i in high-D and embedding respectively.

**What it measures:** The mean fraction of a point's high-dimensional k-nearest neighbors that survive in the 2D embedding — a direct local topology fidelity score. Closely related to Continuity (Category A, metric 2) but framed as recall: what fraction of high-D neighbors are preserved in the embedding.

**Range and interpretation:** [0, 1]. Higher = better. k = 15 for comparability with UMAP's own neighborhood parameter.

**Computational complexity:** O(n·k). Full dataset feasible with approximate NN (pynndescent).

**Implementation:** Custom (`sklearn.neighbors.NearestNeighbors` or `pynndescent.NNDescent` for large n).

**Thresholds:** No explicit benchmark gate. Report for Python and Rust; similar values expected.
