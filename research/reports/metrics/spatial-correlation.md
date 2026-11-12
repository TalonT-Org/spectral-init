# Category B: Spatial Correlation Metrics

These 5 metrics use physical spatial coordinates as ground truth, enabling benchmark questions no standard DR metric can answer. They quantify whether physically adjacent cells remain adjacent in UMAP space.

---

### Spatial Neighbor Agreement (SNA)

**Definition:** `SNA(k) = mean_i [ |NN_spatial(i,k) ∩ NN_embed(i,k)| / |NN_spatial(i,k) ∪ NN_embed(i,k)| ]`
(mean Jaccard similarity between spatial and embedding k-nearest-neighbor sets).

**What it measures:** Fraction of a cell's physical spatial neighbors that are also embedding neighbors — the core spatial-fidelity metric of this study.

**Range and interpretation:** [0, 1]. Expected range for UMAP: 0.05–0.30 (UMAP optimizes expression structure, not physical space). The Python–Rust comparison matters more than the absolute value.

**Computational complexity:** O(n·k). Full dataset feasible via kd-tree on 2D spatial coords.

**Implementation:** Custom (`sklearn.neighbors.NearestNeighbors` with `algorithm='kd_tree'`). See benchmark plan Section 8, B1 for reference implementation.

**Thresholds:** PASS gate: `SNA_rust >= SNA_python - 0.02`.

---

### Spatial Distance Correlation

**Definition:** `ρ = spearmanr(pdist(spatial_coords_sample), pdist(embedding_sample))`
Spearman rank correlation between pairwise physical distances and pairwise embedding distances.

**What it measures:** Global alignment between the physical tissue layout and the embedding — whether cells that are far apart in the brain are also far apart in UMAP.

**Range and interpretation:** [-1, 1]. Higher = better global spatial preservation.

**Computational complexity:** O(n²). Subsample 5K cells for large datasets.

**Implementation:** `scipy.stats.spearmanr(scipy.spatial.distance.pdist(spatial_sample), scipy.spatial.distance.pdist(embed_sample))`

**Thresholds:** PASS gate: `ρ_rust >= ρ_python - 0.02`.

---

### Moran's I (Embedding Dimensions)

**Definition:** `I = (n / W) · (Σ_i Σ_{j ∈ NN(i)} w_ij · z_i · z_j) / (Σ_i z_i²)`
where z = values − mean(values), W = Σ w_ij (total weight = n·k for uniform binary weights), and NN(i) is the k-nearest spatial neighbors of cell i.
Computed separately for each UMAP dimension; report max(I_dim0, I_dim1).

**What it measures:** Spatial autocorrelation of each UMAP coordinate — whether a UMAP dimension varies smoothly across the physical tissue (I near +1) or randomly (I near 0).

**Range and interpretation:** [-1, 1]. Near +1 = strong positive spatial autocorrelation (expected for a quality embedding). Near 0 = spatial randomness.

**Computational complexity:** O(n·k). Full dataset with sparse weights.

**Implementation:** Custom (see benchmark plan Section 8, B3) or `esda.Moran` / `squidpy.gr.spatial_autocorr`.

**Thresholds:** PASS gate: `max(I_rust) >= max(I_python) - 0.05`.

---

### CHAOS Score

**Definition:** `CHAOS = (Σ_c n_c · mean_1NN_dist(c)) / n`
weighted average over all clusters c of the mean 1-nearest-neighbor distance within cluster c in physical space.

**What it measures:** Spatial compactness of cell-type clusters — lower CHAOS means cells of the same type are physically closer together. Derived from SpatialPCA (Nature Communications 2022).

**Range and interpretation:** [0, ∞). Lower = better (more spatially compact clusters). Absolute value depends on physical coordinate scale; compare Python vs Rust embeddings at the same scale.

**Computational complexity:** O(n·k per cluster). Full dataset feasible.

**Implementation:** Custom (`sklearn.neighbors.NearestNeighbors(n_neighbors=2)`). See benchmark plan Section 8, B4 for reference implementation.

**Thresholds:** No explicit benchmark gate. Report `CHAOS_python - CHAOS_rust`; negative delta (Rust lower) is better.

---

### PAS Score (Percentage of Abnormal Spots)

**Definition:** `PAS = (1/n) · Σ_i 1[ mean(labels[NN_spatial(i,k)] == labels[i]) < (1 - threshold) ]`
where threshold = 0.6 and NN_spatial uses physical coordinates.

**What it measures:** Fraction of cells whose spatial neighborhood is dominated by cells of a different cluster — a cell-level spatial coherence check. Derived from SpatialPCA (Nature Communications 2022).

**Range and interpretation:** [0, 1]. Lower = better (fewer spatially anomalous cells).

**Computational complexity:** O(n·k). Full dataset feasible.

**Implementation:** Custom (`sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree')`). See benchmark plan Section 8, B5 for reference implementation.

**Thresholds:** No explicit benchmark gate. Report PAS for Python and Rust; lower is better.
