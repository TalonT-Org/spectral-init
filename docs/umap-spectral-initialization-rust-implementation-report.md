# UMAP Spectral Initialization: Complete Implementation Guide for Rust

**Date:** 2026-03-10
**Purpose:** Map out exactly what is needed to implement an exact clone of Python UMAP's spectral initialization in Rust, enabling the `umap-rs` crate to produce results comparable to the Python reference implementation.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why Spectral Initialization Is Everything](#2-why-spectral-initialization-is-everything)
3. [The Python UMAP Pipeline: End to End](#3-the-python-umap-pipeline-end-to-end)
4. [Spectral Initialization: Full Algorithm Specification](#4-spectral-initialization-full-algorithm-specification)
5. [The Rust UMAP Landscape](#5-the-rust-umap-landscape)
6. [Integration Point: `umap-rs` (wilsonzlin)](#6-integration-point-umap-rs-wilsonzlin)
7. [Rust Ecosystem: Sparse Matrices & Eigensolvers](#7-rust-ecosystem-sparse-matrices--eigensolvers)
8. [Implementation Plan](#8-implementation-plan)
9. [Component-by-Component Build & Verify Plan](#9-component-by-component-build--verify-plan)
10. [Fortran Performance Parity & SIMD Optimization](#10-fortran-performance-parity--simd-optimization)
11. [Mathematical Reference](#11-mathematical-reference)
12. [Risk Assessment & Mitigations](#12-risk-assessment--mitigations)
13. [Phase 1 Completion & Phase 2 Status](#13-phase-1-completion--phase-2-status)
14. [Phase 2 Implementation Audit (2026-03-21)](#14-phase-2-implementation-audit-2026-03-21)
15. [Numerical Accuracy Improvement Campaign (2026-03-22)](#15-numerical-accuracy-improvement-campaign-2026-03-22)
16. [Sources](#16-sources)

---

## 1. Executive Summary

Python UMAP (`umap-learn`) consistently produces superior embeddings compared to every reimplementation in other languages. The single differentiating factor is **spectral initialization** — using Laplacian eigenvectors of the fuzzy k-NN graph as the starting coordinates for SGD optimization.

The primary Rust UMAP crate (`umap-rs` by wilsonzlin) deliberately omits spectral initialization, accepting a pre-computed `init: Array2<f32>` from the caller. It performs terribly with random or PCA initialization because UMAP's non-convex SGD optimization gets trapped in poor local minima without a globally-aware starting point.

**The fix is straightforward in architecture but requires careful numerical implementation:**
1. After `umap-rs` constructs its fuzzy simplicial set (the `LearnedManifold`), extract the graph via `manifold.graph()`
2. Build the symmetric normalized Laplacian: `L = I - D^{-1/2} W D^{-1/2}`
3. Compute the `n_components` smallest non-trivial eigenvectors using an iterative sparse eigensolver
4. Scale to `[0, 10]` per dimension, add tiny noise
5. Pass as `init` to `Optimizer::new()`

The main challenge is step 3: the Rust ecosystem lacks a mature, production-ready sparse symmetric eigensolver equivalent to SciPy's `eigsh`. The solution is a **solver escalation chain** — LOBPCG first (via `ndarray-linalg`), then LOBPCG with regularization, then randomized SVD (via the `2I - L` transform, pure Rust), then dense EVD as the nuclear option. The eigenvectors always exist (spectral theorem); our job is to find them. **No random fallback, ever.** Python UMAP's random fallback is a bug we will not reproduce.

---

## 2. Why Spectral Initialization Is Everything

### 2.1 The Mathematical Proof

A 2025 paper ("UMAP Is Spectral Clustering on the Fuzzy Nearest-Neighbor Graph") proved via **Theorem 3.1** that UMAP's cross-entropy loss decomposes as:

```
L_UMAP(Y) = 2a * tr(Y^T L(V) Y) + L_repel(Y)
```

The attractive term is a **Laplacian quadratic form**. Spectral initialization solves:

```
Y_init = argmin_{Z^T Z = I} tr(Z^T L_sym(V) Z)
```

This is the **exact global minimizer** of the linearized UMAP loss (ignoring the repulsive term). SGD then performs local nonlinear refinement of this globally-optimal spectral solution.

### 2.2 Why Random Initialization Fails

UMAP's SGD updates are **local** — they operate on individual edges, pulling connected nodes together and pushing non-connected nodes apart. Once the global topology is scrambled by random placement, local updates have **no mechanism to reorganize it at a global scale**.

A 2021 Nature Biotechnology paper confirmed empirically that UMAP's alleged superiority over t-SNE in preserving global structure was **entirely attributable to UMAP using spectral initialization** while t-SNE used random initialization. When both used the same initialization, results were equivalent.

### 2.3 The Mass-Spring Analogy

Think of graph nodes as masses and edges as springs with stiffness proportional to edge weight. The eigenvectors of the graph Laplacian are the **normal modes of vibration**:
- The 1st mode (eigenvalue 0) is trivial uniform translation
- The 2nd mode (Fiedler vector) is the fundamental oscillation — it bisects the graph optimally
- The 3rd, 4th, ... modes capture progressively finer structural detail

Using eigenvectors 2 through `k+1` as initial coordinates places every node at a position that reflects the graph's **global community structure** before SGD begins its local refinement.

---

## 3. The Python UMAP Pipeline: End to End

### 3.1 Full Pipeline Diagram

```
Raw data X (n_samples x n_features)
    |
    v
[1] nearest_neighbors()
    |   - n < 4096: exact pairwise distances
    |   - n >= 4096: PyNNDescent (approximate)
    |   - Returns: knn_indices (n x k), knn_dists (n x k)
    v
[2] smooth_knn_dist()
    |   - Binary search (64 iters) for per-point sigma_i
    |   - Target: sum of weights = log2(k)
    |   - Computes rho_i = distance to nearest neighbor
    |   - Returns: sigmas (n,), rhos (n,)
    v
[3] compute_membership_strengths()
    |   - val = exp(-max(0, d - rho_i) / sigma_i)
    |   - Self-loops: val = 0
    |   - Within rho: val = 1.0
    |   - Returns: directed sparse COO matrix (n x n)
    v
[4] Fuzzy union (symmetrize)
    |   - A_sym = A + A^T - A * A^T  (probabilistic OR)
    |   - eliminate_zeros()
    |   - Returns: self.graph_ (symmetric COO, values in [0,1])
    v
[5] simplicial_set_embedding()
    |   [5a] Edge pruning: remove weights < max_weight / n_epochs
    |   [5b] Spectral initialization (THIS IS THE KEY STEP)
    |   [5c] noisy_scale_coords: scale to max_abs=10, add N(0, 0.0001) noise
    |   [5d] Min-max rescale to [0, 10] per dimension
    v
[6] optimize_layout_euclidean()
        - SGD with negative sampling
        - Linear learning rate decay
        - Returns: final embedding (n x n_components)
```

### 3.2 What Feeds Into Spectral Initialization

The spectral initialization function receives:

| Input | Type | Description |
|-------|------|-------------|
| `data` | `ndarray` | Original high-dim data (only used for multi-component centroid computation) |
| `graph` | `scipy.sparse.coo_matrix` | Symmetrized, edge-pruned fuzzy simplicial set |
| `dim` | `int` | Target embedding dimensions (e.g., 2) |
| `random_state` | `RandomState` | For reproducibility |

The graph at this point is:
- **Shape**: `(n_samples, n_samples)`, square, symmetric
- **Format**: COO sparse
- **Values**: `float32` in `(0, 1]` — fuzzy membership strengths
- **Diagonal**: zero (no self-loops)
- **Sparsity**: ~`k * n_samples` non-zeros (before symmetrization doubles some), minus pruned edges

---

## 4. Spectral Initialization: Full Algorithm Specification

### 4.0 Source Location

File: `umap/spectral.py` in the `lmcinnes/umap` repository.

### 4.1 Step 1: Connected Components Check

```python
n_components, labels = scipy.sparse.csgraph.connected_components(graph)
if n_components > 1:
    return multi_component_layout(data, graph, n_components, labels, dim, ...)
```

If the graph is disconnected, route to the multi-component handler (Section 4.6).

### 4.2 Step 2: Build Symmetric Normalized Laplacian

```python
# Degree vector (weighted degree = sum of edge weights per node)
sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())

# Build L = I - D^{-1/2} * A * D^{-1/2}
I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
L = I - D * graph * D
```

**Why the symmetric normalized Laplacian (not unnormalized or random-walk)?**
- It is **symmetric** → standard symmetric eigensolvers apply (Lanczos, not general Arnoldi)
- Eigenvalues lie in `[0, 2]` → bounded numerical conditioning
- Eigenvectors relate to random-walk Laplacian via `v_sym = D^{1/2} v_rw`

### 4.3 Step 3: Eigensolver Initialization

```python
k = dim + 1   # Request dim+1 eigenvectors (one will be trivial)
num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
```

**Warm-start vector construction:**

For `init="random"` (default path via `spectral_layout`):
```python
X = gen.normal(size=(L.shape[0], k))
```

For `init="tsvd"` (via `tswspectral_layout`):
```python
X = TruncatedSVD(n_components=k).fit_transform(L)
```

**Critical optimization — inject known first eigenvector:**
```python
X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)
```

The first eigenvector of the symmetric normalized Laplacian is always proportional to `sqrt(degree_i)`. Injecting this analytically-known vector accelerates convergence.

### 4.4 Step 4: Eigendecomposition

**Solver selection:**
```python
method = "eigsh" if L.shape[0] < 2_000_000 else "lobpcg"
```

**ARPACK path (`eigsh`, for n < 2M):**
```python
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
    L,
    k,                          # number of eigenvalues
    which="SM",                 # smallest magnitude
    ncv=num_lanczos_vectors,    # Krylov subspace dimension
    tol=1e-4,                   # convergence tolerance
    v0=np.ones(L.shape[0]),     # starting vector (all-ones)
    maxiter=graph.shape[0] * 5, # max iterations
)
```

Under the hood, `which="SM"` triggers ARPACK's **shift-invert mode**: it solves `(L - 0*I)^{-1} v = nu * v`, transforming small eigenvalues `lambda ~ 0` into large values `nu = 1/lambda`, which ARPACK converges to quickly.

**LOBPCG path (for n >= 2M or `tswspectral`):**
```python
with warnings.catch_warnings():
    warnings.filterwarnings(
        category=UserWarning,
        message=r"(?ms).*not reaching the requested tolerance",
        action="error",   # Convert tolerance warning → exception → triggers fallback
    )
    eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
        L,
        np.asarray(X),    # (n, k) initial approximation
        largest=False,     # find smallest eigenvalues
        tol=1e-4,
        maxiter=5 * graph.shape[0],
    )
```

### 4.5 Step 5: Eigenvector Selection

```python
order = np.argsort(eigenvalues)[1:k]   # Skip eigenvalue 0 (trivial)
return eigenvectors[:, order]           # Shape: (n_samples, dim)
```

The first eigenvector (eigenvalue ~0, corresponding to the connected component) is discarded. The remaining `dim` eigenvectors — the **Fiedler vectors** — form the initial embedding coordinates.

### 4.6 Step 6: Disconnected Graph Handling (`multi_component_layout`)

When `n_components > 1`:

**Phase A — Meta-embedding of component positions:**
- If `n_components > 2 * dim`:
  - Compute centroids of each component in original data space
  - Build Gaussian affinity matrix: `exp(-dist^2)` between centroids
  - Run `sklearn.manifold.SpectralEmbedding(n_components=dim, affinity="precomputed")`
  - Normalize by max: `component_embedding /= component_embedding.max()`
- If `n_components <= 2 * dim`:
  - Use deterministic orthogonal placement (identity matrix rows and negatives)

**Phase B — Per-component spectral embedding:**
- For components with `>= 2*dim` nodes and `> dim+1` nodes:
  - Extract subgraph
  - Run `_spectral_layout` on subgraph (using the same solver escalation chain)
  - Scale to `data_range` (half the nearest inter-component distance in meta-embedding)
  - Translate to meta-embedding position
- For tiny components (fewer nodes than dimensions needed for spectral):
  - Python uses random positions here — this is one of the few places random is *mathematically justified*, because a 2-node component embedded in 2D genuinely has 1 degree of freedom, not 2. Dense EVD on the 2x2 subgraph Laplacian gives the 1 non-trivial eigenvector; the remaining axis can be set to 0 (or use the data-space coordinate if available). For a single isolated node, place it at the component's meta-embedding centroid.

### 4.7 Step 7: Python's Fallback (What We Will NOT Do)

Python UMAP gives up and falls back to random initialization:

```python
except (scipy.sparse.linalg.ArpackError, UserWarning):
    warn("Spectral initialisation failed! The eigenvector solver failed. "
         "This is likely due to too small an eigengap. Consider adding some "
         "noise or jitter to your data. Falling back to random initialisation!")
    return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
```

**This is a bug, not a feature.** The eigenvectors always exist — the Laplacian is real symmetric and the spectral theorem guarantees a complete orthonormal eigenbasis. Failure to find them is a solver implementation problem, not a mathematical impossibility.

**Known causes of solver failure (all addressable):**
- **Small eigengap** → Increase tolerance, use LOBPCG (more robust to clustering), or add `epsilon * I` regularization to separate eigenvalue 0 from eigenvalue ~0
- **Near-duplicate data points** → Deduplicate before graph construction, or add controlled jitter to the graph weights (not the data)
- **ARPACK iteration limit** → Use LOBPCG or randomized SVD which have different convergence characteristics
- **Numerical instability** → Build Laplacian in f64 even when graph is f32; use reorthogonalization

**Our approach: solver escalation chain** (Section 8.2, Phase 4). Every level addresses a different failure mode. The chain is exhaustive — it will always produce spectral coordinates.

### 4.8 Step 8: Post-Processing

**Immediately after spectral_layout returns (in `simplicial_set_embedding`):**

```python
# Step 1: Scale to max_abs=10, add tiny noise
def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(np.float32)

# Step 2: Min-max rescale to [0, 10] per axis
embedding = (
    10.0
    * (embedding - np.min(embedding, 0))
    / (np.max(embedding, 0) - np.min(embedding, 0))
).astype(np.float32, order="C")
```

**Note:** `umap-rs` already performs step 2 internally in `Optimizer::new()` (normalizes init to `[0, 10]` per dimension), so only step 1 (the noisy scaling) needs to be done before passing init to `umap-rs`.

---

## 5. The Rust UMAP Landscape

### 5.1 Available Crates

| Crate | GitHub | Spectral Init | Notes |
|-------|--------|---------------|-------|
| **`umap-rs`** (v0.4.5) | `wilsonzlin/umap-rs` | **No** — caller must supply | High-perf, minimal, designed for 100M+ samples |
| `fast-umap` | `eugenehp/fast-umap` | No | Parametric/neural UMAP, GPU via Burn/WGPU |
| `annembed` | crate only | **Yes** — via diffusion maps / randomized SVD | UMAP-like, includes spectral init |
| `rag-umap` | `richardanaya/rag-umap` | **Partial** — power iteration | Pure Rust, basic implementation |

### 5.2 Why `umap-rs` Is the Target

`umap-rs` is the only Rust crate that:
- Faithfully reimplements the core UMAP optimization (fuzzy simplicial set + Hogwild SGD)
- Is designed for production-scale workloads (parallel via Rayon)
- Has an explicit, documented API for accepting external initialization
- Shares mathematically identical graph construction and SGD update rules with Python UMAP

---

## 6. Integration Point: `umap-rs` (wilsonzlin)

### 6.1 The `fit()` API

```rust
pub fn fit(
    &self,
    data: ArrayView2<f32>,          // n_samples x n_features (validation only)
    knn_indices: ArrayView2<u32>,   // n_samples x n_neighbors (PRE-COMPUTED)
    knn_dists: ArrayView2<f32>,     // n_samples x n_neighbors (PRE-COMPUTED)
    init: ArrayView2<f32>,          // n_samples x n_components (PRE-COMPUTED INIT)
) -> FittedUmap
```

### 6.2 Two-Phase API (Preferred for Spectral Init)

```rust
// Phase 1: Learn the manifold (deterministic, cacheable)
let manifold: LearnedManifold = umap.learn_manifold(
    data.view(), knn_indices.view(), knn_dists.view(),
);

// Phase 2: Access the graph for spectral init
let graph: &CsMatI<f32, u32, usize> = manifold.graph();  // sprs CSR sparse matrix

// Phase 3: Compute spectral initialization externally
let spectral_init: Array2<f32> = compute_spectral_embedding(graph, n_components);

// Phase 4: Create optimizer with spectral init
let mut opt = Optimizer::new(
    manifold, spectral_init.view(), total_epochs, &config, metric_type,
);

// Phase 5: Run SGD
while opt.remaining_epochs() > 0 {
    opt.step_epochs(10, &metric);
}
let fitted = opt.into_fitted(config);
```

### 6.3 Key Data Structure Details

- **Graph format**: `CsMatI<f32, u32, usize>` from `sprs` — CSR with `u32` column indices
- **Init type**: `ArrayView2<f32>` — shape `(n_samples, n_components)`, `f32` precision
- **Init preprocessing in `Optimizer::new()`**: Normalizes to `[0, 10]` per dimension (matches Python UMAP's final rescaling)

### 6.4 What We Need to Implement

A function with this signature:

```rust
fn spectral_init(
    graph: &CsMatI<f32, u32, usize>,   // Fuzzy simplicial set from umap-rs
    n_components: usize,                 // Target embedding dimensions
    seed: u64,                           // Random seed for reproducibility
) -> Array2<f32>                         // (n_samples, n_components) initial coordinates
```

This function must:
1. Check connected components of the graph
2. Build the symmetric normalized Laplacian from the graph
3. Compute the `n_components` smallest non-trivial eigenvectors — **guaranteed to converge** via solver escalation (see Section 8)
4. Handle disconnected components (multi-component layout)
5. Scale results and add noise

**No random fallback. Ever.** If the first solver doesn't converge, escalate to the next one. The eigenvectors exist — the math guarantees it. Our job is to find them.

---

## 7. Rust Ecosystem: Sparse Matrices & Eigensolvers

### 7.1 Sparse Matrix Crates

| Crate | Version | Formats | Eigensolvers | LAPACK | Status |
|-------|---------|---------|--------------|--------|--------|
| **`sprs`** | 0.11.4 | CSR, CSC, COO | None | No | Active, 500k downloads/mo |
| `nalgebra-sparse` | — | CSR, CSC, COO | None | No | Active |
| **`faer`** | 0.24.0 | Dense; sparse via `faer-sparse` | Dense only (`self_adjoint_eigen`) | No (self-contained) | Very active |

**Recommendation:** Use `sprs` for the Laplacian (matches `umap-rs`'s internal format). Use `faer` as dense fallback for small datasets.

### 7.2 Eigensolver Options

#### Option A: LOBPCG via `ndarray-linalg` (RECOMMENDED)

**Crate:** `ndarray-linalg` v0.18.1 (active, 64k downloads/mo)

**Key feature:** The `lobpcg()` function accepts a **closure-based operator** — you pass a function that does sparse matrix-vector multiply, so you never densify the Laplacian:

```rust
use ndarray_linalg::lobpcg::{lobpcg, Order};

// Build Laplacian as sprs CSR matrix
let laplacian: CsMat<f64> = build_normalized_laplacian(&graph);

// LOBPCG with sparse operator via closure
let op = |x: ArrayView2<f64>| -> Array2<f64> {
    sparse_mat_mul_dense(&laplacian, &x)
};

let result = lobpcg(
    op,              // sparse matrix-vector product as closure
    x_init,          // (n, k) initial approximation
    |_| {},          // preconditioner (identity = none)
    None,            // constraints
    1e-4,            // tolerance
    500,             // max iterations
    Order::Smallest, // find smallest eigenvalues
);
```

**SPD requirement:** LOBPCG requires **strictly positive definite** matrices. The graph Laplacian is positive **semi**-definite (eigenvalue 0 exists). Standard practice: add regularization `L + epsilon * I` where `epsilon ~ 1e-7`. This shifts all eigenvalues by epsilon, preserving eigenvector ordering while making the matrix SPD. This is Level 2 in our escalation chain; Level 1 attempts LOBPCG without regularization first (it often works despite the theoretical requirement, since the trivial eigenvector is being discarded anyway).

**Dependency:** Requires LAPACK (openblas-src, netlib-src, or intel-mkl-src).

#### Option B: Randomized SVD via `annembed` (PURE RUST)

**The 2I - L Trick (used by uwot/R):**

For the normalized Laplacian `L = I - D^{-1/2} A D^{-1/2}`:
- The matrix `M = 2I - L = I + D^{-1/2} A D^{-1/2}` has **largest** eigenvectors that correspond to **smallest** eigenvectors of `L`
- This converts the hard "smallest eigenvalues" problem into a "largest singular values" problem
- Solvable by randomized SVD (Halko-Tropp algorithm), which is well-implemented for sparse matrices

**The `annembed` crate** implements randomized SVD for `sprs` CSR matrices:
- `subspace_iteration_csr` — QR-stabilized, more accurate
- `adaptative_range_finder_matrep` — faster, less accurate for very large matrices

```rust
// Pseudocode for the 2I-L approach
let m = two_i_minus_l(&laplacian);  // Form M = 2I - L as sprs CSR
let (u, s, _vt) = randomized_svd(&m, n_components + 1, oversampling);
// u[:, 0..n_components] are the spectral embedding coordinates
// (skip column 0 which corresponds to the trivial eigenvector)
```

**Advantage:** No LAPACK dependency. Pure Rust. Already production-tested in `annembed`.

#### Option C: Dense Fallback via `faer` (Small Datasets Only)

For `n < ~2000–3000`:

```rust
use faer::Mat;

let dense_laplacian: Mat<f64> = sparse_to_dense(&laplacian);
let evd = dense_laplacian.self_adjoint_eigen(faer::Side::Lower);
// evd.eigenvalues() and evd.eigenvectors() give the full spectrum
// Take columns corresponding to eigenvalues 1..n_components+1
```

**Performance:** O(n^3) time and O(n^2) memory. Infeasible for n > ~5000.

#### Option D: Implement LOBPCG from Scratch

The LOBPCG algorithm is ~150 lines of dense linear algebra:
1. Compute residuals `R = A*X - X*diag(rho)` where `rho` = Rayleigh quotients
2. Orthogonalize `[X, R, P_prev]` via QR or Cholesky
3. Solve small `(3k x 3k)` generalized eigenvalue problem (Rayleigh-Ritz)
4. Extract best `k` eigenpairs as new `X`
5. Repeat until convergence

The `ndarray-linalg` implementation is MIT-licensed and can serve as a reference.

### 7.3 Crates to AVOID

| Crate | Why |
|-------|-----|
| `arpack-ng` (v0.2.2) | Only exposes `zneupd` (complex arithmetic), not `dseupd` (real symmetric) |
| `eigs` (v0.0.3) | Near-abandoned, complex-only API |
| `eigenvalues` (v0.4.0) | Abandoned since 2021, Davidson method poorly suited for graph Laplacians |
| `scirs2-sparse` (v0.3.1) | Comprehensive API but **suspected LLM-generated code** (2.59M lines by 2 contributors); numerical correctness unvalidated. Community raised concerns directly. |

### 7.4 Ecosystem Summary

```
                    Solver Escalation Chain (guaranteed convergence)
                                    |
            Level 0 (n<2k)          |
            faer dense EVD ─────────┤ exact, cannot fail for small n
                                    |
            Level 1                 |
            ndarray-linalg LOBPCG ──┤ fast, closure-based sparse operator
                                    |
            Level 2                 |
            LOBPCG + ε*I reg ───────┤ widens eigengap, fixes near-singular L
                                    |
            Level 3                 |
            Randomized SVD ─────────┤ 2I-L trick, entirely different algorithm
            (annembed / custom)     │ pure Rust, no LAPACK needed
                                    |
            Level 4 (nuclear)       |
            Dense EVD forced ───────┘ O(n³) but spectral theorem guarantees it
                                      .expect() = true assertion, not fallback
```

---

## 8. Implementation Plan

### 8.1 Architecture Decision: Standalone Crate

Create a new crate `umap-spectral` (or similar) that:
- Takes a `sprs` CSR graph as input
- Returns an `ndarray` `Array2<f32>` as output
- Has no dependency on `umap-rs` internals (operates on the public `graph()` accessor)
- Can be used by any Rust UMAP implementation

### 8.2 Implementation Steps

#### Phase 1: Core Single-Component Spectral Init

**Step 1.1 — Connected components detection**

Implement BFS/DFS on the CSR graph to find connected components. `sprs` does not include graph connectivity utilities, so this must be implemented directly (~30 lines).

```rust
fn connected_components(graph: &CsMat<f32>) -> (usize, Vec<usize>) {
    let n = graph.rows();
    let mut labels = vec![usize::MAX; n];
    let mut component = 0;
    let mut queue = VecDeque::new();

    for start in 0..n {
        if labels[start] != usize::MAX { continue; }
        labels[start] = component;
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            for (neighbor, _weight) in graph.outer_view(node).unwrap().iter() {
                if labels[neighbor] == usize::MAX {
                    labels[neighbor] = component;
                    queue.push_back(neighbor);
                }
            }
        }
        component += 1;
    }
    (component, labels)
}
```

**Step 1.2 — Normalized Laplacian construction**

```rust
fn build_normalized_laplacian(graph: &CsMat<f32>) -> CsMat<f64> {
    let n = graph.rows();

    // Compute degree vector (sum of edge weights per row)
    let degrees: Vec<f64> = (0..n).map(|i| {
        graph.outer_view(i).unwrap().iter()
            .map(|(_, &w)| w as f64)
            .sum::<f64>()
    }).collect();

    // Compute D^{-1/2}
    let inv_sqrt_deg: Vec<f64> = degrees.iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Build L = I - D^{-1/2} * A * D^{-1/2}
    // For each non-zero entry A[i,j]:
    //   L[i,j] = -inv_sqrt_deg[i] * A[i,j] * inv_sqrt_deg[j]  (off-diagonal)
    //   L[i,i] = 1.0 - inv_sqrt_deg[i]^2 * A[i,i]             (diagonal, but A[i,i]=0)
    //          = 1.0                                             (since no self-loops)

    let mut triplets = Vec::new();

    // Identity diagonal
    for i in 0..n {
        triplets.push((i, i, 1.0_f64));
    }

    // Off-diagonal: -D^{-1/2} * A * D^{-1/2}
    for i in 0..n {
        if let Some(row) = graph.outer_view(i) {
            for (j, &w) in row.iter() {
                let val = -(inv_sqrt_deg[i] * (w as f64) * inv_sqrt_deg[j]);
                triplets.push((i, j, val));
            }
        }
    }

    // Build CSR from triplets (summing duplicates for the diagonal)
    TriMat::from_triplets((n, n), triplets).to_csr()
}
```

**Step 1.3 — Eigensolver (LOBPCG path)**

```rust
fn spectral_embedding(
    laplacian: &CsMat<f64>,
    n_components: usize,
    seed: u64,
) -> Result<Array2<f64>, SpectralError> {
    let n = laplacian.rows();
    let k = n_components + 1;  // +1 for trivial eigenvector

    // Add small regularization for SPD requirement
    let epsilon = 1e-7;
    let l_reg = add_diagonal(&laplacian, epsilon);

    // Sparse matrix-vector product closure
    let op = |x: ArrayView2<f64>| -> Array2<f64> {
        let mut result = Array2::zeros(x.dim());
        for col_idx in 0..x.ncols() {
            let x_col = x.column(col_idx);
            let mut y_col = result.column_mut(col_idx);
            // sprs sparse-dense multiply
            for (row, row_vec) in l_reg.outer_iterator().enumerate() {
                let mut sum = 0.0;
                for (j, &val) in row_vec.iter() {
                    sum += val * x_col[j];
                }
                y_col[row] = sum;
            }
        }
        result
    };

    // Random initial vectors
    let mut rng = StdRng::seed_from_u64(seed);
    let x_init = Array2::from_shape_fn((n, k), |_| rng.gen::<f64>());

    // Inject known first eigenvector (sqrt of degrees, normalized)
    // ... (see Python implementation for details)

    match lobpcg(op, x_init, |_| {}, None, 1e-4, n * 5, Order::Smallest) {
        Ok(result) => {
            let eigenvalues = result.eigenvalues;
            let eigenvectors = result.eigenvectors;

            // Sort by eigenvalue, skip trivial first
            let mut indices: Vec<usize> = (0..k).collect();
            indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

            // Take eigenvectors 1..n_components+1
            let mut embedding = Array2::zeros((n, n_components));
            for (col, &idx) in indices[1..=n_components].iter().enumerate() {
                embedding.column_mut(col).assign(&eigenvectors.column(idx));
            }

            Ok(embedding)
        }
        Err(_) => Err(SpectralError::ConvergenceFailure),
    }
}
```

**Step 1.4 — Scaling and noise**

```rust
fn noisy_scale_coords(
    coords: &mut Array2<f32>,
    rng: &mut impl Rng,
    max_coord: f32,
    noise: f32,
) {
    let abs_max = coords.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if abs_max > 0.0 {
        let expansion = max_coord / abs_max;
        coords.mapv_inplace(|x| x * expansion);
    }

    let normal = Normal::new(0.0, noise as f64).unwrap();
    coords.mapv_inplace(|x| x + normal.sample(rng) as f32);
}
```

#### Phase 2: Multi-Component Layout

Implement `multi_component_layout`:

1. Component centroid computation (in original data space)
2. Inter-component affinity matrix: `exp(-dist^2)` between centroids
3. Spectral embedding of centroid graph (small, can use dense EVD via `faer`)
4. Per-component spectral embedding of subgraphs
5. Scale and translate each component

This is more complex but applies the same core spectral algorithm to smaller subproblems.

#### Phase 3: Randomized SVD Eigensolver (Second Line of Defense)

The `2I - L` trick converts smallest-eigenvalue problems into largest-eigenvalue problems, which randomized SVD handles natively. This is what uwot (R's UMAP) uses, and it has **different failure modes** than LOBPCG — making it an ideal escalation target.

```rust
fn spectral_embedding_rsvd(
    laplacian: &CsMat<f64>,
    n_components: usize,
    oversampling: usize,
    n_iter: usize,
) -> Array2<f64> {
    let n = laplacian.rows();

    // Form M = 2I - L (flip eigenvalue order)
    // Largest eigenvectors of M = smallest eigenvectors of L
    let m = two_i_minus_l(laplacian);

    // Randomized SVD of M for top-(n_components+1) singular vectors
    // Halko-Tropp algorithm: QR-stabilized subspace iteration
    let (u, _s) = randomized_svd(&m, n_components + 1, oversampling, n_iter);

    // Skip first column (trivial), take next n_components
    u.slice(s![.., 1..=n_components]).to_owned()
}
```

#### Phase 4: Solver Escalation Chain (No Random Fallback)

The design principle: **the eigenvectors exist, the math guarantees it, our job is to find them.** Each solver in the chain addresses failure modes the previous one cannot handle. The chain is exhaustive.

```rust
fn spectral_init(
    graph: &CsMatI<f32, u32, usize>,
    n_components: usize,
    seed: u64,
) -> Array2<f32> {
    let n = graph.rows();

    // Convert u32 indices to usize for eigensolver compatibility
    let graph_usize = convert_indices(graph);

    // Check connectivity
    let (n_comp, labels) = connected_components(&graph_usize);

    if n_comp > 1 {
        return multi_component_layout(&graph_usize, &labels, n_comp, n_components, seed);
    }

    // Build Laplacian in f64 for numerical stability
    let laplacian = build_normalized_laplacian(&graph_usize);

    // ── Solver Escalation Chain ──
    //
    // Level 0: Dense EVD (n < 2000) — exact, cannot fail
    // Level 1: LOBPCG — fast iterative, may not converge for tiny eigengaps
    // Level 2: LOBPCG with regularization — shifts eigenvalue 0 away from near-zero cluster
    // Level 3: Randomized SVD (2I-L trick) — entirely different algorithm family
    // Level 4: Dense EVD forced (any size) — O(n^3) but mathematically guaranteed
    //
    // Level 4 is the nuclear option. For n=50k it takes ~minutes and ~20GB RAM.
    // But it will produce correct eigenvectors. We never reach it in practice
    // because Level 3 (randomized SVD) handles everything LOBPCG cannot.

    let embedding: Array2<f64> = if n < 2000 {
        // Level 0: Small enough for exact dense EVD — guaranteed correct
        dense_spectral(&laplacian, n_components)
    } else {
        // Level 1: LOBPCG (fast path)
        lobpcg_spectral(&laplacian, n_components, seed)
            .or_else(|_| {
                // Level 2: LOBPCG with epsilon*I regularization
                // Separates eigenvalue 0 from the near-zero cluster
                let l_reg = add_diagonal(&laplacian, 1e-5);
                lobpcg_spectral(&l_reg, n_components, seed)
            })
            .or_else(|_| {
                // Level 3: Completely different algorithm — randomized SVD via 2I-L
                // Different numerical path, different failure modes
                rsvd_spectral(&laplacian, n_components, seed)
            })
            .or_else(|_| {
                // Level 4: Nuclear option — dense EVD
                // Expensive but mathematically cannot fail (spectral theorem)
                // Will OOM for n > ~50k, but if we're here, something is very wrong
                // with the graph and we need to know about it, not silently degrade
                dense_spectral_forced(&laplacian, n_components)
            })
            .expect("spectral initialization: all solvers exhausted — this is a bug")
    };

    let mut result = embedding.mapv(|x| x as f32);
    let mut rng = StdRng::seed_from_u64(seed);
    noisy_scale_coords(&mut result, &mut rng, 10.0, 0.0001);
    result
}
```

**Why this chain is exhaustive:**

| Level | Algorithm | Fails when... | Next level addresses it by... |
|-------|-----------|---------------|-------------------------------|
| 0 | Dense EVD | n > ~5k (OOM) | N/A — only used for small n |
| 1 | LOBPCG | Small eigengap, poor conditioning | Adding regularization to widen the gap |
| 2 | LOBPCG + reg | Still poor conditioning, or LOBPCG implementation bug | Switching to an entirely different algorithm family |
| 3 | Randomized SVD | Extremely pathological spectrum (near-zero singular values of 2I-L) | Falling through to exact computation |
| 4 | Dense EVD (forced) | OOM for very large n | **Cannot fail mathematically** — `.expect()` is a true assertion, not a lazy fallback |

In practice, Level 3 (randomized SVD) is the terminal solver for all realistic inputs. Level 4 exists as a correctness proof — if the graph is small enough to fit in dense form and all iterative solvers failed, the dense solver will produce the answer. If the graph is too large for dense and all iterative solvers failed, the `.expect()` is a genuine bug report, not a degraded experience.

### 8.3 Testing Strategy

1. **Numerical correctness against Python reference:** Generate a known graph (e.g., two well-separated clusters), compute spectral init in both Python (via `umap.spectral.spectral_layout`) and Rust, verify eigenvectors match up to sign and rotation
2. **End-to-end embedding quality:** Run full UMAP with spectral init vs random init on MNIST/Fashion-MNIST, compare trustworthiness and continuity metrics
3. **Solver escalation coverage:** Construct adversarial graphs that force each level of the escalation chain to activate — verify every level produces valid spectral coordinates (not random garbage)
4. **Edge cases that must all produce valid spectral output:**
   - Single point (degenerate but deterministic — place at origin)
   - Two points (trivial 1D embedding)
   - Fully connected graph (uniform eigenvectors — valid, just not informative)
   - Graph with many disconnected components (multi-component layout path)
   - Graph with eigenvalue degeneracies / tiny eigengap (tests escalation from Level 1 → Level 2)
   - Very large sparse graph (n > 100k) — performance test
   - Near-duplicate points creating near-singular Laplacian (tests regularization)
5. **Golden file tests:** Snapshot the spectral init output for 5-10 reference graphs and assert bitwise reproducibility given the same seed

### 8.4 Dependency Summary

**Minimum viable (LOBPCG path):**
```toml
[dependencies]
sprs = "0.11"
ndarray = "0.16"
ndarray-linalg = { version = "0.18", features = ["openblas-static"] }
rand = "0.8"
```

**Pure Rust (randomized SVD path):**
```toml
[dependencies]
sprs = "0.11"
ndarray = "0.16"
rand = "0.8"
# annembed or custom randomized SVD implementation
```

**Both paths + dense fallback:**
```toml
[dependencies]
sprs = "0.11"
ndarray = "0.16"
ndarray-linalg = { version = "0.18", features = ["openblas-static"] }
faer = "0.24"
rand = "0.8"
```

---

## 9. Component-by-Component Build & Verify Plan

The spectral initialization pipeline decomposes into **7 independent computational components**. Each one gets built in Rust, tested against Python's exact numerical output, and only wired together after all components independently match.

### 9.0 Methodology

For each component:
1. **Write a Python harness** that runs the component in isolation, dumps inputs and outputs to `.npz` files (NumPy's binary format — exact floating-point preservation)
2. **Write the Rust implementation** of that component
3. **Write a Rust test** that loads the `.npz` inputs, runs the Rust code, and asserts the outputs match within tolerance
4. **Tolerance strategy**: exact match for integer outputs (component labels, indices). For float outputs: `max(abs(rust - python)) < 1e-12` for f64 operations, `< 1e-6` for f32 operations. Eigenvectors get sign-normalized (flip so first nonzero element is positive) before comparison.
5. **Multiple test fixtures**: at least 3 graphs per component — small (n=50), medium (n=5000), adversarial (near-degenerate eigenvalues, disconnected, etc.)

The Python harness is not throwaway — it becomes the **ground truth oracle** for CI. Any future Rust change that breaks numerical parity with Python is a regression.

### 9.1 Component Map

```
Input: graph (sparse symmetric matrix, f32 values in (0,1])
  │
  ├─ Component A: Degree Vector
  │    Input:  sparse graph
  │    Output: degree[i] = sum of row i weights
  │    Verify: exact f64 match against np.asarray(graph.sum(axis=0))
  │
  ├─ Component B: Normalized Laplacian Construction
  │    Input:  sparse graph, degree vector
  │    Output: L = I - D^{-1/2} A D^{-1/2} (sparse, f64)
  │    Verify: element-wise match of all nonzero entries
  │
  ├─ Component C: Connected Components
  │    Input:  sparse graph
  │    Output: (n_components, labels[])
  │    Verify: exact integer match against scipy.sparse.csgraph.connected_components
  │
  ├─ Component D: Eigensolver (the big one)
  │    Input:  Laplacian L (sparse, f64), k = dim+1
  │    Output: eigenvalues[k], eigenvectors[n, k]
  │    Verify: eigenvalues match within tolerance;
  │            eigenvectors match up to sign per column;
  │            A·v = λ·v residual < tolerance for each pair
  │
  ├─ Component E: Eigenvector Selection
  │    Input:  eigenvalues[k], eigenvectors[n, k]
  │    Output: embedding[n, dim] (skip trivial, take dim smallest nonzero)
  │    Verify: exact index match for selected columns
  │
  ├─ Component F: Coordinate Scaling (noisy_scale_coords)
  │    Input:  embedding[n, dim], seed
  │    Output: scaled embedding (max_abs=10, f32, +noise)
  │    Verify: pre-noise scaling exact; noise distribution statistical test
  │
  └─ Component G: Multi-Component Layout (deferred)
       Input:  graph, component labels, original data
       Output: composite embedding[n, dim]
       Verify: per-component sub-embeddings match; meta-layout positions match
```

### 9.2 Component A: Degree Vector

**What it does:** Sum the weights of each row in the sparse graph.

**Python reference:**
```python
degrees = np.asarray(graph.sum(axis=0)).squeeze()  # shape (n,), f64
sqrt_deg = np.sqrt(degrees)
```

**Rust implementation scope:**
- Iterate CSR rows, sum nonzero values
- Cast f32 weights to f64 during accumulation (Python does this implicitly — `graph.sum()` promotes to f64)

**Verification:**
- Input: sparse graph as COO triplets (row, col, val) saved to npz
- Output: `degrees` array, `sqrt_deg` array
- Tolerance: exact f64 match (summation order may differ — use relative tolerance `1e-14`)

**Difficulty: Trivial.** This is a warm-up component to validate the test harness itself.

### 9.3 Component B: Normalized Laplacian Construction

**What it does:** Given graph A and degree vector, compute `L = I - D^{-1/2} A D^{-1/2}`.

**Python reference:**
```python
I = scipy.sparse.identity(n, dtype=np.float64)
D_inv_sqrt = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, n, n)
L = I - D_inv_sqrt * graph * D_inv_sqrt
```

**Rust implementation scope:**
- Compute `inv_sqrt_deg[i] = 1.0 / sqrt(degree[i])` (handle zero-degree nodes: set to 0.0)
- For each nonzero `A[i,j]`: compute `L[i,j] = -inv_sqrt_deg[i] * A[i,j] * inv_sqrt_deg[j]`
- Diagonal: `L[i,i] = 1.0` (identity minus the zero self-loop contribution)
- Output as sparse CSR in f64

**Verification:**
- Input: graph COO triplets + degree vector
- Output: Laplacian COO triplets (row, col, val)
- Tolerance: `1e-14` per element (same arithmetic, same precision)
- Additional check: L is symmetric (`L[i,j] == L[j,i]`), diagonal is all 1.0

**Difficulty: Easy.** Arithmetic is straightforward. Main subtlety is handling the identity diagonal correctly in the sparse format.

### 9.4 Component C: Connected Components

**What it does:** BFS/DFS on the sparse graph to find connected components.

**Python reference:**
```python
n_components, labels = scipy.sparse.csgraph.connected_components(graph)
```

**Rust implementation scope:**
- BFS traversal on CSR adjacency structure
- ~30 lines of code
- Output: component count and label array

**Verification:**
- Input: graph COO triplets
- Output: `n_components` (int), `labels` array (int)
- Tolerance: exact match. Note: label numbering may differ (component 0 vs component 1 assignment is arbitrary). Compare by grouping: same-component node pairs must match, not label values.

**Difficulty: Easy.** Standard graph algorithm. The only subtlety is the label-equivalence comparison.

### 9.5 Component D: Eigensolver

**What it does:** Given the Laplacian L, compute the k smallest eigenvalues and their eigenvectors.

This is the core component and the hardest one. It's also the one where we're filling a gap in the Rust ecosystem, so it deserves the most care.

**Python reference (ARPACK path):**
```python
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
    L, k, which="SM", ncv=num_lanczos_vectors,
    tol=1e-4, v0=np.ones(n), maxiter=n*5
)
```

**Python reference (LOBPCG path):**
```python
eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
    L, X_init, largest=False, tol=1e-4, maxiter=n*5
)
```

**Rust implementation scope — build in sub-components:**

```
Component D breaks down further:

D.1  SpMV: sparse matrix × dense vector product
     (the operator that the eigensolver calls repeatedly)
     Verify: y = L·x matches scipy L.dot(x) for random x vectors

D.2  Dense GEMV: V^T · r (project residual onto Lanczos/LOBPCG basis)
     Verify: against numpy matmul

D.3  Orthogonalization: QR or Cholesky-based
     Verify: Q^T Q = I within tolerance after orthogonalization

D.4  Rayleigh-Ritz projection (LOBPCG): small dense generalized eigensolve
     Verify: against scipy.linalg.eigh on same small matrix

D.5  Full eigensolver iteration: assemble D.1-D.4 into the loop
     Verify: eigenvalues match Python; eigenvectors satisfy A·v = λ·v
```

**Verification strategy for eigenvectors:**

Eigenvectors are unique only up to sign (and up to rotation within degenerate eigenspaces). Direct comparison requires:
1. **Sign normalization**: for each column, flip so the element with largest absolute value is positive
2. **Residual check**: for each (λ, v) pair, verify `||L·v - λ·v|| / ||v|| < tol`
3. **Eigenvalue match**: `|λ_rust - λ_python| < 1e-8` for each eigenvalue
4. **Subspace match** (for near-degenerate eigenvalues): if `|λ_i - λ_j| < 1e-6`, compare the subspace spanned by the corresponding eigenvectors rather than individual vectors. Check that `||V_rust · V_rust^T - V_python · V_python^T|| < tol` (projector comparison).

**Difficulty: Hard. This is the whole point.** But it decomposes into sub-components (D.1–D.5) that are each independently testable.

### 9.6 Component E: Eigenvector Selection

**What it does:** Sort eigenvalues, skip the trivial zero eigenvalue, take the next `dim` eigenvectors.

**Python reference:**
```python
order = np.argsort(eigenvalues)[1:k]
result = eigenvectors[:, order]
```

**Rust implementation scope:**
- Sort eigenvalues, get sorted indices
- Slice columns 1 through dim (skip index 0)

**Verification:**
- Input: eigenvalues array, eigenvectors matrix
- Output: selected column indices, resulting embedding matrix
- Tolerance: exact index match; matrix values follow from Component D

**Difficulty: Trivial.** Array indexing.

### 9.7 Component F: Coordinate Scaling

**What it does:** Scale embedding to max_abs=10, cast to f32, add tiny Gaussian noise.

**Python reference:**
```python
expansion = 10.0 / np.abs(coords).max()
coords = (coords * expansion).astype(np.float32)
coords += random_state.normal(scale=0.0001, size=coords.shape).astype(np.float32)
```

**Rust implementation scope:**
- Find global max absolute value
- Multiply all coordinates by `10.0 / max_abs`
- Cast f64 → f32
- Add N(0, 0.0001) noise per element

**Verification:**
- Pre-noise output: exact f32 match (deterministic)
- Post-noise output: match only if RNG produces identical sequence. Use same seed and verify the RNG streams are equivalent, OR verify pre-noise only and test noise distribution statistically (mean ≈ 0, std ≈ 0.0001)

**Difficulty: Easy.** The only subtlety is RNG compatibility between Python's NumPy and Rust's rand. For practical purposes, verify pre-noise scaling exactly and treat noise as a statistical property.

### 9.8 Component G: Multi-Component Layout (Deferred)

**Deferred until Components A–F pass for connected graphs.** This component only activates for disconnected graphs, which are less common and can be addressed after the main path works.

Sub-components when we get to it:
- G.1: Component centroid computation (dense, trivial)
- G.2: Inter-centroid distance matrix + Gaussian affinity (dense, trivial)
- G.3: Meta-embedding via spectral embedding of centroid graph (reuses Component D on a small dense matrix)
- G.4: Per-component spectral embedding (reuses Components A–F on subgraphs)
- G.5: Scale and translate assembly

### 9.9 Build Order & Dependencies

```
Phase 1 — Foundation (no dependencies between these, build in parallel):
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Component A  │  │ Component C  │  │ Component E  │
    │ Degree Vec   │  │ Conn. Comps  │  │ Eigvec Select│
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
Phase 2 — Depends on A:     │                 │
    ┌──────┴───────┐         │                 │
    │ Component B  │         │                 │
    │ Laplacian    │         │                 │
    └──────┬───────┘         │                 │
           │                 │                 │
Phase 3 — Depends on B (the hard part):       │
    ┌──────┴───────┐                           │
    │ Component D  │                           │
    │ Eigensolver  │                           │
    │  D.1 SpMV    │                           │
    │  D.2 GEMV    │                           │
    │  D.3 Ortho   │                           │
    │  D.4 Ritz    │                           │
    │  D.5 Loop    │                           │
    └──────┬───────┘                           │
           │                                   │
Phase 4 — Wire together (D output → E → F):   │
    ┌──────┴───────────────────────────────────┘
    │ Component F: Scaling
    └──────┬───────┘
           │
Phase 5 — Integration test:
    Full spectral_init(graph) → compare against
    Python umap.spectral.spectral_layout(data, graph, dim, random_state)

Phase 6 — Multi-component (deferred):
    Component G (uses A–F on subgraphs)
```

### 9.10 Python Harness Specification

A single Python script that generates all test fixtures:

```python
"""generate_spectral_fixtures.py

Run with: python generate_spectral_fixtures.py --output-dir fixtures/

Generates .npz files for each component's inputs and outputs,
using real UMAP graphs from small reference datasets.
"""

import numpy as np
import scipy.sparse
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from umap.spectral import _spectral_layout
from sklearn.datasets import make_blobs, make_moons

def generate_fixture(name, X, n_neighbors=15, n_components=2, seed=42):
    random_state = np.random.RandomState(seed)

    # Build the graph exactly as UMAP does
    knn_indices, knn_dists, _ = nearest_neighbors(X, n_neighbors, ...)
    graph, sigmas, rhos = fuzzy_simplicial_set(X, n_neighbors, random_state, ...)

    # Prune edges (as simplicial_set_embedding does)
    graph = graph.tocoo()
    graph.sum_duplicates()
    graph.data[graph.data < (graph.data.max() / 500.0)] = 0.0
    graph.eliminate_zeros()

    # === Component A: Degree Vector ===
    degrees = np.asarray(graph.sum(axis=0)).squeeze()
    sqrt_deg = np.sqrt(degrees)
    np.savez(f'{name}_A_degree.npz',
             graph_row=graph.row, graph_col=graph.col, graph_data=graph.data,
             graph_shape=graph.shape,
             degrees=degrees, sqrt_deg=sqrt_deg)

    # === Component B: Laplacian ===
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, *graph.shape)
    L = I - D * graph * D
    L_coo = L.tocoo()
    np.savez(f'{name}_B_laplacian.npz',
             L_row=L_coo.row, L_col=L_coo.col, L_data=L_coo.data,
             L_shape=L_coo.shape)

    # === Component C: Connected Components ===
    n_comp, labels = scipy.sparse.csgraph.connected_components(graph)
    np.savez(f'{name}_C_components.npz',
             n_components=n_comp, labels=labels)

    # === Component D: Eigensolver ===
    k = n_components + 1
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L, k, which="SM", tol=1e-4,
        v0=np.ones(L.shape[0]), maxiter=graph.shape[0]*5)
    np.savez(f'{name}_D_eigen.npz',
             eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    # === Component E: Selection ===
    order = np.argsort(eigenvalues)[1:k]
    embedding = eigenvectors[:, order]
    np.savez(f'{name}_E_selection.npz',
             order=order, embedding=embedding)

    # === Component F: Scaling ===
    expansion = 10.0 / np.abs(embedding).max()
    scaled = (embedding * expansion).astype(np.float32)
    noise = random_state.normal(scale=0.0001, size=scaled.shape).astype(np.float32)
    final = scaled + noise
    np.savez(f'{name}_F_scaling.npz',
             pre_noise=scaled, noise=noise, final=final)

    # === Full pipeline reference ===
    full_result = _spectral_layout(None, graph, n_components, random_state)
    np.savez(f'{name}_full.npz', embedding=full_result)

# Generate fixtures for multiple graph types
for name, X in [
    ('blobs_50', make_blobs(50, centers=3, random_state=42)[0]),
    ('blobs_500', make_blobs(500, centers=5, random_state=42)[0]),
    ('moons_200', make_moons(200, noise=0.1, random_state=42)[0]),
    ('blobs_5000', make_blobs(5000, centers=10, random_state=42)[0]),
]:
    generate_fixture(name, X)
```

This script is the **single source of truth**. Every Rust test loads from these fixtures.

---

## 10. Fortran Performance Parity & SIMD Optimization

### 10.1 What ARPACK Actually Does (And Why Fortran Doesn't Matter)

ARPACK is not a monolithic Fortran performance miracle. It's a thin algorithmic orchestration layer that delegates all heavy computation to BLAS/LAPACK. The entire real symmetric path (`dsaupd` → `dsaup2` → `dsaitr` → `dseupd`) breaks down as:

**ARPACK's own code does:** reverse-communication state machine, convergence checks (`dsconv` — pure scalar loop), shift selection (`dsgets` — sorting), QR bulge-chasing via Givens rotations (`dsapps`). None of this is performance-critical.

**BLAS does all the work:**

| Routine | Operation | Where called | Bottleneck type |
|---------|-----------|-------------|-----------------|
| `dgemv` | Matrix-vector multiply (V^T·r and V·w) | `dsaitr` (2-4x per step) | **Memory-bound** |
| `ddot` | Inner product | `dsaitr`, `dsaup2` | Memory-bound |
| `dcopy` | Vector copy | Everywhere (~6x per step) | Memory-bound |
| `dscal` | Vector scale | Everywhere (~4x per step) | Memory-bound |
| `dnrm2` | Euclidean norm | `dsaitr` | Memory-bound |
| `dsteqr` | Tridiagonal eigendecomposition | `dseupd` (post-processing) | Compute-bound (small matrix) |
| `dgemv` | Final Ritz vector computation | `dseupd` | Memory-bound |

**The user-supplied matrix-vector product (SpMV) dominates total runtime.** ARPACK's internal BLAS calls are secondary.

### 10.2 Why Rust Can Match or Beat This

The performance analysis reveals that nothing about ARPACK is language-dependent:

**Memory-bound operations (dgemv, ddot, SpMV):** Performance is determined by DRAM bandwidth, not instruction throughput. Rust, C, and Fortran all hit the same memory bandwidth ceiling. Arithmetic intensity for CSR SpMV is ~0.25-0.5 FLOP/byte — modern CPUs with ~100 FLOP/byte peak are bandwidth-limited at 100x+. The language literally cannot matter here.

**Compute-bound operations (small dense eigensolves, Cholesky):** The `faer` crate, written in pure Rust with hand-written AVX-512 microkernels, matches or beats OpenBLAS on modern hardware. This is not theoretical — `faer`'s GEMM substrate ([github.com/sarah-quinones/gemm](https://github.com/sarah-quinones/gemm)) uses BLIS-style cache blocking with hand-tuned microkernels for each ISA.

**Fortran's one real advantage — non-aliasing guarantees:** Fortran arrays cannot alias unless explicitly declared, allowing the compiler to generate simpler code without barrier instructions. Rust provides the exact same guarantee via the borrow checker. This is a language-level match that C requires `restrict` hacks for.

### 10.3 SIMD Optimization Opportunities

#### Where SIMD Matters

For the Lanczos/LOBPCG algorithms, SIMD primarily helps in:

1. **SpMV (the hot path):** The sparse matrix-vector product dominates runtime. SIMD helps with dense vector accumulation within each row but random-access column indexing defeats prefetchers. Format matters more than SIMD: ELLPACK or SELL-C-sigma formats expose more vectorization than CSR.

2. **Dense GEMV in orthogonalization:** When projecting against the Lanczos basis (V^T · r), this is a dense matrix-vector product. `faer`'s gemv uses 8x accumulator unrolling to hide memory latency — this is where AVX-512's 8-wide f64 operations help.

3. **Small dense eigenproblems in Rayleigh-Ritz:** LOBPCG's inner loop solves a (3k x 3k) generalized eigenvalue problem each iteration (k = number of desired eigenvectors, typically 2-3, so 6-9 x 6-9). This is tiny — SIMD helps marginally at best.

4. **Vector norms, dot products, axpy:** All trivially vectorizable. Auto-vectorization typically handles these, but explicit SIMD avoids loop-peeling overhead.

#### Rust SIMD Stack (Available Now, Stable)

**`std::arch::x86_64`** — stable since Rust 1.27:

```rust
use std::arch::x86_64::*;

// AVX-512: 8 f64 values per register
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = _mm512_setzero_pd();
    for chunk in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let va = _mm512_loadu_pd(chunk.0.as_ptr());
        let vb = _mm512_loadu_pd(chunk.1.as_ptr());
        acc = _mm512_fmadd_pd(va, vb, acc);  // fused multiply-add
    }
    _mm512_reduce_add_pd(acc)
}
```

Runtime feature detection:
```rust
if std::arch::is_x86_feature_detected!("avx512f") {
    unsafe { dot_avx512(a, b) }
} else if std::arch::is_x86_feature_detected!("avx2") {
    unsafe { dot_avx2(a, b) }
} else {
    dot_scalar(a, b)
}
```

**`pulp` crate** ([docs.rs/pulp](https://docs.rs/pulp/latest/pulp/)) — safe SIMD abstraction by faer's author:

```rust
use pulp::Arch;

let arch = Arch::new();  // runtime CPU feature detection
arch.dispatch(|| {
    // Code here gets compiled for all ISA levels,
    // dispatched to the best available at runtime
});
```

Supports: SSE, AVX, FMA, AVX-512 (x86); NEON (ARM); SIMD128 (WASM); scalar fallback.

**`wide` crate** ([github.com/Lokathor/wide](https://github.com/Lokathor/wide)) — portable wide types on stable Rust:

Provides `f64x4` (AVX2), `f64x8` (AVX-512 if available, otherwise emulated via 2x f64x4). Works on stable Rust today. Uses `safe_arch` internally for the actual intrinsics.

#### How faer Does It (The Gold Standard for Rust Numerics)

faer's GEMM kernel structure ([github.com/sarah-quinones/gemm](https://github.com/sarah-quinones/gemm)):

```
gemm-f64/src/microkernel.rs — Hand-written SIMD microkernels per ISA:
    ├── AVX2+FMA:    N=4 doubles (_mm256_fmadd_pd)
    ├── AVX-512F:    N=8 doubles (_mm512_fmadd_pd)
    ├── ARM NEON:    N=2 doubles
    ├── ARM AMX:     N=8 (tile matrix ops)
    ├── WASM SIMD128: N=2
    └── Scalar:      N=1

gemm-common/src/gemm.rs — 3-level cache blocking (NC/MC/KC):
    ├── k ≤ 2:  dispatches to GEVV (vector-vector)
    ├── m = 1:  dispatches to GEMV (matrix-vector)
    └── general: BLIS-style blocked GEMM with MR×NR tiles
```

faer's GEMV (`gemm-common/src/gemv.rs`) uses 8x accumulator unrolling to saturate memory bandwidth — this is exactly the technique that matters for Lanczos orthogonalization.

The 13.2% C code in the faer-rs repo is FFI binding glue, not kernels. **All hot SIMD microkernels are pure Rust using `std::arch` intrinsics.**

### 10.4 Concrete SIMD Optimization Plan for the Eigensolver

| Component | Default (no SIMD) | SIMD opportunity | Expected speedup |
|-----------|-------------------|-------------------|-----------------|
| SpMV (CSR) | Scalar row accumulation | AVX-512 accumulation within rows; prefetch next row's column indices | 1.5-2x (still memory-bound) |
| SpMV (SELL-C-sigma) | N/A — requires format change | Fully vectorized: C consecutive rows processed in lock-step | 2-4x over CSR |
| Dense GEMV (V^T · r) | Use faer's gemv | Already hand-optimized with SIMD | ~1x (just use faer) |
| ddot / dnrm2 | Auto-vectorized | AVX-512 FMA accumulator chain | 1.2x (already auto-vec'd) |
| Rayleigh-Ritz (3k×3k eigh) | faer self_adjoint_eigen | Already SIMD | ~1x |
| QR in orthogonalization | faer/LAPACK | Already SIMD | ~1x |
| Givens rotations (if implementing IRLM) | Scalar | Batched Givens with SIMD | 1.5-2x |

**Bottom line:** For a Lanczos/LOBPCG eigensolver, the SIMD wins come from:
1. **SpMV** — the dominant cost. Consider SELL-C-sigma format for vectorized multi-row processing.
2. **Orthogonalization GEMV** — delegate to faer's hand-tuned GEMV.
3. Everything else is either already handled by faer or too small to matter.

The realistic overall speedup from explicit SIMD over a naive Rust implementation is **2-3x**. Over a well-vectorized baseline (using faer for dense ops), it's **1.2-1.5x** — and that gain comes almost entirely from the SpMV format choice.

### 10.5 Can We Actually Beat Python UMAP's Performance?

Yes, and here's why:

| Factor | Python (SciPy eigsh) | Rust (our implementation) |
|--------|---------------------|---------------------------|
| SpMV | SciPy CSR → calls MKL/OpenBLAS dcsrmv | `sprs` CSR with explicit SIMD accumulation, or SELL-C-sigma |
| ARPACK overhead | Python ↔ Fortran marshalling per iteration (reverse communication crosses FFI boundary) | Zero-cost — everything is native Rust |
| Dense ops | NumPy/LAPACK via Python FFI | faer — native Rust, no FFI overhead |
| Memory allocation | Python/NumPy allocator | Rust stack/arena allocation, no GC pressure |
| Parallelism | ARPACK is single-threaded | LOBPCG block operations naturally parallelize via Rayon; SpMV rows are embarrassingly parallel |

**The reverse-communication overhead in ARPACK is significant:** each Lanczos step crosses the Python → C → Fortran → C → Python boundary twice (once for OP*v, once for B*v). For a 50k-node graph requiring ~200 iterations, that's ~400 FFI round-trips with Python object marshalling. In Rust, this is a function pointer call.

**LOBPCG in Rust has a structural advantage:** its block operations (A*[X|R|P], QR factorization, block eigh) naturally decompose into parallelizable chunks. SciPy's LOBPCG is single-threaded Python with LAPACK calls. A Rust implementation using Rayon for the SpMV and faer for the dense blocks could see **3-5x wall-clock improvement** on multi-core systems.

### 10.6 Source Code Links

| Resource | URL |
|----------|-----|
| **ARPACK-NG** (BSD-3-Clause) | [github.com/opencollab/arpack-ng](https://github.com/opencollab/arpack-ng) |
| ARPACK dsaitr.f (Lanczos iteration) | [github.com/opencollab/arpack-ng/blob/master/SRC/dsaitr.f](https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaitr.f) |
| ARPACK dsaupd.f (top-level driver) | [github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f](https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f) |
| ARPACK dseupd.f (eigenpair extraction) | [github.com/opencollab/arpack-ng/blob/master/SRC/dseupd.f](https://github.com/opencollab/arpack-ng/blob/master/SRC/dseupd.f) |
| ARPACK dsapps.f (implicit shifts) | [github.com/opencollab/arpack-ng/blob/master/SRC/dsapps.f](https://github.com/opencollab/arpack-ng/blob/master/SRC/dsapps.f) |
| **SciPy eigsh wrapper** | [github.com/scipy/scipy/.../arpack.py](https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_eigen/arpack/arpack.py) |
| **SciPy LOBPCG** (~550 lines of algorithm) | [github.com/scipy/scipy/.../lobpcg.py](https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_eigen/lobpcg/lobpcg.py) |
| **faer crate** (Rust linear algebra) | [github.com/sarah-quinones/faer-rs](https://github.com/sarah-quinones/faer-rs) |
| faer GEMM microkernels (SIMD) | [github.com/sarah-quinones/gemm/.../microkernel.rs](https://github.com/sarah-quinones/gemm/blob/main/gemm-f64/src/microkernel.rs) |
| faer GEMV (memory-bound optimization) | [github.com/sarah-quinones/gemm/.../gemv.rs](https://github.com/sarah-quinones/gemm/blob/main/gemm-common/src/gemv.rs) |
| faer benchmarks | [faer.veganb.tw/benchmarks/](https://faer.veganb.tw/benchmarks/) |
| **pulp crate** (safe SIMD abstraction) | [docs.rs/pulp](https://docs.rs/pulp/latest/pulp/) |
| **wide crate** (portable SIMD on stable) | [github.com/Lokathor/wide](https://github.com/Lokathor/wide) |
| `std::arch` docs | [doc.rust-lang.org/std/arch](https://doc.rust-lang.org/std/arch/index.html) |
| `std::simd` tracking (nightly) | [github.com/rust-lang/rust/issues/86656](https://github.com/rust-lang/rust/issues/86656) |
| portable-simd repo | [github.com/rust-lang/portable-simd](https://github.com/rust-lang/portable-simd) |
| Cache-blocking matmul analysis (Algorithmica) | [en.algorithmica.org/hpc/algorithms/matmul/](https://en.algorithmica.org/hpc/algorithms/matmul/) |
| Rust auto-vectorization (Nick Wilcox) | [nickwilcox.com/blog/autovec/](https://www.nickwilcox.com/blog/autovec/) |
| ARPACK-NG license (BSD-3) | [github.com/opencollab/arpack-ng/blob/master/COPYING](https://github.com/opencollab/arpack-ng/blob/master/COPYING) |

---

## 11. Mathematical Reference

### 11.1 Key Formulas

| Formula | Description |
|---------|-------------|
| `L = D - W` | Unnormalized graph Laplacian |
| `L_sym = I - D^{-1/2} W D^{-1/2}` | Symmetric normalized Laplacian (**UMAP uses this**) |
| `L_rw = I - D^{-1} W` | Random-walk Laplacian |
| `x^T L x = (1/2) sum_{i,j} W_ij (x_i - x_j)^2` | Laplacian quadratic form (why it works) |
| `v_ij = exp(-max(0, d_ij - rho_i) / sigma_i)` | UMAP membership strength |
| `A_sym = A + A^T - A * A^T` | Fuzzy union (probabilistic OR) |
| `L_UMAP(Y) ~ 2a * tr(Y^T L(V) Y) + L_repel(Y)` | UMAP loss decomposition (Theorem 3.1) |

### 11.2 Eigenvalue Properties

| Property | Value |
|----------|-------|
| Eigenvalue range of L_sym | [0, 2] |
| Multiplicity of eigenvalue 0 | = number of connected components |
| First eigenvector of L_sym | Proportional to `sqrt(degree_i)` |
| Fiedler value (lambda_2) | Algebraic connectivity of the graph |
| Cheeger inequality | `lambda_2 / 2 <= h(G) <= sqrt(2 * lambda_2)` |

### 11.3 Algorithm Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build Laplacian | O(nnz) | O(nnz) |
| LOBPCG (k eigenvectors, m iterations) | O(k * m * nnz) | O(k * n) |
| Randomized SVD (k components, p oversampling, q power iterations) | O((k+p) * q * nnz) | O((k+p) * n) |
| Dense EVD | O(n^3) | O(n^2) |
| Connected components (BFS) | O(n + nnz) | O(n) |

For typical UMAP: `n=50k`, `k=15 neighbors`, `nnz ~ 750k`, requesting 2–3 eigenvectors → LOBPCG converges in seconds.

---

## 12. Risk Assessment & Mitigations

### 12.1 Numerical Accuracy

| Risk | Mitigation |
|------|------------|
| LOBPCG convergence failure for near-degenerate eigenvalues | Solver escalation chain: regularization (Level 2) → randomized SVD (Level 3) → dense EVD (Level 4) |
| Loss of orthogonality in Lanczos-type iterations | Use LOBPCG (has built-in reorthogonalization) rather than bare Lanczos |
| f32 precision loss in Laplacian construction | Build Laplacian in f64, downsample eigenvectors to f32 for init |
| Sign ambiguity of eigenvectors | Eigenvectors are unique up to sign; doesn't matter for UMAP (coordinates are relative) |
| All iterative solvers fail on pathological input | Dense EVD (Level 4) is mathematically guaranteed by the spectral theorem; `.expect()` on chain exhaustion is a true assertion |

### 12.2 Index Type Mismatch

`umap-rs` uses `CsMatI<f32, u32, usize>` (u32 column indices). Most eigensolvers expect `usize` indices. A conversion step is needed when constructing the Laplacian. This is O(nnz) and negligible.

### 12.3 Performance at Scale

| Scale | Strategy | Expected Time |
|-------|----------|---------------|
| n < 2k | Dense faer EVD | < 100ms |
| 2k < n < 100k | LOBPCG | 1-10s |
| 100k < n < 2M | LOBPCG with preconditioning | 10-60s |
| n > 2M | Randomized SVD (2I-L trick) | 30-120s |

### 12.4 Missing Pieces in the Ecosystem

| Gap | Status | Workaround |
|-----|--------|------------|
| No Rust `eigsh` with shift-invert | Fundamental gap | Use LOBPCG or randomized SVD |
| `sprs` has no graph algorithms | Expected | Implement BFS for connected components (~30 lines) |
| `ndarray-linalg` LOBPCG needs LAPACK | Structural dependency | Accept LAPACK dep, or use randomized SVD path |
| No preconditioning support in `ndarray-linalg` LOBPCG | Performance impact at scale | Implement diagonal preconditioner `M = diag(L)^{-1}`; or escalate to randomized SVD which doesn't need preconditioning |

---

## 13. Phase 1 Completion & Phase 2 Status

### 13.1 Phase 1: Reference Data Validation (COMPLETE)

All Phase 1 issues are closed. The Python fixture infrastructure is complete and validated:

| Issue | Title | Status |
|-------|-------|--------|
| #1 | Set up micromamba Python environment | DONE |
| #2 | Create fixture infrastructure and dataset generators | DONE |
| #3 | Implement KNN pipeline steps | DONE |
| #4 | Implement graph construction steps | DONE |
| #5 | Implement Laplacian construction steps | DONE |
| #6 | Implement eigendecomposition chain | DONE |
| #7 | Add pipeline references, verification, documentation | DONE |
| #16 | Add guaranteed-connected datasets | DONE |
| #17 | Add exact-distance fixture path | DONE |
| #18 | Fix verification false positives | DONE |
| #19 | Add solver metadata to eigensolver fixtures | DONE |
| #20 | Fix test code connectivity assumptions | DONE |

Nine datasets in the registry: `blobs_50`, `blobs_500`, `moons_200`, `blobs_5000`, `circles_300`, `near_dupes_100`, `disconnected_200`, `blobs_connected_200`, `blobs_connected_2000`. Each produces 14 fixture files. Small datasets also have exact-distance KNN fixtures.

### 13.2 Phase 2: Rust Implementation Status

All Phase 2 implementation tasks have been completed as GitHub issues #27-#41. The Rust implementation includes:

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| A: Degree vector | `src/laplacian.rs` | Implemented | f32→f64 upcast, zero-degree guard |
| B: Normalized Laplacian | `src/laplacian.rs` | Implemented | f64 throughout, COO→CSR construction |
| C: Connected components | `src/components.rs` | Implemented | BFS, CSR debug_assert, partition-based comparison |
| D: Dense EVD | `src/solvers/dense.rs` | Implemented | faer self_adjoint_eigen, Level 0/4 |
| D: LOBPCG | `src/solvers/lobpcg.rs` | Implemented | linfa-linalg, Level 1/2, regularization |
| D: Randomized SVD | `src/solvers/rsvd.rs` | Implemented | 2I-L trick, pure Rust, Halko-Tropp |
| D: Escalation chain | `src/solvers/mod.rs` | Implemented | 5-level chain with residual gating on all levels |
| E: Eigenvector selection | `src/selection.rs` | Implemented | Sort + skip trivial |
| F: Coordinate scaling | `src/scaling.rs` | Implemented | max_abs=10, f64→f32 cast, noise |
| G: Multi-component | `src/multi_component.rs` | Implemented | Full 492-line implementation, not a stub |
| LinearOperator + SpMV | `src/operator.rs` | Implemented | Trait + standalone spmv_csr function |
| Pipeline integration | `src/lib.rs` | Implemented | Full spectral_init() public API |
| NPZ test loading | `tests/common/mod.rs` | Implemented | Dense + sparse loaders, residual helpers |

### 13.3 Test Isolation (Verified Correct)

Every component test loads its inputs from Python fixtures, NOT from the Rust output of the previous step:

| Test | Input Source | Reference Source | Correctly Isolated? |
|------|-------------|-----------------|---------------------|
| Component A | `step5a_pruned.npz` (Python) | `comp_a_degrees.npz` (Python) | Yes |
| Component B | `step5a_pruned.npz` + `comp_a_degrees.npz` (both Python) | `comp_b_laplacian.npz` (Python) | Yes |
| Component D (all solvers) | `comp_b_laplacian.npz` (Python) | `comp_d_eigensolver.npz` (Python) | Yes |
| Component E | `comp_d_eigensolver.npz` (Python) | `comp_e_selection.npz` (Python) | Yes |
| Component F | `comp_e_selection.npz` (Python) | `comp_f_scaling.npz` (Python) | Yes |

No test chains Rust step N-1 output into step N. Error accumulation between steps is not possible.

---

## 14. Phase 2 Implementation Audit & Remediation Record

### 14.1 Initial Audit (2026-03-21)

A 10-agent audit examined every source file, test file, tolerance, and architectural decision. Eight issues were identified (A–H). All have been resolved via GitHub issues #57–#71.

### 14.2 Audit Issue Resolution

| Issue | Finding | Resolution | PR |
|-------|---------|------------|-----|
| A | Missing fixtures for `blobs_connected_200` / `blobs_connected_2000` | Fixtures regenerated; all 9 datasets have complete `.npz` files including exact-distance paths | #57 (PR #80) |
| B | Missing exact-distance KNN fixtures | All 9 datasets now have `step5a_pruned_exact.npz` | #57 (PR #80) |
| C | No dedicated Component C test | `test_comp_c_components.rs` created, tests all 9 datasets with partition grouping comparison | #59 (PR #82) |
| D | Component E test broken (1 dataset, missing fixtures) | Expanded to 6 datasets; all pass | #64 (PR #83) |
| E | LOBPCG missing trivial eigenvector injection | `lobpcg_solve()` now accepts `sqrt_deg` and injects `d^{1/2}/||d^{1/2}||` as column 0 | #60 (PR #79) |
| F | Level 2 eigenvalues shifted by `REGULARIZATION_EPS` | `lobpcg_solve()` now subtracts `REGULARIZATION_EPS` from eigenvalues when `regularize=true` | #62 (PR #75) |
| G | No residual quality gate on Levels 0, 1, 2 | All levels now gated: Dense EVD 1e-6, LOBPCG 1e-3, rSVD 1e-2; threshold ordering invariant tested | #61 (PR #74) |
| H | `spmv_csr` unused dead code | Wired into `CsrOperator::apply` for k=1 (single-vector fast path); k>1 still uses `csr_mulacc_dense_rowmaj` | #63 (PR #73) |

### 14.3 Tolerance Tightening (RESOLVED — #66, PR #72)

| Location | Old | New | Rationale |
|----------|-----|-----|-----------|
| `test_comp_a_degrees.rs` | `1e-5` | `3e-7` (relative) | Observed max relative error ≈ 2.2e-7 (~1.84 f32_epsilon); 37% headroom |
| `test_e2e_validation.rs` | `0.05` | `0.005` | 10x tighter; still 50x above expected noise floor |
| `scaling.rs` (2 approx assertions) | `1e-5` | `1e-6` | f32 ULP at 10.0 ≈ 9.5e-7; 1e-6 is just above single-ULP bound |
| `solvers/mod.rs` non-negativity | Mixed | `-1e-10` (dense EVD), `-1e-6` (LOBPCG) | Two-level design: tight for exact, loose for iterative |

### 14.4 Test Coverage After Remediation

| Component | Datasets Tested | Method |
|-----------|----------------|--------|
| A (degrees) | **9/9** | All datasets via parametrized tests |
| B (Laplacian) | **9/9** | All datasets via parametrized tests |
| C (components) | **9/9** | All datasets with partition grouping |
| D (dense EVD) | **5/5** connected | All connected datasets |
| D (LOBPCG) | **5/5** connected | All connected datasets |
| D (rSVD) | **6** | 5 connected + blobs_500 |
| E (selection) | **6** | 5 connected + disconnected_200 |
| F (scaling) | **9/9** | All datasets (pre-noise exact, noise statistical) |

### 14.5 Additional Test Infrastructure (Remediation Issues #67–#70)

**Adversarial synthetic graph test suite** (#67, PR #77): 10 graph topologies (barbell, path, star, epsilon-bridge, complete bipartite, ring, weighted exponential, single edge, complete, lollipop) with 16 tests covering no-panic/no-NaN validation, community separation verification, coordinate stability checks, and solver escalation level verification. Requires `--features testing`.

**cargo-nextest with CI profile** (#68, PR #81): `.config/nextest.toml` with JUnit XML output at `target/nextest/ci/junit.xml`. GitHub Actions workflow at `.github/workflows/test.yml` runs on push/PR with test result annotation via `mikepenz/action-junit-report@v6`.

**Numerical accuracy report generator** (#69, PR #86): `test_accuracy_report.rs` generates `target/accuracy-report.md` and `target/accuracy-report.json` covering all 9 datasets with per-eigenpair residuals, eigenvalue error, subspace alignment, pre-noise scaling accuracy, and a tolerance margin analysis table.

**Criterion benchmark baselines** (#70, PR #78): 9 benchmarks in `benches/spectral_bench.rs` covering SpMV (n=200/2000), dense EVD, LOBPCG, rSVD, Laplacian build, BFS components, and full pipeline (n=200/2000). Baseline results documented in `benches/README.md`.

### 14.6 Current Test Suite Summary

| Metric | Count |
|--------|-------|
| Total tests (with `--features testing`) | ~130 |
| Passing | 130+ |
| Failing | 0 |
| Ignored (fixture-gated, pass when un-ignored) | 29 |
| Compiler warnings | 0 |
| Adversarial graph types | 10 |
| Datasets with full fixture coverage | 9/9 |

### 14.7 Remaining Minor Items

1. `scaling.rs:223` — one `#[ignore]`-gated test still uses `1e-5` tolerance (should be `1e-6` for consistency)
2. `manifest.json` — 7 of 9 datasets don't list `step_files_exact` in metadata (files exist on disk; metadata gap only)
3. `test_accuracy_report.rs` chains Rust D→E→F outputs (by design — it's an accuracy report measuring end-to-end quality, not a component isolation test)

### 14.8 Stale References in Earlier Sections

Sections 7.2 and 8.4 reference `ndarray-linalg` as the LOBPCG provider. The actual implementation uses `linfa-linalg` (pure Rust, no LAPACK dependency). Section 8.2 mentions `annembed` for rSVD; the actual implementation is a custom Halko-Tropp implementation using `faer` for dense QR and eigendecomposition. These sections remain as historical design documentation; the actual implementation details are in Section 13.2 and the source code.

---

## 15. Numerical Accuracy Improvement Campaign (2026-03-22)

### 15.1 Background

A comprehensive numerical error propagation investigation was conducted across the full 6-step pipeline. The investigation used 18 sub-agents to analyze every source file, trace Python's exact computation (down to the C++ level in scipy), apply perturbation theory (Weyl's theorem, Davis-Kahan), and research LOBPCG convergence improvements. Full reports are in `temp/numerical-error-propagation-report.md` and `temp/numerical-accuracy-improvement-plan.md`.

### 15.2 Root Cause Discovery

The primary error source was identified as **degree summation order mismatch**: Rust summed CSR rows in f64; Python's `scipy.sparse.csr_matrix.sum(axis=0)` performs f32 scatter-add column accumulation via C++ `csc_matvec`. The ~2.7e-7 relative error propagated through `inv_sqrt_deg` into the Laplacian (~3.6e-8 absolute), then into eigenvalue comparisons (~2.5e-8, breaching the 1e-8 tolerance).

A second critical finding: the accuracy report tested disconnected graphs through a code path (full-Laplacian eigensolver) that never executes in production, producing misleading metrics (blobs_5000 subspace Gram determinant = 0.313).

### 15.3 Implementation Summary

Eight implementation tickets (A1, A2, B1, B2, B4, C1, C2, C3) and one verification ticket (E1) were completed:

| Ticket | PR | Summary |
|--------|-----|---------|
| A1 | #102 | Match Python's f32 column-sum degree computation via `ComputeMode::PythonCompat` |
| A2 | #100 | Fix fixture generator — remove explicit f64 upcast that real UMAP doesn't do |
| B1 | #99 | Chebyshev Filtered Subspace Iteration (ChFSI) preconditioning for LOBPCG |
| B2 | #103 | Shift-and-invert LOBPCG via faer sparse Cholesky (new Level 2 in escalation) |
| B4 | #104 | Tighten LOBPCG convergence tolerance from 1e-4 to 1e-6 after preconditioning |
| C1 | #101 | Accuracy report tests production paths for disconnected graphs (`embed_disconnected`) |
| C2 | #106 | Per-component residual metrics for disconnected datasets |
| C3 | #98 | Split comp_b tolerance into isolated and chained measurements |
| E1 | #105 | Verify Dense EVD eigenvalue breach resolved by A1; tighten tolerances |

Post-campaign fix: split comp_f pre_noise tolerance into isolated (scaling arithmetic, exact) and chained (includes eigenvector solver differences, 2.6x margin).

### 15.4 Accuracy Report: Before vs After

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| comp_a degrees | 2.71e-7, margin 1.11x | **0.0, margin inf** | Eliminated |
| comp_b Laplacian (chained) | 3.61e-8, margin 2.8e-7x | **0.0, margin inf** | Eliminated |
| comp_d eigenvalues, Dense EVD | 2.53e-8, margin **0.40x (BREACHED)** | 1.66e-10, margin **6x** | Fixed |
| comp_d eigenvalues, LOBPCG | 1.20e-6, margin 0.83x | 6.59e-10, margin **1518x** | Fixed |
| comp_d residuals, LOBPCG | 1.02e-4, margin **0.98x (CRITICAL)** | 9.10e-6, margin **11x** | Fixed |
| E2E residuals | 8.63e-5, margin 57.9x | 3.16e-5, margin **158x** | Improved |
| Disconnected (blobs_5000) | subspace det **0.313** | per-comp residual **2e-13** | Fixed (production path) |
| comp_f pre_noise (isolated) | 0.0 | **0.0, margin inf** | Now measured separately |
| comp_f pre_noise (chained) | 1.90e-3, margin 0.00x | 1.90e-3, margin **2.6x** | Tolerance corrected |

All tolerance margins are now positive. No breaches remain.

### 15.5 Solver Escalation Chain (Post-Campaign)

```
Level 0: Dense EVD (n < 2000)               — quality threshold 1e-6
Level 1: LOBPCG with ChFSI preconditioning   — quality threshold 1e-5
Level 2: Shift-and-invert LOBPCG (faer LLT)  — quality threshold 1e-6
Level 3: Randomized SVD (2I-L trick)          — quality threshold 1e-2
Level 4: Forced Dense EVD                     — unconditional
```

### 15.6 Current Test Suite Summary

| Metric | Count |
|--------|-------|
| Total tests (with `--features testing`) | ~190 |
| Passing | 190 |
| Failing | 0 |
| Ignored (fixture-gated) | 29 |
| Adversarial graph types | 10 |
| Datasets with full fixture coverage | 9/9 |
| Tolerance margin rows (all positive) | 12 |

### 15.7 Evaluation Ticket Status

| Ticket | Issue | Status | Outcome |
|--------|-------|--------|---------|
| B3 | #107 | **Closed (NO-GO)** | scirs2-sparse fails on correctness: TRL produces negative eigenvalues for PSD matrices, IRAM never converges, LOBPCG diverges on multi-cluster data. See `temp/scirs2-sparse-evaluation-decision.md` |
| B5 | #108 | **Closed (no action)** | linfa-linalg uses soft locking (confirmed). Pre-2023 scipy gaps mitigated by existing defense-in-depth. See `temp/evaluation-linfa-linalg-soft-locking.md` |
| D1 | — | Deferred | Investigate umap-rs fork for f64 weights (post-A1, may be unnecessary) |

### 15.8 Post-Campaign Improvements

| Issue | PR | Summary |
|-------|-----|---------|
| #111 | #114 | Final exact Rayleigh-Ritz refinement for LOBPCG — post-processing step that recomputes `X^T A X` eigenproblem and rotates eigenvectors. Improved subspace orthogonality to exactly 1.0. |
| #112 | — | Unconvergence detection for LOBPCG activemask — in progress |

### 15.9 Visual UMAP Evaluation Pipeline (Next Phase)

The numerical accuracy report validates that Rust's eigenvectors match Python's within tight tolerances. The next validation layer is **visual**: feed Rust's spectral initialization into Python UMAP's SGD optimizer and compare the resulting embeddings side-by-side.

**Method**: Python UMAP's `init` parameter accepts any `(n, n_components)` array, bypassing its own spectral init and going straight to SGD. Three-way comparison:

1. **Python spectral init → Python SGD** (baseline)
2. **Rust spectral init → Python SGD** (what we're testing)
3. **Random init → Python SGD** (control — should look visibly worse)

This isolates exactly one variable (the spectral initialization) while keeping graph construction and SGD optimization constant.

**Success criteria**:
- Procrustes distance (Rust vs Python): < 0.05
- Pairwise distance correlation: > 0.99
- Visual: Python and Rust init embeddings should be indistinguishable

**Datasets**: Synthetic (blobs, circles, swiss roll, moons, high-dim blobs), real-world (MNIST, Fashion-MNIST, pendigits), with a single-cell data placeholder.

Full specification: `temp/umap-visual-evaluation-prompt.md`

---

## 16. Sources

### Primary Source Code
- [umap/spectral.py — lmcinnes/umap](https://github.com/lmcinnes/umap/blob/master/umap/spectral.py)
- [umap/umap_.py — lmcinnes/umap](https://github.com/lmcinnes/umap/blob/master/umap/umap_.py)
- [wilsonzlin/umap-rs](https://github.com/wilsonzlin/umap-rs)

### Research Papers
- "UMAP Is Spectral Clustering on the Fuzzy Nearest-Neighbor Graph" (2025) — [arXiv:2602.11662](https://arxiv.org/html/2602.11662)
- "Generalizable Spectral Embedding with Application to UMAP" (2025) — [arXiv:2501.11305](https://arxiv.org/html/2501.11305v2)
- "Initialization is critical for preserving global data structure in both t-SNE and UMAP" — Nature Biotechnology (2021)
- Belkin & Niyogi, "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation" (2003)
- Von Luxburg, "A Tutorial on Spectral Clustering" — [CMU](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf)
- "Accelerating UMAP via Spectral Coarsening" (2024) — [arXiv:2411.12331](https://arxiv.org/html/2411.12331v1)

### Rust Crates
- [sprs](https://crates.io/crates/sprs) — Sparse matrix library
- [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) — LOBPCG implementation
- [faer](https://docs.rs/faer/latest/faer/) — Dense linear algebra
- [annembed](https://lib.rs/crates/annembed) — UMAP-like with randomized SVD
- [linfa](https://github.com/rust-ml/linfa) — ML framework with spectral methods

### Related Implementations
- [uwot (R UMAP) — init.R](https://rdrr.io/cran/uwot/src/R/init.R) — Uses 2I-L trick with randomized SVD
- [UMAP official docs — How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)

---

*Report generated 2026-03-10 via multi-agent research investigation. Updated 2026-03-21 with Phase 1 completion, Phase 2 implementation status, 10-agent audit findings, and full remediation record.*
