# **SpectralInit: Development Guidelines**

Mandatory instructions for AI-assisted development in this repository.

## **1. Core Project Goal**

A standalone Rust crate that implements spectral initialization for UMAP embeddings. It computes Laplacian eigenvectors of the fuzzy k-NN graph to provide globally-aware starting coordinates for SGD optimization — the key factor that makes Python UMAP produce superior embeddings compared to random or PCA initialization.

Designed to integrate with `umap-rs` (wilsonzlin) via its public `graph()` accessor, but usable by any Rust UMAP implementation that provides a sparse adjacency matrix.

**Core algorithm:** Build the symmetric normalized Laplacian `L = I - D^{-1/2} W D^{-1/2}`, compute the smallest non-trivial eigenvectors via a solver escalation chain (dense EVD → LOBPCG → LOBPCG+regularization → randomized SVD → forced dense EVD), scale and add noise. No random fallback, ever.

## **2. General Principles**

  * **Follow the Task Description**: Your primary source of truth is the issue, ticket, or work package description provided for your assignment.
  * **Adhere to Task Scope**: Do not work on unassigned features, unrelated refactoring, or bug fixes outside the current assignment.
  * **Implement Faithfully**: Produce functionally correct implementations. Do not add unrequested features.
  * **Adhere to Project Standards**: Write clean, maintainable Rust following established conventions and idiomatic patterns.

## **3. Critical Rules - DO NOT VIOLATE**

### **3.1. Code and Implementation**

  * **Do Not Oversimplify**: Implement logic with its required complexity. No shortcuts that compromise correctness.
  * **Respect the Existing Architecture**: Build upon established project structure and design patterns. Understand existing code before modifying.
  * **Address the Root Cause**: Debug to find and fix root causes. No hardcoded workarounds.
  * **No Backward Compatibility Hacks**: No comments about dead code. Remove dead code entirely.
  * **Avoid Redundancy**: Do not duplicate logic or utilities.
  * **Use Current Package Versions**: Web search for current stable versions when adding dependencies.
  * **Numerical Precision**: Build the Laplacian in f64 for numerical stability; only downcast to f32 for the final output. Validate eigenvector quality via residual checks (`||L·v - λ·v|| / ||v||`).
  * **ComputeMode Parity**: Any optimization or algorithm change that could cause divergence from Python UMAP's behavior must be gated behind `ComputeMode::RustNative`, preserving the `PythonCompat` path as the reference-matching implementation.

### **3.2. File System**

  * **Temporary Files:** All temp files must go in the project's `temp/` directory.
  * **Do Not Add Root Files**: Never create new root files unless explicitly required.
  * **Never commit unless told to do so**

### **3.3. CLAUDE.md Modifications**

  * **Correcting existing content is permitted**: If you discover that CLAUDE.md contains inaccurate information, you may correct it without being asked.
  * **Adding new content requires explicit instruction**: Never add new sections or information to CLAUDE.md unless the user has explicitly asked you to update or extend it.

## **4. Testing Guidelines**

  * **Run tests**: `cargo test` from the project root.
  * **Always run tests at end of task**
  * **Add tests for new features**
  * **Follow existing test patterns** — avoid test code redundancy
  * **Numerical verification**: Test against Python reference outputs saved as `.npz` fixtures. Eigenvectors match up to sign; use residual checks and subspace comparison for near-degenerate eigenvalues.

## **5. Architecture**

```
src/
├── lib.rs               # Public API: spectral_init()
├── components.rs         # Connected components (BFS on sparse graph)
├── laplacian.rs          # Symmetric normalized Laplacian construction
├── solvers/              # Eigensolver implementations
│   ├── mod.rs            # Solver escalation chain
│   ├── dense.rs          # Dense EVD via faer (small n)
│   ├── lobpcg.rs         # LOBPCG iterative solver
│   └── rsvd.rs           # Randomized SVD via 2I-L trick
├── multi_component.rs    # Disconnected graph handling
└── scaling.rs            # Coordinate scaling and noise

tests/
├── fixtures/             # Python-generated .npz reference data
└── integration/          # End-to-end tests against Python UMAP output

temp/                     # Temporary/working files (gitignored)
```

## **6. Key Dependencies**

  * `sprs` — Sparse matrices (CSR/CSC/COO), matches `umap-rs` internal format
  * `ndarray` — Dense array operations
  * `faer` — Dense linear algebra (eigendecomposition, QR)
  * `ndarray-linalg` — LOBPCG eigensolver
  * `rand` — RNG for reproducible initialization

## **7. Reference**

  * Implementation report: see `docs/umap-spectral-initialization-rust-implementation-report.md`
  * Python UMAP spectral source: `umap/spectral.py` in `lmcinnes/umap`
  * Target integration: `umap-rs` by wilsonzlin (`CsMatI<f32, u32, usize>` graph format)
