# **SpectralInit: Development Guidelines**

Mandatory instructions for AI-assisted development in this repository.

## **1. Core Project Goal**

A standalone Rust crate that implements spectral initialization for UMAP embeddings. It computes Laplacian eigenvectors of the fuzzy k-NN graph to provide globally-aware starting coordinates for SGD optimization — the key factor that makes Python UMAP produce superior embeddings compared to random or PCA initialization.

Designed to integrate with `umap-rs` (wilsonzlin) via its public `LearnedManifold::graph()` accessor, but usable by any Rust UMAP implementation that provides a sparse adjacency matrix.

**Core algorithm:** Build the symmetric normalized Laplacian `L = I - D^{-1/2} W D^{-1/2}`, compute the smallest non-trivial eigenvectors via a six-level solver escalation chain (dense EVD → LOBPCG → shift-invert LOBPCG → LOBPCG+regularization → randomized SVD → dense EVD for any size within the dense memory budget), scale and add noise. When no level produces an acceptable result within the budget, return an error. No random fallback, ever.

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

  * **Run tests**: `cargo nextest run --workspace --all-features` from the project root. `cargo test` alone runs only the root package's default-feature tests.
  * **Always run tests at end of task**
  * **Add tests for new features**
  * **Follow existing test patterns** — avoid test code redundancy
  * **Numerical verification**: Test against Python reference outputs saved as `.npz` fixtures. Eigenvectors match up to sign; use residual checks and subspace comparison for near-degenerate eigenvalues.

## **5. Architecture**

The repository is a Cargo workspace. The root package is the published `spectral-init` crate; `trustworthiness/` and `eval/` are unpublished members.

```
src/                      # The published crate
├── lib.rs                # Public API: spectral_init(); hidden `__internal` re-exports
├── config.rs             # SpectralInitConfig, ComputeMode
├── error.rs              # SpectralError
├── validate.rs           # Boundary validation of config, graph and data
├── components.rs         # Connected components (BFS on sparse graph)
├── laplacian.rs          # Degrees and symmetric normalized Laplacian construction
├── operator.rs           # LinearOperator; scalar and AVX2 sparse matrix-vector product
├── selection.rs          # Selection of the non-trivial eigenvectors
├── solvers/              # Eigensolver implementations
│   ├── mod.rs            # Solver escalation chain, SolverParams, dense memory budget
│   ├── quality.rs        # Residual thresholds and checks
│   ├── dense.rs          # Dense EVD via faer (Levels 0 and 5)
│   ├── lobpcg.rs         # LOBPCG iterative solver (Levels 1 and 3)
│   ├── sinv.rs           # Shift-and-invert LOBPCG (Level 2)
│   ├── rsvd.rs           # Randomized SVD via 2I-L trick (Level 4)
│   └── nd16_bridge.rs    # The only user of linfa-linalg and ndarray 0.16
├── multi_component.rs    # Disconnected graph handling
└── scaling.rs            # Sign normalization, coordinate scaling and noise

examples/                 # Runnable examples, shipped with the crate
docs/                     # User guide, umap-rs integration guide, KaTeX header

tests/
├── public_api/           # Tests of the public API (default features)
├── internals/            # Tests of internals (feature `__internal`)
├── common/               # Fixture loaders and diagnostics, shared with eval/
├── determinism_threads.rs  # Thread-configuration test (its own binary)
├── fixtures/             # Python-generated .npz reference data
└── visual_eval/          # Python visual-evaluation tooling

trustworthiness/          # Workspace member: embedding-quality metric and CLI
eval/                     # Workspace member: research harness (benches, sweeps, exporters)
research/                 # Experiment artifacts and reports

temp/                     # Temporary/working files (gitignored)
```

## **6. Key Dependencies**

  * `sprs` — Sparse matrices (CSR/CSC/COO), matches `umap-rs` internal format; public dependency, re-exported
  * `ndarray` — Dense array operations; public dependency, re-exported
  * `faer` — Dense linear algebra (eigendecomposition, QR) and sparse Cholesky
  * `linfa-linalg` — pure-Rust LOBPCG eigensolver (no LAPACK dependency), reached only through `src/solvers/nd16_bridge.rs`
  * `rand`, `rand_chacha`, `rand_distr` — ChaCha12 RNG for reproducible initialization
  * `thiserror`, `log` — error type and logging facade

## **7. Reference**

  * Implementation report: see `research/reports/umap-spectral-initialization-rust-implementation-report.md`
  * Python UMAP spectral source: `umap/spectral.py` in `lmcinnes/umap`
  * Target integration: `umap-rs` by wilsonzlin (`LearnedManifold::graph()` returns `CsMatI<f32, u32, usize>`)
