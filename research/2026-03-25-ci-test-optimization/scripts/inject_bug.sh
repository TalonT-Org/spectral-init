#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BUG_ID="${1:?Usage: inject_bug.sh <B1..B8> <apply|revert>}"
ACTION="${2:?Usage: inject_bug.sh <B1..B8> <apply|revert>}"

apply_patch() {
    local target="$1"
    local diff_content="$2"
    local sentinel="$3"

    if grep -qF "$sentinel" "$target" 2>/dev/null; then
        echo "[inject_bug] ${BUG_ID}: already applied, skipping"
        return 0
    fi
    printf '%s\n' "$diff_content" | patch -p1 --forward --reject-file=/dev/null --no-backup-if-mismatch
    rm -f "${target}.orig"
    echo "[inject_bug] ${BUG_ID}: applied to $target"
}

revert_patch() {
    local target="$1"
    local diff_content="$2"
    local sentinel="$3"

    if ! grep -qF "$sentinel" "$target" 2>/dev/null; then
        echo "[inject_bug] ${BUG_ID}: already reverted, skipping"
        return 0
    fi
    printf '%s\n' "$diff_content" | patch -p1 -R --forward --reject-file=/dev/null --no-backup-if-mismatch
    rm -f "${target}.orig"
    echo "[inject_bug] ${BUG_ID}: reverted $target"
}

case "$BUG_ID" in
  B1)
    # src/solvers/dense.rs — take eigenvalues in reverse order (largest instead of smallest)
    TARGET="src/solvers/dense.rs"
    SENTINEL="// BUG_B1"
    DIFF=$(cat <<'PATCH'
--- a/src/solvers/dense.rs
+++ b/src/solvers/dense.rs
@@ -30,14 +30,14 @@
     let s = evd.S();
     let u = evd.U();

     let eigenvalues = Array1::from_iter(
-        s.column_vector().iter().take(k).copied()
+        s.column_vector().iter().rev().take(k).copied() // BUG_B1
     );

     let mut eigenvectors = Array2::<f64>::zeros((n, k));
     for j in 0..k {
-        let col = u.col(j);
+        let col = u.col(n - 1 - j); // BUG_B1
         for (i, &val) in col.iter().enumerate() {
             eigenvectors[[i, j]] = val;
         }
PATCH
)
    ;;

  B2)
    # src/solvers/mod.rs — set DENSE_N_THRESHOLD = 0 (all graphs bypass dense EVD)
    TARGET="src/solvers/mod.rs"
    SENTINEL="// BUG_B2"
    DIFF=$(cat <<'PATCH'
--- a/src/solvers/mod.rs
+++ b/src/solvers/mod.rs
@@ -23,7 +23,7 @@
 pub type EigenResult = (Array1<f64>, Array2<f64>);

 /// Graphs with n < DENSE_N_THRESHOLD use Level 0 (dense EVD) directly.
-const DENSE_N_THRESHOLD: usize = 2000;
+const DENSE_N_THRESHOLD: usize = 0; // BUG_B2

 /// Maximum acceptable max-residual from rSVD before falling to Level 4.
 /// rSVD with 2 power iterations typically achieves 1e-4 to 1e-6 on well-conditioned
PATCH
)
    ;;

  B3)
    # src/scaling.rs — negate the scaled coordinates
    TARGET="src/scaling.rs"
    SENTINEL="// BUG_B3"
    DIFF=$(cat <<'PATCH'
--- a/src/scaling.rs
+++ b/src/scaling.rs
@@ -17,7 +17,7 @@
         )));
     }
     let expansion = max_coord / max_abs;
-    Ok(coords.mapv(|x| (x * expansion) as f32))
+    Ok(coords.mapv(|x| (-(x * expansion)) as f32)) // BUG_B3
 }

 /// Internal helper: scale then add per-element Gaussian noise.
PATCH
)
    ;;

  B4)
    # src/multi_component.rs — skip embedding for all components except index 0
    TARGET="src/multi_component.rs"
    SENTINEL="// BUG_B4"
    DIFF=$(cat <<'PATCH'
--- a/src/multi_component.rs
+++ b/src/multi_component.rs
@@ -56,6 +56,7 @@
     for (comp_idx, members) in component_members.iter().enumerate() {
         let meta_pos = meta_embedding.row(comp_idx);
+        if comp_idx >= 1 { continue; } // BUG_B4
         let comp_coords = embed_single_component(
             graph,
             members,
PATCH
)
    ;;

  B5)
    # src/laplacian.rs — skip D^{-1/2} normalization in Laplacian construction
    TARGET="src/laplacian.rs"
    SENTINEL="// BUG_B5"
    DIFF=$(cat <<'PATCH'
--- a/src/laplacian.rs
+++ b/src/laplacian.rs
@@ -69,11 +69,11 @@
     for (row_idx, row_vec) in graph.outer_iterator().enumerate() {
         for (col_idx, &val) in row_vec.iter() {
             if row_idx != col_idx {
                 tri.add_triplet(
                     row_idx,
                     col_idx,
-                    -inv_sqrt_deg[row_idx] * (val as f64) * inv_sqrt_deg[col_idx],
+                    -(val as f64), // BUG_B5
                 );
             }
         }
     }
PATCH
)
    ;;

  B6)
    # src/solvers/lobpcg.rs — limit warm-restart loop to 1 iteration (no restarts)
    TARGET="src/solvers/lobpcg.rs"
    SENTINEL="// BUG_B6"
    DIFF=$(cat <<'PATCH'
--- a/src/solvers/lobpcg.rs
+++ b/src/solvers/lobpcg.rs
@@ -302,1 +302,1 @@
-    for restart in 0..=MAX_WARM_RESTARTS {
+    for restart in 0..=0 { // BUG_B6
PATCH
)
    ;;

  B7)
    # src/solvers/rsvd.rs — return trivial eigenvector for all non-trivial slots
    TARGET="src/solvers/rsvd.rs"
    SENTINEL="// BUG_B7"
    DIFF=$(cat <<'PATCH'
--- a/src/solvers/rsvd.rs
+++ b/src/solvers/rsvd.rs
@@ -153,9 +153,9 @@
     // Trivial pair (λ_L ≈ 0)
     eig_vals_vec.push(2.0 - m_eigenvals[trivial_m_idx]);
     eig_col_indices.push(trivial_m_idx);

-    // Non-trivial pairs in ascending L eigenvalue order
-    for i in 0..n_components {
-        let idx = actual_n.saturating_sub(2 + i);
+    // Non-trivial pairs — BUG_B7: always use trivial index
+    for _i in 0..n_components { // BUG_B7
+        let idx = trivial_m_idx;
         eig_vals_vec.push(2.0 - m_eigenvals[idx]);
         eig_col_indices.push(idx);
     }
PATCH
)
    ;;

  B8)
    # src/solvers/sinv.rs — return zeros without performing Cholesky solve
    TARGET="src/solvers/sinv.rs"
    SENTINEL="// BUG_B8"
    DIFF=$(cat <<'PATCH'
--- a/src/solvers/sinv.rs
+++ b/src/solvers/sinv.rs
@@ -159,5 +159,6 @@
-    let result_opt = match result {
-        Ok(r) => Some(extract(r)),
-        Err((_, Some(r))) if r.rnorm.iter().all(|&norm| norm < 1e-6) => Some(extract(r)),
-        _ => None,
-    };
+    let result_opt: Option<EigenResult> = Some((Array1::zeros(k), Array2::zeros((n, k)))); // BUG_B8
+    let _ = match result {
+        Ok(r) => Some(extract(r)),
+        Err((_, Some(r))) if r.rnorm.iter().all(|&norm| norm < 1e-6) => Some(extract(r)),
+        _ => None,
+    };
PATCH
)
    ;;

  *)
    echo "Unknown bug_id: $BUG_ID (expected B1..B8)" >&2
    exit 1
    ;;
esac

case "$ACTION" in
  apply)
    apply_patch "$TARGET" "$DIFF" "$SENTINEL"
    cargo check --features testing --quiet
    ;;
  revert)
    revert_patch "$TARGET" "$DIFF" "$SENTINEL"
    cargo check --features testing --quiet
    ;;
  *)
    echo "Unknown action: $ACTION (expected apply|revert)" >&2
    exit 1
    ;;
esac
