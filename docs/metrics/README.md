# Metrics Reference: MERFISH UMAP Benchmark

This directory documents all 17 metrics used in the MERFISH UMAP benchmark pipeline. All definitions, formulas, and thresholds derive from [`docs/merfish-umap-benchmark-plan.md`](../merfish-umap-benchmark-plan.md) Sections 8 and 10. Each metric is fully specified in its category file below.

**Category files:**
- [Category A: Structure Preservation](structure-preservation.md) — 5 metrics
- [Category B: Spatial Correlation](spatial-correlation.md) — 5 metrics
- [Category C: Cluster Preservation](cluster-preservation.md) — 3 metrics
- [Category D: Global Structure](global-structure.md) — 4 metrics

---

## Full Metric Catalog

| # | Metric | Category | Local/Global | Needs Labels | Needs Spatial | Complexity | Library |
|---|--------|----------|-------------|--------------|---------------|------------|---------|
| 1 | Trustworthiness | A | Local | No | No | O(n²) | sklearn |
| 2 | Continuity | A | Local | No | No | O(n²) | Custom |
| 3 | Silhouette score | A | Cluster | Yes | No | O(n²) | sklearn |
| 4 | Procrustes disparity | A | Global | No | No | O(n) | scipy |
| 5 | Pairwise distance correlation | A | Global | No | No | O(n²) | scipy |
| 6 | Spatial Neighbor Agreement | B | Local-spatial | No | Yes | O(n·k) | Custom |
| 7 | Spatial Distance Correlation | B | Global-spatial | No | Yes | O(n²) | scipy |
| 8 | Moran's I (embedding dims) | B | Global-spatial | No | Yes | O(n·k) | Custom/esda |
| 9 | CHAOS score | B | Cluster-spatial | Yes | Yes | O(n·k) | Custom |
| 10 | PAS score | B | Local-spatial | Yes | Yes | O(n·k) | Custom |
| 11 | ARI | C | Cluster | Yes | No | O(n) | sklearn |
| 12 | NMI | C | Cluster | Yes | No | O(n) | sklearn |
| 13 | Cell-type purity | C | Cluster | Yes | No | O(n) | Custom |
| 14 | Random triplet accuracy | D | Global | No | No | O(triplets) | Custom |
| 15 | Shepard diagram correlation | D | Global | No | No | O(n²) | Custom |
| 16 | Centroid distance correlation | D | Global | Yes | No | O(c²) | Custom |
| 17 | kNN preservation rate | D | Local | No | No | O(n·k) | Custom |

---
