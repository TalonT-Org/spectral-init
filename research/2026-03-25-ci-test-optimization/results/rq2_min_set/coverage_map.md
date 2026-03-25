# Solver-Level Coverage Map (RQ2)

## Solver Escalation Chain

| Level | Solver | Condition |
|-------|--------|-----------|
| 0 | Dense EVD | n < 2000 (DENSE_N_THRESHOLD) |
| 1 | LOBPCG (no regularization) | n >= 2000, quality < LOBPCG_QUALITY_THRESHOLD (1e-5) |
| 2 | Shift-and-invert LOBPCG | Level 1 quality >= threshold |
| 3 | LOBPCG + ε·I regularization | Level 2 quality >= threshold or Cholesky fails |
| 4 | Randomized SVD (2I-L trick) | Levels 1-3 all fail |
| 5 | Forced dense EVD | All prior levels fail (nuclear option) |

## Test Coverage by Level

### Level 0 — Dense EVD (n < DENSE_N_THRESHOLD = 2000)

**Exercised by (through escalation chain):**
- `test_adversarial_graphs::test_level_0_dense_evd_for_small_n` — directly asserts level == 0
- All adversarial topology tests with small n (barbell, path, star, lollipop, etc.) — route through Level 0

**Exercised by (direct unit tests):**
- `test_comp_d_dense_evd::comp_d_dense_evd_blobs_connected_200`
- `test_comp_d_dense_evd::comp_d_dense_evd_moons_200`
- `test_comp_d_dense_evd::comp_d_dense_evd_near_dupes_100`
- `test_comp_d_dense_evd::comp_d_dense_evd_circles_300`
- `test_comp_d_dense_evd::comp_d_dense_evd_blobs_connected_2000` (boundary: n == 2000 uses dense EVD; n >= 2000 skips it)

### Level 1 — LOBPCG (no regularization)

**Exercised by (through escalation chain):**
- `test_adversarial_graphs::test_level_1_lobpcg_for_large_well_conditioned_n` — asserts level == 1

**Exercised by (direct solver tests):**
- `test_comp_f_lobpcg` tests (regularize=false path, n >= 2000 graphs)

### Level 2 — Shift-and-invert LOBPCG

**Exercised by (direct solver tests):**
- `test_comp_g_sinv` tests — exercise `lobpcg_sinv_solve` directly

**NOT exercised through escalation chain** — no existing test constructs an input where
Level 1 LOBPCG fails quality threshold (>= 1e-5) but sinv succeeds. This is a coverage gap:
sinv effectiveness is validated in isolation but never as a fallback in the full chain.

### Level 3 — LOBPCG + ε·I regularization

**Exercised by (through escalation chain):**
- `test_adversarial_graphs::test_level_2_regularized_lobpcg_produces_valid_result` — calls
  `lobpcg_solve(regularize=true)` directly (test name uses pre-sinv level numbering).

**Exercised by (direct solver tests):**
- `test_comp_f_lobpcg` tests (regularize=true path)

**Reachability through escalation chain:** Level 3 requires both Level 1 AND Level 2 to fail;
no existing test drives the full escalation to Level 3. Level 3 is tested in isolation only.

### Level 4 — Randomized SVD

**Exercised by (direct solver tests):**
- `test_comp_d_rsvd::comp_d_rsvd_blobs_connected_200`
- `test_comp_d_rsvd::comp_d_rsvd_blobs_connected_2000`
- `test_comp_d_rsvd::comp_d_rsvd_blobs_500`
- `test_comp_d_rsvd::comp_d_rsvd_moons_200`
- `test_comp_d_rsvd::comp_d_rsvd_circles_300`
- `test_comp_d_rsvd::comp_d_rsvd_near_dupes_100`
- `test_adversarial_graphs::test_level_3_rsvd_valid_on_large_path` — tests `rsvd_solve` directly

**NOT exercised through escalation chain** — rSVD is not reached via natural inputs in practice.
Well-conditioned graphs succeed at Level 0 or Level 1; adversarial graphs that stress Levels 2–3
do not currently exist in the fixture set.

### Level 5 — Forced Dense EVD (nuclear option)

**No existing test exercises Level 5.** It is the fallback of last resort; reaching it would
require Levels 0–4 all to fail, which requires a pathological matrix beyond any current fixture.
Level 5 is intentionally excluded from the minimum covering set (untestable via natural inputs).

## Coverage Gaps

| Gap | Impact |
|-----|--------|
| Level 2 unreachable through chain | Sinv effectiveness only validated in isolation |
| Level 3 unreachable through chain | Reg-LOBPCG never tested as fallback |
| Level 5 unreachable | Nuclear path untested (acceptable: spectral theorem guarantee) |

## Minimum Covering Set (for RQ2)

Minimum set covering all reachable levels (0, 1, 2-direct, 3-direct, 4):

| Level | Chosen Test | Rationale |
|-------|-------------|-----------|
| 0 | `test_level_0_dense_evd_for_small_n` | Asserts level == 0 explicitly |
| 1 | `test_level_1_lobpcg_for_large_well_conditioned_n` | Asserts level == 1 explicitly |
| 2 | `test_comp_g_sinv` (any one) | Only tests exercising Level 2 (sinv) |
| 3 | `test_comp_f_lobpcg` (regularize=true path) | Only tests exercising Level 3 |
| 4 | `comp_d_rsvd_blobs_connected_200` | Smallest rSVD fixture test |

### Nextest filter expression

```
test(test_level_0_dense_evd_for_small_n) \
  + test(test_level_1_lobpcg_for_large_well_conditioned_n) \
  + binary(test_comp_g_sinv) \
  + binary(test_comp_f_lobpcg) \
  + test(comp_d_rsvd_blobs_connected_200)
```
