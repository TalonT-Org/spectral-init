# Revision Guidance: Trustworthiness Step-Timing Validation on MERFISH Data

**Verdict: REVISE** — 3 critical findings require design-level changes before execution.

## Required Revisions (Critical Findings)

### 1. Asymmetric Data Provenance (baseline_fairness)

**Gap:** The Gaussian baseline is generated in-session while MERFISH data is pre-existing git-committed fixture data. These asymmetric provenance paths mean the Gaussian dataset is subject to in-session environmental effects (memory state, generation overhead) that MERFISH is not exposed to.

**Risk:** Any latency or cache effects during Gaussian data generation may carry over into its profiling measurements, introducing a systematic artifact that is absent from the MERFISH condition. This undermines the fairness of the Gaussian-vs-MERFISH comparison.

### 2. Confounded Comparison Dimensions (baseline_fairness)

**Gap:** The two primary compared systems differ simultaneously on input dimensionality (d_x=10 vs d_x=50) and data geometry (synthetic Gaussian vs real biological). The plan acknowledges the dimensionality confound but treats the comparison as a single-axis contrast.

**Risk:** Any observed difference in step-timing fractions between MERFISH and Gaussian cannot be attributed to either dimensionality or geometry alone. Conclusions about "whether X-dominance transfers to real data" conflate the effect of higher dimensionality with the effect of non-Gaussian neighborhood structure. The plan's primary question ("does the step-timing breakdown hold on real data?") is unanswerable from a design that co-varies both factors.

### 3. Page-Cache Spillover Between Sequential Runs (unit_interference)

**Gap:** The three dataset runs (Gaussian → MERFISH 10K → MERFISH 50K) share OS page-cache state across their boundaries with no cache flushing between runs. The first dataset runs cold while subsequent datasets run warm.

**Risk:** The warm-cache state from earlier datasets alters the I/O latency profile and potentially the memory access patterns of later datasets. This is a directional, asymmetric interference embedded in all MERFISH measurements. Combined with the fixed execution order (never randomized), position effects are fully confounded with dataset identity.

## Recommended Revisions (Warning Findings)

### Estimand Structure

**Gap:** The plan conflates two distinct contrasts (one-sample threshold test at 50% and two-sample MERFISH-vs-Gaussian comparison) without formally separating them. The success criteria have an ambiguous corridor near 50% where the conclusive-negative and inconclusive bands overlap.

**Risk:** Readers cannot determine which estimand drives the verdict when results fall in the overlap zone. The post-hoc nature of the 50% threshold compounds this ambiguity.

### Variance and Replication Protocol

**Gap:** Execution order is fixed and never randomized. No inter-run seed control is documented for the profiler binary. R=5 with no extension mechanism limits statistical power.

**Risk:** Systematic position effects (cache warming, thermal accumulation, Rayon scheduling variance) are confounded with dataset identity across all replicates. The hard R=5 cap means high-variance conditions (MERFISH, which prior experiments show produces 3x wider CIs) will likely yield uninformative confidence intervals.

### Measurement Proxy Alignment

**Gap:** The primary metric (x_space_pct) measures compute-share in thread-summed nanoseconds, but the optimization decision it informs targets wall-clock time improvement. The plan acknowledges SIMD throughput asymmetry but does not address the parallelism-scaling confound (if steps differ in thread utilization, thread-ns share diverges from wall-clock share).

**Risk:** x_space_pct may systematically overstate or understate X-space's actual wall-clock dominance. A positive verdict (x_space_pct >= 50%) does not guarantee that wall-clock share is also >= 50%, making the investment decision based on a biased proxy.

### Reproducibility Gaps

**Gap:** Hardware is described only as "16-core system" with no CPU model or ISA specification. No environment lockfile exists for Python dependencies.

**Risk:** An independent reproducer cannot confirm whether the AVX2 dispatch path was active, cannot match the compilation target, and cannot guarantee bitwise-identical Gaussian data generation.

## Red-Team Findings (Require Author Decision)

### RT-1: Proxy Bias Direction Favors Rejecting H1

The AtomicU64 thread-summed ns proxy may compress x_dist's apparent share at d_x=50 (where AVX2 retires more FLOPs per ns), making it structurally easier to observe x_space_pct < 50%. The author must decide whether this bias direction is acceptable given the research question's intent.

### RT-2: Three-Axis Confound Makes Causal Attribution Impossible

Dimensionality, geometry, and sparsity pattern all co-vary between MERFISH and Gaussian. Prior research found unexplained 2x wall-time differences between the datasets that cannot be attributed to d_x alone. The author must decide whether the comparison is meaningful without controlling for these axes.

### RT-3: Historical Baseline Invalidated by Code Change

The 56.2% reference was measured pre-PR#242 (AtomicU64 counter reset fix). Divergence from this baseline may reflect the code change rather than dataset effects. The author must decide whether the historical comparison should be retained, rebaselined, or clearly caveated.

### RT-4: Instrumentation Overhead Varies by Condition

Probe overhead from AtomicU64 fetch_add operations is not constant across dataset configurations. The perturbation is proportionally larger for fast steps and smaller for slow steps, systematically distorting the fraction measurements differently across conditions.

### RT-5: Sample Size Inadequate for MERFISH Variance

MERFISH produces ~3x wider CIs than Gaussian at the same n. With R=5 (df=4), the MERFISH 50K condition will almost certainly produce an "inconclusive" result regardless of the true step distribution. The author must decide whether to accept this limitation or increase the iteration count.
