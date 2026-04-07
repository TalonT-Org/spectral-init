# Causal Assumptions Diagram: y_heap Bottleneck Optimization

**Lens:** Causal Assumptions (Causal-Structural)
**Question:** What causal assumptions support this design?
**Date:** 2026-04-06
**Scope:** Causal isolation of y_heap performance bottleneck — allocation cost vs. data structure overhead vs. SIMD arithmetic, with correctness and thermal confounds mapped.

## Causal Variables

| Variable | Type | Measured? | Controlled? |
|----------|------|-----------|-------------|
| Algorithm variant | Treatment | Yes | Yes (exhaustive enumeration) |
| n (dataset size) | Treatment (scale) | Yes | Yes (fixed set: 1K/5K/10K) |
| Thermal state | Confounder | No | Partial (60s gap instrument) |
| Cache state | Confounder | No | Partial (Criterion warm-up) |
| Rayon thread scheduling noise | Confounder | No | Partial (fixed nproc) |
| d_y=2 / AVX2 availability | Confounder | Yes | Yes (fixed d_y=2, AVX2 confirmed) |
| Allocation cost path | Mediator (null) | Indirect | — |
| Data structure overhead | Mediator | Indirect | — |
| AVX2 SIMD throughput | Mediator (conditional) | Indirect | — |
| y_heap step fraction | Mediator | Yes (profiler) | — |
| Total trustworthiness() wall time | Outcome | Yes (Criterion) | — |
| \|ΔT\| correctness | Outcome | Yes (unit tests) | — |
| Criterion Flat sampling (10 samples) | Selection | Yes | Partial (W8 guard) |

## Causal DAG

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 60, 'curve': 'basis'}}}%%
flowchart TB
    %% CLASS DEFINITIONS %%
    classDef cli fill:#1a237e,stroke:#7986cb,stroke-width:2px,color:#fff;
    classDef stateNode fill:#004d40,stroke:#4db6ac,stroke-width:2px,color:#fff;
    classDef handler fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff;
    classDef phase fill:#6a1b9a,stroke:#ba68c8,stroke-width:2px,color:#fff;
    classDef newComponent fill:#2e7d32,stroke:#81c784,stroke-width:2px,color:#fff;
    classDef output fill:#00695c,stroke:#4db6ac,stroke-width:2px,color:#fff;
    classDef detector fill:#b71c1c,stroke:#ef5350,stroke-width:2px,color:#fff;
    classDef gap fill:#ff6f00,stroke:#ffa726,stroke-width:2px,color:#000;
    classDef terminal fill:#1a237e,stroke:#7986cb,stroke-width:2px,color:#fff;

    subgraph Instruments ["INSTRUMENTS (Blocking Confounders)"]
        direction LR
        GAP60["ThermalGap 60s<br/>━━━━━━━━━━<br/>60s sleep between<br/>variant runs"]
        WARMUP["Criterion Warm-up<br/>━━━━━━━━━━<br/>10s warm-up per variant<br/>cache stabilisation"]
        SEED["Seeded Gaussian Data<br/>━━━━━━━━━━<br/>d_x=10, d_y=2<br/>fixed seed reproducibility"]
    end

    subgraph Confounders ["CONFOUNDERS"]
        direction TB
        THERMAL["Thermal State<br/>━━━━━━━━━━<br/>CPU throttling<br/>Sequential runs at risk"]
        CACHE["Cache State<br/>━━━━━━━━━━<br/>Warm/cold cache<br/>First variant disadvantaged"]
        RAYON["Rayon Scheduling Noise<br/>━━━━━━━━━━<br/>Non-deterministic threads<br/>Fixed nproc mitigation"]
        DY2["d_y=2 + AVX2 Present<br/>━━━━━━━━━━<br/>Determines flat_simd path<br/>Controlled: always true"]
    end

    subgraph Treatment ["TREATMENT ASSIGNMENT"]
        direction LR
        VARIANT["Algorithm Variant<br/>━━━━━━━━━━<br/>baseline → heap_reuse<br/>→ flat_partial → flat_simd<br/>Causal isolation ladder"]
        N["n (Dataset Scale)<br/>━━━━━━━━━━<br/>1K / 5K / 10K<br/>Scales step fraction leverage"]
    end

    subgraph Mediators ["MEDIATING MECHANISMS"]
        direction TB
        ALLOC["Allocation Cost<br/>━━━━━━━━━━<br/>BinaryHeap::with_capacity(k+1)<br/>Measured: −0.011 (null path)"]
        DS["Data Structure Overhead<br/>━━━━━━━━━━<br/>BinaryHeap branchy push/pop<br/>vs flat Vec contiguous scan<br/>Attribution: 0.443"]
        SIMD["AVX2 SIMD Throughput<br/>━━━━━━━━━━<br/>2 pts/iter YMM registers<br/>_mm256_hadd_pd<br/>Attribution: 0.109"]
        STEPFRAC["y_heap Step Fraction<br/>━━━━━━━━━━<br/>baseline: 69.8%<br/>flat_simd: 27.6%<br/>High fraction = high leverage"]
    end

    subgraph Outcomes ["OUTCOME MEASUREMENT"]
        direction LR
        WALLTIME["Total Wall Time<br/>━━━━━━━━━━<br/>Criterion mean (ms)<br/>flat_simd ~2× at n=10K<br/>CI: 1.73–2.27×"]
        CORRECT["|ΔT| Correctness<br/>━━━━━━━━━━<br/>All 21 tests < 1e-12<br/>Tie-breaking comparator<br/>verified"]
    end

    subgraph Selection ["SELECTION / COLLIDER"]
        direction LR
        CRIT["Criterion Flat Sampling<br/>━━━━━━━━━━<br/>10 samples, flat mode<br/>CI width collider:<br/>wide but decisive (lb 1.73)"]
    end

    %% INSTRUMENT → CONFOUNDER (blocking) %%
    GAP60 -->|"blocks"| THERMAL
    WARMUP -->|"blocks"| CACHE
    SEED -->|"controls"| RAYON

    %% CONFOUNDER → OUTCOME (unblocked residual paths) %%
    THERMAL -->|"confounds"| WALLTIME
    CACHE -->|"confounds"| WALLTIME
    RAYON -->|"confounds"| WALLTIME

    %% CONFOUNDER → TREATMENT INTERACTION %%
    DY2 -->|"gates"| SIMD

    %% TREATMENT → MEDIATORS (causal) %%
    VARIANT -->|"heap_reuse isolates"| ALLOC
    VARIANT -->|"flat_partial isolates"| DS
    VARIANT -->|"flat_simd adds"| SIMD
    N -->|"scales leverage"| STEPFRAC

    %% MEDIATORS → STEP FRACTION (causal) %%
    ALLOC -->|"mediates (null)"| STEPFRAC
    DS -->|"mediates"| STEPFRAC
    SIMD -->|"mediates"| STEPFRAC

    %% STEP FRACTION → WALL TIME %%
    STEPFRAC -->|"causal"| WALLTIME

    %% TREATMENT → CORRECTNESS (direct path via comparator) %%
    VARIANT -->|"direct"| CORRECT

    %% SELECTION (post-treatment collider) %%
    WALLTIME -->|"selects"| CRIT
    VARIANT -->|"selects"| CRIT

    %% CLASS ASSIGNMENTS %%
    class GAP60,WARMUP,SEED newComponent;
    class THERMAL,CACHE,RAYON,DY2 stateNode;
    class VARIANT,N cli;
    class ALLOC,DS,SIMD,STEPFRAC handler;
    class WALLTIME,CORRECT output;
    class CRIT detector;
```

**Color Legend:**
| Color | Category | Description |
|-------|----------|-------------|
| Dark Blue | Treatment | Algorithm variant and scale factor |
| Teal | Confounder | Thermal, cache, scheduling, d_y=2 state |
| Green | Instrument | Blocking controls (thermal gap, warm-up, seed) |
| Orange | Mediator | Allocation cost, data structure overhead, SIMD throughput, step fraction |
| Dark Teal | Outcome | Wall time and correctness measurements |
| Red | Selection | Criterion sampling mode (post-treatment collider) |

## Identification Strategy

| Assumption | Testable? | Evidence |
|------------|-----------|----------|
| Variants are causally isolated (each adds exactly one change) | Yes | Code review: heap_reuse = clear() only; flat_partial = Vec+introselect; flat_simd = +AVX2 batch |
| Thermal confounder is blocked by 60s gap | Partially | 60s gap mitigates but does not eliminate; WSL2 thermal characteristics uncertain |
| d_y=2 / AVX2 path is constant across runs | Yes | Fixed Gaussian data d_y=2; `is_x86_feature_detected!("avx2")` stable per process |
| Criterion warm-up stabilizes cache state | Partially | 10s warm-up + flat sampling; W8 guard verified in run_criterion.sh |
| Sequential variant ordering does not introduce learning effects | Yes | Code paths share no mutable global state; thread-locals cleared per call |
| Tie-breaking comparator replicates BinaryHeap exactly | Yes | 21 correctness tests pass; `|ΔT| < 1e-12` confirmed |
| Step profiler isolation is valid (profiling instrumentation absent in Criterion) | Yes | W8 guard aborts run if CARGO_FEATURE_PROFILING is set |

## Unblocked Backdoor Paths

| Path | Variables | Severity | Mitigation |
|------|-----------|----------|------------|
| Thermal drift across sequential runs | Variant ordering → ThermalState → WallTime | Medium | 60s gap; not fully eliminated. Ordering effect: baseline first, flat_simd last (cool machine = pessimistic for baseline, warm = pessimistic for flat_simd — conservative for speedup claim) |
| Criterion 10-sample CI inflation | CriterionSampling → CI width collider | Low | CI lb=1.73 decisive; no Stage 2 escalation needed. Wide CI is known limitation documented in report |
| Rayon non-determinism at large n | Thread scheduling → WallTime variance | Low | Fixed nproc; parallel sections are embarrassingly parallel per-row with no inter-row dependencies |
