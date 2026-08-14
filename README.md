# ASIR (Adaptive Sparse Inference & Routing)

**ASIR** is a research framework for memory-efficient training and high-performance execution of Mixture-of-Experts (MoE) Large Language Models on consumer GPU hardware (e.g., NVIDIA GeForce RTX 3060 12GB).

---

## Executive Summary

- **Problem**: Training a 1.626 Billion parameter MoE model (M2, 140M active parameters per token) in FP32 with standard AdamW requires **~24.66 GB VRAM**, causing Out-Of-Memory (OOM) failures.
- **Memory Solution (ASIR-TR-1)**: Selective `bf16-storage` precision combined with `bitsandbytes` 8-bit AdamW (`adam8bit`) reduces peak VRAM allocation to **7.01 GB**.
- **Compute Solution (ASIR-TR-6.1)**: `BatchedMoEDispatcher` implements **Grouped Expert GEMM** execution. Using GPU token packing (`torch.argsort`) and stacked weights, it runs all experts in a single batch via `torch.bmm`, reducing CUDA expert kernel launch overhead by **>93%**.
- **Residency Management (ASIR-TR-8.1)**: `ResidencyManager` implements a multi-tier memory subsystem (VRAM / RAM) with active **Lease ownership** to prevent expert eviction during execution. LRU is shown to **collapse completely (0% hit rate)** under layer-cycling MoE access, while FFU/LFRU reach stable hit rates of ~17-19%.
- **Async Prefetch Path (ASIR-TR-8.2)**: `ThreadPoolExecutor` and `Future` abstractions enable background non-blocking disk reads, yielding **+91.7% throughput speedup** for **Top-2** transition prefetching (**30.83 tok/s**, reaching **96.2% of the Oracle Upper Bound** of 32.03 tok/s). Prefetch buffer isolation (`prefetch_cache`) prevents eviction thrashing.

---

## System Architecture

```text
                               ASIR Framework
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  Memory Subsystem                                       Compute Subsystem
  (runtime/ & training/memory/)                           (training/moe/)
  ├── residency.py (VRAM/RAM Lease Manager)             ├── reference.py  (Oracle baseline)
  ├── expert_cache.py (Aisolated Prefetch Cache)        ├── dispatcher.py (Sparse MoE)
  ├── expert_store.py (Async ThreadPoolExecutor)        └── batched_dispatcher.py (Grouped GEMM)
```

### Key Modules

- **[`runtime/expert_store.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/runtime/expert_store.py)**: `NVMeExpertStore` implements background non-blocking loading of expert weights via a `ThreadPoolExecutor`, yielding `Future` weights immediately.
- **[`runtime/expert_cache.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/runtime/expert_cache.py)**: `ExpertCache` contains an isolated speculative buffer `prefetch_cache` to store background futures, preventing eviction thrashing of the main LRU active cache.
- **[`runtime/residency.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/runtime/residency.py)**: `ResidencyManager` manages VRAM residency states and lease allocations.

---

## Experimental Milestones & Benchmark Progression

| Milestone | Objective | Key Findings & Results | Status |
| :--- | :--- | :--- | :---: |
| **ASIR-TR-1** | VRAM Memory Reduction | selective `bf16-storage` + `adam8bit` = **7.01 GB Peak VRAM** (RTX 3060 12GB compatible). | ✅ PASS |
| **ASIR-TR-6.1**| Grouped GEMM Engine | Stacked expert parameters. Replaced sequential loop with single `torch.bmm`, reducing expert launches by **>93%**. | ✅ PASS |
| **ASIR-TR-8.1**| Residency & Lease Manager | VRAM/RAM tiering with Lease ownership. Proved **LRU fails completely (0% hit rate)** under cycling layers; **LFU/LFRU** achieved ~19% hit rate. | ✅ PASS |
| **ASIR-TR-8.2**| Async Prefetch Engine | Background thread execution and double buffering solapado. Reached **30.83 tok/s** for **Top-2** prediction (96.2% of Oracle Upper Bound). | ✅ PASS |

---

## Summary of Empirical Prefetch Metrics (TR-8.2)

Evaluated under 30 token decode generation steps (capacity = 4 experts/layer):

- **No Prefetch**: Throughput: **16.08 tok/s**, Hit Rate: **69.86%**
- **Async Prefetch (Top-1)**: Throughput: **26.31 tok/s**, Hit Rate: **78.45%**, Precision: **91.04%**, Recall: **8.59%**
- **Async Prefetch (Top-2)**: Throughput: **30.83 tok/s**, Hit Rate: **80.56%**, Precision: **91.57%**, Recall: **10.70%** (Optimal)
- **Async Prefetch (Top-4)**: Throughput: **30.60 tok/s**, Hit Rate: **81.69%**, Precision: **92.31%**, Recall: **11.83%**
- **Oracle Prefetch** *(Techo)*: Throughput: **32.03 tok/s**, Hit Rate: **82.11%**, Precision: **100.00%**, Recall: **12.54%**

---

## Directory Layout

```text
.
├── R specification.md               # Research specification and roadmap
├── README.md                        # Project documentation
├── adaptive-inference/
│   ├── experiments/
│   │   ├── benchmark_tr8_residency.py # TR-8.1 residency & capacity sweeps
│   │   ├── benchmark_tr82_prefetch.py # TR-8.2 prefetch sweeps & oracle comparisons
│   │   └── measure_prefetch_overhead.py # Micro-benchmark for async read overheads
│   ├── runtime/
│   │   ├── residency.py             # ResidencyManager & Lease contracts
│   │   ├── expert_cache.py          # Isolated prefetch cache
│   │   └── expert_store.py          # Background ThreadPoolExecutor store
│   ├── tests/
│   │   ├── test_residency.py        # ResidencyManager & Lease unit tests
│   │   └── test_prefetch_async.py   # Prefetch engine async unit tests
│   └── training/
│       ├── memory/                  # Precision & activation checkpointing
│       └── moe/                     # Reference, Sparse, and Batched MoE layers
```

---

## Quickstart Guide

### 1. Installation

```bash
python3 -m venv ~/venv-adaptive
source ~/venv-adaptive/bin/activate
cd adaptive-inference
pip install -r requirements-gpu.txt
```

### 2. Running Unit Tests

```bash
# Run all unit tests
.venv/bin/python -m unittest discover tests/
```

### 3. Running Async Prefetch Benchmarks

```bash
# Evaluates Top-K and Oracle policies
.venv/bin/python experiments/benchmark_tr82_prefetch.py

# Measures disk load vs async overlap overhead
.venv/bin/python experiments/measure_prefetch_overhead.py
```

---

## License

Internal Research Repository — ASIR Project.
