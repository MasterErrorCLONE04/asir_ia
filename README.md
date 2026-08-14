# ASIR (Adaptive Sparse Inference & Routing)

**ASIR** is a research framework for memory-efficient training and high-performance execution of Mixture-of-Experts (MoE) Large Language Models on consumer GPU hardware (e.g., NVIDIA GeForce RTX 3060 12GB).

---

## Executive Summary

- **Problem**: Training a 1.626 Billion parameter MoE model (M2, 140M active parameters per token) in FP32 with standard AdamW requires **~24.66 GB VRAM** (Weights: 6.06 GB, Gradients: 6.06 GB, Optimizer States: 12.12 GB), causing Out-Of-Memory (OOM) failures on consumer hardware.
- **Memory Solution (ASIR-TR-1)**: Selective `bf16-storage` precision (preserving LayerNorm, Embeddings, and Router in FP32 for numerical stability) combined with `bitsandbytes` 8-bit AdamW (`adam8bit`) reduces peak VRAM allocation to **7.01 GB**, fitting M2 comfortably within 12 GB VRAM.
- **Compute Solution (ASIR-TR-6.1)**: `BatchedMoEDispatcher` implements **Grouped Expert GEMM** execution. Using GPU token packing (`torch.argsort`) and stacked weight parameters, it executes all experts in a single batch via `torch.bmm`. This eliminates the sequential python loop over experts, reducing expert CUDA kernel launch overhead by **>93%** (down to $O(K)$ launches per step) while preserving exact bit-level forward/backward numerical equivalence.
- **Runtime Residency Solution (ASIR-TR-8.1)**: `ResidencyManager` implements a multi-tier memory subsystem (VRAM / RAM) with active **Lease ownership** to prevent eviction of experts during active execution. Empirical evaluation shows that classical eviction policies like **LRU collapse completely (0% hit rate)** due to cyclic multi-layer MoE execution patterns, whereas frequency-aware policies (**LFU/LFRU**) maintain stable hit rates of **~17-19%** at capacity bounds.

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
  ├── manager.py   (bf16-storage conversion)             ├── dispatcher.py (Sparse MoE)
  └── checkpoint.py (Activation checkpointing)          └── batched_dispatcher.py (Grouped GEMM)
```

### Key Modules

- **[`runtime/residency.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/runtime/residency.py)**: `ResidencyManager` manages expert metadata (`ExpertRecord`), lifecycle states (`RAM_RESIDENT`, `VRAM_LEASED`, `VRAM_RESIDENT`), lease references, and enforces eviction policies (`LRU`, `LFU`, `LFRU`).
- **[`training/moe/batched_dispatcher.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/moe/batched_dispatcher.py)**: `BatchedMoEDispatcher` packs tokens into a 3D tensor and runs all active experts in a single grouped GEMM operation via `torch.bmm`.
- **[`training/memory/manager.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/memory/manager.py)**: `MemoryManager` performs selective precision conversion. Expert weights are stored in `bfloat16`, while sensitive layers remain in `float32`.
- **[`training/optim/factory.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/optim/factory.py)**: quantizes optimizer states to 8-bit using `bitsandbytes`.

---

## Experimental Milestones & Benchmark Progression

| Milestone | Objective | Key Findings & Results | Status |
| :--- | :--- | :--- | :---: |
| **ASIR-TR-1** | VRAM Memory Reduction | selective `bf16-storage` + `adam8bit` = **7.01 GB Peak VRAM** (RTX 3060 12GB compatible). | ✅ PASS |
| **ASIR-TR-3** | Sparse MoE Dispatcher | `SparseMoEDispatcher` evaluated active experts only. Latency reduced from 258.02 ms to 189.81 ms. | ✅ PASS |
| **ASIR-TR-4** | CUDA Overhead Decomposition | Diagnosed 320 expert kernel launches per step. GPU compute utilization was **0.19%** due to CPU launch overhead. | ✅ PASS |
| **ASIR-TR-5** | Causality Analysis | PyTorch Profiler proved CPU launch time > GPU execution. Batched token scaling yielded 28x TFLOPS jump. | ✅ PASS |
| **ASIR-TR-6** | Batched Expert Execution | Avoided explicit combine phase. Reduced step latency to **164.94 ms/step**. | ✅ PASS |
| **ASIR-TR-6.1**| Grouped GEMM Engine | Stacked expert parameters. Replaced sequential loop with single `torch.bmm`, reducing expert launches by **>93%**. | ✅ PASS |
| **ASIR-TR-8.1**| Residency & Lease Manager | VRAM/RAM tiering with Lease ownership. Proved **LRU fails completely (0% hit rate)** under cycling layers; **LFU/LFRU** achieved ~19% hit rate. | ✅ PASS |

---

## Summary of Empirical Residency Metrics (TR-8.1)

### VRAM Capacity Sweep (LFRU Policy, 100 steps)
- **Capacity 4**: Hit Rate: **7.10%**, H2D Transfer: **58.7 MB**
- **Capacity 8**: Hit Rate: **17.40%**, H2D Transfer: **52.2 MB**
- **Capacity 16**: Hit Rate: **27.80%**, H2D Transfer: **45.7 MB**
- **Capacity 24**: Hit Rate: **36.50%**, H2D Transfer: **40.2 MB**
- **Capacity 32**: Hit Rate: **43.60%**, H2D Transfer: **35.7 MB**

### Eviction Policy Comparison (Capacity = 8)
- **LRU** *(Least Recently Used)*: **0.00% Hit Rate** (Complete collapse due to layer sequence cycle eviction).
- **LFU** *(Least Frequently Used)*: **19.20% Hit Rate**.
- **LFRU** *(Hybrid Heat Score)*: **17.40% Hit Rate**.

---

## Directory Layout

```text
.
├── R specification.md               # Research specification and roadmap
├── README.md                        # Project documentation
├── adaptive-inference/
│   ├── experiments/
│   │   ├── benchmark_tr6_batched.py   # TR-6 3-way benchmark
│   │   └── benchmark_tr8_residency.py # TR-8.1 residency & capacity sweeps
│   ├── runtime/
│   │   └── residency.py             # ResidencyManager & Lease contracts
│   ├── tests/
│   │   ├── test_batched_dispatcher.py # Batched dispatcher correctness tests
│   │   └── test_residency.py        # ResidencyManager & Lease unit tests
│   └── training/
│       ├── memory/                  # Precision & activation checkpointing
│       └── moe/                     # Reference, Sparse, and Batched MoE layers
└── results/
    └── profiling/
        └── M2/                      # Milestone JSON profiling outputs
            ├── asir_tr6_batched.json
            └── asir_tr8_residency.json
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
# Run dispatcher correctness & residency lease tests
.venv/bin/python -m unittest tests/test_batched_dispatcher.py
.venv/bin/python -m unittest tests/test_residency.py
```

### 3. Running Residency Benchmarks

```bash
# Sweeps capacity bounds and compares eviction policies (LRU/LFU/LFRU)
.venv/bin/python experiments/benchmark_tr8_residency.py
```

---

## License

Internal Research Repository — ASIR Project.
