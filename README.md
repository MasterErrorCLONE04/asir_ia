# ASIR (Adaptive Sparse Inference & Routing)

**ASIR** is a research framework for memory-efficient training and high-performance execution of Mixture-of-Experts (MoE) Large Language Models on consumer GPU hardware (e.g., NVIDIA GeForce RTX 3060 12GB).

---

## Executive Summary

- **Problem**: Training a 1.626 Billion parameter MoE model (M2, 140M active parameters per token) in FP32 with standard AdamW requires **~24.66 GB VRAM** (Weights: 6.06 GB, Gradients: 6.06 GB, Optimizer States: 12.12 GB), exceeding consumer GPU VRAM limits and causing Out-Of-Memory (OOM) failures.
- **Memory Solution (ASIR-TR-1)**: Selective `bf16-storage` precision (preserving LayerNorm, Embeddings, and Router in FP32 for numerical stability) combined with `bitsandbytes` 8-bit AdamW (`adam8bit`) reduces peak VRAM allocation to **7.01 GB**, fitting M2 comfortably within 12 GB VRAM.
- **Compute Solution (ASIR-TR-3 to TR-6)**: `BatchedMoEDispatcher` replaces $O(E \times K)$ Python loops with GPU token packing (`torch.argsort`), contiguous slice execution, and inline output accumulation (`scatter_add_`). This eliminates **83.6% of CUDA kernel launch calls** (from 1381 to 227 per forward pass) and accelerates end-to-end training step latency by **~36.1%** (from 258.02 ms to 164.94 ms/step) with **exact bit-level numerical equivalence** (`0.00e+00` error).

---

## System Architecture

```text
                               ASIR Framework
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  Memory Subsystem                                       Compute Subsystem
  (training/memory/)                                     (training/moe/)
  ├── manager.py   (bf16-storage conversion)             ├── reference.py  (Oracle baseline)
  └── checkpoint.py (Activation checkpointing)          ├── dispatcher.py (Sparse MoE)
                                                         ├── batched_dispatcher.py (Batched MoE)
                                                         └── combine.py    (Tensor accumulation)
```

### Key Modules

- **[`training/memory/manager.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/memory/manager.py)**: `MemoryManager` performs selective precision conversion. Expert linear weights are stored in `bfloat16`, while numerically sensitive layers (LayerNorm, Embeddings, Router) remain in `float32`.
- **[`training/optim/factory.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/optim/factory.py)**: Factory supporting `adamw` (FP32) and `adam8bit` (8-bit quantized optimizer states).
- **[`training/moe/batched_dispatcher.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/moe/batched_dispatcher.py)**: `BatchedMoEDispatcher` sorts tokens by expert assignment in a single GPU operation, computes slice offsets via `torch.bincount`, executes experts on contiguous token segments, and accumulates outputs via `scatter_add_`.
- **[`training/profiling/dispatch_timing.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/training/profiling/dispatch_timing.py)**: CUDA event-synchronized profiler measuring sub-stage latency across `Router`, `Dispatch`, `Experts`, and `Combine`.
- **[`analysis/moe_telemetry.py`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/analysis/moe_telemetry.py)**: Telemetry tracking effective active experts ($N_{\text{eff}}$), expert co-occurrence matrices, and token distribution Coefficient of Variation ($CV$).

---

## Experimental Milestones & Benchmark Progression

| Milestone | Objective | Key Findings & Results | Status |
| :--- | :--- | :--- | :---: |
| **ASIR-TR-0** | Baseline GPU Environment & Profiling Infra | Integrated `MemoryProfiler` and `DispatchTimingProfiler`. Dockerfile.gpu and GPU requirements established. | ✅ PASS |
| **ASIR-TR-1** | VRAM Memory Reduction | FP32 + AdamW = 24.66 GB (OOM). `bf16-storage` + `adam8bit` = **7.01 GB Peak VRAM** (RTX 3060 12GB compatible). | ✅ PASS |
| **ASIR-TR-2** | MoE Sub-stage Timing | Instrumented `Router`, `Dispatch`, `Experts`, and `Combine` timing hooks across MoE layers. | ✅ PASS |
| **ASIR-TR-3** | Sparse MoE Dispatcher | `SparseMoEDispatcher` evaluated active experts only. `max_abs_error = 0.00e+00`. E2E step latency reduced from 258.02 ms to 189.81 ms (~26.4% speedup). | ✅ PASS |
| **ASIR-TR-4** | CUDA Overhead Decomposition | Diagnosed 320 expert kernel invocations per step. FFN compute utilization was **0.19%**, indicating low Tensor Core saturation due to small micro-GEMMs (8 tokens/expert). | ✅ PASS |
| **ASIR-TR-5** | Causal Kernel & Launch Analysis | PyTorch Profiler proved CPU API launch time (24.01 ms) > GPU kernel execution time (22.73 ms). GEMM batch size scaling showed 28x TFLOPS jump when batching tokens (0.76 TFLOPS at 8 tokens vs 21.95 TFLOPS at 1024 tokens). | ✅ PASS |
| **ASIR-TR-6** | Batched Expert Execution | `BatchedMoEDispatcher` token packing (`argsort` + `scatter_add_`) eliminated explicit Combine phase (30.39 ms -> 0.00 ms inline). Reduced step latency to **164.94 ms/step** (~36.1% total speedup over Base). | ✅ PASS |
| **ASIR-TR-7** | Hardware Efficiency Diagnostic | Proved `BatchedMoEDispatcher` reduced CPU launch time by **73%** (22.41 ms -> 6.01 ms) and CUDA launch calls by **83.6%** (1381 -> 227), shifting system bottleneck from CPU launch-bound to GPU compute-bound. | ✅ PASS |

### Summary of Performance & Memory Metrics

```text
M2 Model: 1.626B Total Parameters / 140M Active Parameters per Token
Hardware: NVIDIA GeForce RTX 3060 12GB VRAM

VRAM Usage:
  FP32 + AdamW:            24.66 GB  [OOM]
  BF16 + Adam8bit:          7.01 GB  [PASS - 71.5% VRAM Reduction]

Full Step Latency (5 MoE Layers, Forward + Backward + AdamW Step):
  Reference MoELayer:      258.02 ms/step  (p50: 211.74 ms, p95: 482.76 ms)
  Sparse MoE Dispatcher:   189.81 ms/step  (p50: 187.68 ms, p95: 314.87 ms)  [-26.4%]
  Batched MoE Dispatcher:  164.94 ms/step  (p50: 161.35 ms, p95: 260.76 ms)  [-36.1%]

Numerical Equivalence (GPU BF16):
  Max Absolute Error:      0.00e+00
  Max Gradient Error:      0.00e+00
```

---

## Directory Layout

```text
.
├── R specification.md               # Research specification and roadmap
├── README.md                        # Project documentation
├── adaptive-inference/
│   ├── analysis/
│   │   └── moe_telemetry.py         # Expert routing telemetry
│   ├── experiments/
│   │   ├── benchmark_dispatcher.py  # TR-3 baseline benchmark
│   │   ├── benchmark_tr4_overheads.py # TR-4 diagnostic script
│   │   ├── benchmark_tr5_causality.py # TR-5 PyTorch profiler causality script
│   │   ├── benchmark_tr6_batched.py   # TR-6 3-way benchmark
│   │   └── benchmark_tr7_efficiency.py# TR-7 hardware efficiency diagnostic
│   ├── requirements-gpu.txt         # PyTorch & bitsandbytes GPU requirements
│   ├── Dockerfile.gpu               # CUDA 12.1 runtime environment
│   ├── tests/
│   │   ├── test_gpu_config.py       # GPU CLI flag & profiler tests
│   │   ├── test_memory_efficient.py # Precision manager & optimizer tests
│   │   ├── test_moe_dispatch.py     # Sparse dispatcher correctness tests
│   │   └── test_batched_dispatcher.py # Batched dispatcher correctness tests
│   └── training/
│       ├── memory/                  # Precision & activation checkpointing
│       ├── moe/                     # Reference, Sparse, and Batched MoE layers
│       ├── models/                  # TransformerBlock & MoETransformer
│       ├── optim/                   # AdamW & Adam8bit optimizer factory
│       ├── profiling/               # Memory & CUDA timing profilers
│       └── train.py                 # Training entry point
└── results/
    └── profiling/
        └── M2/                      # Immutable milestone json artifacts
            ├── asir_tr3_summary.json
            ├── asir_tr4_overheads.json
            ├── asir_tr5_causality.json
            ├── asir_tr6_batched.json
            └── asir_tr7_efficiency.json
```

---

## Quickstart Guide

### 1. Installation

Using virtual environment (WSL / Ubuntu / Linux):

```bash
python3 -m venv ~/venv-adaptive
source ~/venv-adaptive/bin/activate
cd adaptive-inference
pip install -r requirements-gpu.txt
```

### 2. Running Unit Tests

Run the CPU-first test suite covering memory efficiency, MoE correctness, and precision conversions:

```bash
PYTHONPATH=. pytest tests/ -v
```

### 3. Running Training with Memory Optimization

Execute M2 model training in `bf16-storage` mode using 8-bit AdamW on GPU:

```bash
PYTHONPATH=. python training/train.py \
  --model M2 \
  --device cuda \
  --precision bf16-storage \
  --optimizer adam8bit \
  --max_steps 100
```

### 4. Reproducing Milestone Benchmarks

To execute the three-way GPU benchmark comparing Reference, Sparse, and Batched MoE dispatchers:

```bash
PYTHONPATH=. python experiments/benchmark_tr6_batched.py
```

To run the hardware efficiency and profiler decomposition diagnostic:

```bash
PYTHONPATH=. python experiments/benchmark_tr7_efficiency.py
```

---

## Immutable Baseline Artifacts

All benchmark outputs, profiler summaries, and numerical equivalence logs are preserved in `results/profiling/M2/`:

- [`results/profiling/M2/asir_tr3_summary.json`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/results/profiling/M2/asir_tr3_summary.json): Baseline reference for Sparse MoE Dispatcher.
- [`results/profiling/M2/asir_tr4_overheads.json`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/results/profiling/M2/asir_tr4_overheads.json): Overheads breakdown and kernel count diagnosis.
- [`results/profiling/M2/asir_tr5_causality.json`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/results/profiling/M2/asir_tr5_causality.json): PyTorch profiler CPU launch vs GPU kernel execution analysis.
- [`results/profiling/M2/asir_tr6_batched.json`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/results/profiling/M2/asir_tr6_batched.json): Batched dispatcher performance & exact equivalence results.
- [`results/profiling/M2/asir_tr7_efficiency.json`](file:///c:/Users/Usuario/asir_ia/adaptive-inference/results/profiling/M2/asir_tr7_efficiency.json): Hardware efficiency & launch-bound vs compute-bound analysis.

---

## License

Internal Research Repository — ASIR Project.
