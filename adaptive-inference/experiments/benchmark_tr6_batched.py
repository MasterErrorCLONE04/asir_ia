import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models.transformer import MoELayer, MoETransformer
from training.moe.dispatcher import SparseMoEDispatcher
from training.moe.batched_dispatcher import BatchedMoEDispatcher
from training.profiling.dispatch_timing import DispatchTimingProfiler, set_global_timing_profiler
from training.optim import create_optimizer
from training.memory.manager import MemoryManager
from training.train import get_model_config


def run_three_way_layer_benchmark(device: torch.device) -> Dict[str, Any]:
    """Three-way single MoE layer comparison: Reference vs Sparse vs Batched."""
    print("\n=================== TR-6.1: THREE-WAY SINGLE MOE LAYER BENCHMARK ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    m_ref = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr = SparseMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr.load_state_dict(m_ref.state_dict())
    m_bat.load_state_dict(m_ref.state_dict())

    x = torch.randn(1, 128, d_model, device=device, dtype=torch.bfloat16)
    models = {"Reference": m_ref, "Sparse": m_spr, "Batched": m_bat}
    results = {}

    header = f"{'Dispatcher':<12} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'Std (ms)':<9} | {'p95 (ms)':<9}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for _ in range(30):
                _ = model(x)

        times = []
        for _ in range(100):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_t = np.mean(times)
        median_t = np.median(times)
        std_t = np.std(times)
        p95_t = np.percentile(times, 95)
        results[name] = {'mean_ms': mean_t, 'median_ms': median_t, 'std_ms': std_t, 'p95_ms': p95_t}
        print(f"{name:<12} | {mean_t:10.2f} | {median_t:11.2f} | {std_t:9.2f} | {p95_t:9.2f}")

    return results


def run_substage_profiling(device: torch.device) -> Dict[str, Any]:
    """Sub-stage timing for Sparse vs Batched using DispatchTimingProfiler."""
    print("\n=================== TR-6.2: SUB-STAGE TIMING (SPARSE VS BATCHED, 100 RUNS) ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    m_spr = SparseMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat.load_state_dict(m_spr.state_dict())

    x = torch.randn(1, 128, d_model, device=device, dtype=torch.bfloat16)
    results = {}

    for name, model in [("Sparse", m_spr), ("Batched", m_bat)]:
        prof = DispatchTimingProfiler(device)
        set_global_timing_profiler(prof)
        model.eval()
        with torch.no_grad():
            for _ in range(30):
                _ = model(x)
            prof.reset()
            for _ in range(100):
                _ = model(x)
        results[name] = prof.get_summary()["averages_ms"]

    header = f"{'Sub-stage':<12} | {'Sparse (ms)':<12} | {'Batched (ms)':<13} | {'Reduction':<10}"
    print(header)
    print("-" * len(header))
    for k in ["router", "dispatch", "experts", "combine"]:
        s_val = results["Sparse"].get(k, 0.0)
        b_val = results["Batched"].get(k, 0.0)
        red = (s_val - b_val) / max(s_val, 1e-5) * 100.0
        print(f"{k.capitalize():<12} | {s_val:12.2f} | {b_val:13.2f} | {red:9.1f}%")

    return results


def run_numerical_equivalence(device: torch.device) -> Dict[str, float]:
    """Exact numerical error metrics between Reference and Batched on GPU BF16."""
    print("\n=================== TR-6.3: NUMERICAL EQUIVALENCE (GPU BF16) ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    m_ref = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat.load_state_dict(m_ref.state_dict())

    x_ref = torch.randn(1, 128, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_bat = x_ref.clone().detach().requires_grad_(True)

    out_ref, _ = m_ref(x_ref)
    out_bat, _ = m_bat(x_bat)
    loss_ref = out_ref.pow(2).sum()
    loss_bat = out_bat.pow(2).sum()
    loss_ref.backward()
    loss_bat.backward()

    abs_diff = (out_bat.float() - out_ref.float()).abs()
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    loss_diff = (loss_bat.float() - loss_ref.float()).abs().item()

    max_grad_diff = 0.0
    for (_, p_ref), (_, p_bat) in zip(m_ref.named_parameters(), m_bat.named_parameters()):
        if p_ref.grad is not None and p_bat.grad is not None:
            g_diff = (p_bat.grad.float() - p_ref.grad.float()).abs().max().item()
            if g_diff > max_grad_diff:
                max_grad_diff = g_diff

    results = {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'loss_diff': loss_diff,
        'max_grad_diff': max_grad_diff
    }

    for k, v in results.items():
        print(f"  {k}: {v:.2e}")
    return results


def run_full_m2_e2e_benchmark(device: torch.device) -> Dict[str, Any]:
    """Full M2 end-to-end training step: Reference vs Sparse vs Batched."""
    print("\n=================== TR-6.4: FULL M2 END-TO-END TRAINING STEP (3-WAY) ===================")
    config = get_model_config('M2')
    input_ids = torch.randint(0, 100, (1, 128), device=device)
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    dispatcher_types = [
        ("Reference", "reference"),
        ("Sparse", "sparse"),
        ("Batched", "batched"),
    ]
    results = {}

    for label, dtype_str in dispatcher_types:
        # Use dispatcher_type for all three
        use_sparse = (dtype_str == "sparse")
        model = MoETransformer(
            vocab_size=100, d_model=1408, n_layers=5, num_heads=8,
            d_ff=config['d_ff'], moe=True, num_experts=32, top_k=2,
            use_sparse_dispatcher=use_sparse,
            dispatcher_type=dtype_str if dtype_str != "sparse" else "reference"
        ).to(device=device)
        model = MemoryManager.convert_precision_selective(model, "bf16-storage")
        optimizer = create_optimizer(model, opt_name="adam8bit", lr=1e-3)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            with amp_ctx:
                out, _ = model(input_ids)
                loss = out.sum()
            loss.backward()
            optimizer.step()

        times = []
        for _ in range(10):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            with amp_ctx:
                out, _ = model(input_ids)
                loss = out.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_t = np.mean(times)
        std_t = np.std(times)
        p50_t = np.percentile(times, 50)
        p95_t = np.percentile(times, 95)
        results[label] = {'mean_ms': mean_t, 'std_ms': std_t, 'p50_ms': p50_t, 'p95_ms': p95_t}
        print(f"{label:<12} | Mean: {mean_t:.2f} ms ± {std_t:.2f} ms | p50: {p50_t:.2f} ms | p95: {p95_t:.2f} ms")

        # Free VRAM between models
        del model, optimizer
        torch.cuda.empty_cache()

    # Compute speedups vs TR-3 frozen baseline
    tr3_baseline_ms = 189.81
    if "Batched" in results:
        bat_speedup = (tr3_baseline_ms - results["Batched"]["mean_ms"]) / tr3_baseline_ms * 100.0
        print(f"\nBatched vs TR-3 Sparse Baseline (189.81 ms): {bat_speedup:.2f}% {'improvement' if bat_speedup > 0 else 'regression'}")

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for TR-6 benchmark.")
        sys.exit(1)
    device = torch.device("cuda")

    layer_results = run_three_way_layer_benchmark(device)
    substage_results = run_substage_profiling(device)
    numerical_results = run_numerical_equivalence(device)
    e2e_results = run_full_m2_e2e_benchmark(device)

    summary = {
        'milestone': 'ASIR-TR-6',
        'hardware': 'NVIDIA GeForce RTX 3060',
        'tr6_1_layer_benchmark': layer_results,
        'tr6_2_substage_profiling': substage_results,
        'tr6_3_numerical_equivalence': numerical_results,
        'tr6_4_e2e_m2_benchmark': e2e_results
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "profiling", "M2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "asir_tr6_batched.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTR-6 results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
