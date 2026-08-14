import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models.transformer import MoELayer
from training.moe.dispatcher import SparseMoEDispatcher
from training.moe.batched_dispatcher import BatchedMoEDispatcher
from training.profiling.dispatch_timing import DispatchTimingProfiler, set_global_timing_profiler


def compute_single_layer_ffn_flops(num_tokens: int, d_model: int, d_ff: int, top_k: int) -> float:
    """
    Theoretical FLOPs for one MoE layer forward pass.
    Each FFN expert: in_proj (d_model->d_ff) + out_proj (d_ff->d_model) = 2 matmuls.
    Each matmul = 2*M*N FLOPs. With top_k experts per token:
      FLOPs = top_k * num_tokens * 2 * (d_model*d_ff + d_ff*d_model)
            = top_k * num_tokens * 4 * d_model * d_ff
    """
    return float(top_k * num_tokens * 4 * d_model * d_ff)


def run_tr7_1_tflops_comparison(device: torch.device) -> Dict[str, Any]:
    """Three-way TFLOPS measurement: Reference vs Sparse vs Batched."""
    print("\n=================== TR-7.1: THREE-WAY TFLOPS MEASUREMENT ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    seq_len = 128
    num_tokens = 1 * seq_len  # batch=1

    theoretical_flops = compute_single_layer_ffn_flops(num_tokens, d_model, d_ff, K)
    theoretical_gflops = theoretical_flops / 1e9
    rtx3060_peak_tflops = 71.7

    m_ref = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr = SparseMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr.load_state_dict(m_ref.state_dict())
    m_bat.load_state_dict(m_ref.state_dict())

    x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16)
    models = {"Reference": m_ref, "Sparse": m_spr, "Batched": m_bat}
    results = {}

    print(f"Theoretical FFN FLOPs per layer call: {theoretical_gflops:.3f} GFLOPs")
    print(f"RTX 3060 Peak BF16 Tensor Core: {rtx3060_peak_tflops} TFLOPS\n")

    header = f"{'Dispatcher':<12} | {'Latency (ms)':<13} | {'Achieved TFLOPS':<16} | {'GPU Util %':<10} | {'Speedup vs TR-5 8-tok':<22}"
    print(header)
    print("-" * len(header))

    tr5_baseline_tflops = 0.764  # 8 tokens from TR-5

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for _ in range(30):
                _ = model(x)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - t0) * 10.0  # ms per call

        latency_sec = max(latency_ms / 1000.0, 1e-9)
        achieved_tflops = (theoretical_flops / latency_sec) / 1e12
        utilization = (achieved_tflops / rtx3060_peak_tflops) * 100.0
        tflops_improvement = achieved_tflops / tr5_baseline_tflops

        results[name] = {
            'latency_ms': latency_ms,
            'achieved_tflops': achieved_tflops,
            'gpu_utilization_pct': utilization,
            'tflops_vs_tr5_8tok': tflops_improvement
        }
        print(f"{name:<12} | {latency_ms:13.2f} | {achieved_tflops:16.3f} | {utilization:9.2f}% | {tflops_improvement:21.1f}x")

    results['theoretical_gflops'] = theoretical_gflops
    results['rtx3060_peak_tflops'] = rtx3060_peak_tflops
    return results


def run_tr7_2_profiler_decomposition(device: torch.device) -> Dict[str, Any]:
    """Three-way PyTorch Profiler: CPU launch time vs GPU kernel execution."""
    print("\n=================== TR-7.2: CPU LAUNCH vs GPU EXECUTION (THREE-WAY) ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    m_ref = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr = SparseMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr.load_state_dict(m_ref.state_dict())
    m_bat.load_state_dict(m_ref.state_dict())

    x = torch.randn(1, 128, d_model, device=device, dtype=torch.bfloat16)
    models = {"Reference": m_ref, "Sparse": m_spr, "Batched": m_bat}
    results = {}
    num_iterations = 10

    header = f"{'Dispatcher':<12} | {'CPU Launch (ms)':<16} | {'GPU Kernel (ms)':<16} | {'Launch/Kernel Ratio':<20}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x)

        key_averages = prof.key_averages()

        cpu_launch_us = 0.0
        cuda_kernel_us = 0.0

        for evt in key_averages:
            if "cudaLaunch" in evt.key or "cuLaunch" in evt.key:
                cpu_launch_us += getattr(evt, 'cpu_time_total', 0.0)
            cuda_time = getattr(evt, 'cuda_time_total',
                        getattr(evt, 'device_time_total',
                        getattr(evt, 'self_device_time_total', 0.0)))
            cuda_kernel_us += cuda_time

        cpu_launch_ms = (cpu_launch_us / num_iterations) / 1000.0
        cuda_kernel_ms = (cuda_kernel_us / num_iterations) / 1000.0
        ratio = cpu_launch_ms / max(cuda_kernel_ms, 1e-6)

        results[name] = {
            'cpu_launch_ms': cpu_launch_ms,
            'cuda_kernel_ms': cuda_kernel_ms,
            'launch_kernel_ratio': ratio
        }
        print(f"{name:<12} | {cpu_launch_ms:16.3f} | {cuda_kernel_ms:16.3f} | {ratio:19.3f}")

    return results


def run_tr7_3_kernel_launch_count(device: torch.device) -> Dict[str, Any]:
    """Estimate CUDA kernel launch count per MoE layer forward pass."""
    print("\n=================== TR-7.3: KERNEL LAUNCH COUNT PER MOE LAYER FORWARD ===================")

    d_model, d_ff, E, K = 1408, 3516, 32, 2
    m_ref = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr = SparseMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_bat = BatchedMoEDispatcher(d_model=d_model, d_ff=d_ff, num_experts=E, top_k=K).to(device=device, dtype=torch.bfloat16)
    m_spr.load_state_dict(m_ref.state_dict())
    m_bat.load_state_dict(m_ref.state_dict())

    x = torch.randn(1, 128, d_model, device=device, dtype=torch.bfloat16)
    models = {"Reference": m_ref, "Sparse": m_spr, "Batched": m_bat}
    results = {}

    header = f"{'Dispatcher':<12} | {'CUDA Kernel Events':<20} | {'cudaLaunch Calls':<18}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            with torch.no_grad():
                _ = model(x)  # single forward pass

        total_cuda_events = 0
        total_launch_calls = 0
        for evt in prof.key_averages():
            cuda_time = getattr(evt, 'cuda_time_total',
                        getattr(evt, 'device_time_total',
                        getattr(evt, 'self_device_time_total', 0.0)))
            if cuda_time > 0:
                total_cuda_events += evt.count
            if "cudaLaunch" in evt.key or "cuLaunch" in evt.key:
                total_launch_calls += evt.count

        results[name] = {
            'cuda_kernel_events': total_cuda_events,
            'cuda_launch_calls': total_launch_calls
        }
        print(f"{name:<12} | {total_cuda_events:<20} | {total_launch_calls:<18}")

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for TR-7 diagnostics.")
        sys.exit(1)
    device = torch.device("cuda")

    res_1 = run_tr7_1_tflops_comparison(device)
    res_2 = run_tr7_2_profiler_decomposition(device)
    res_3 = run_tr7_3_kernel_launch_count(device)

    summary = {
        'milestone': 'ASIR-TR-7',
        'status': 'DIAGNOSTIC_COMPLETE',
        'hardware': 'NVIDIA GeForce RTX 3060',
        'tr7_1_tflops': res_1,
        'tr7_2_profiler': res_2,
        'tr7_3_kernel_launches': res_3
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "profiling", "M2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "asir_tr7_efficiency.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTR-7 results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
