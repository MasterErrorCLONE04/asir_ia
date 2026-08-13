import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models.transformer import MoELayer, MoETransformer
from training.moe.dispatcher import SparseMoEDispatcher
from training.profiling.dispatch_timing import DispatchTimingProfiler, set_global_timing_profiler
from training.optim import create_optimizer
from training.memory.manager import MemoryManager
from training.train import get_model_config

def compute_numerical_errors(ref_out: torch.Tensor, opt_out: torch.Tensor,
                             ref_loss: torch.Tensor, opt_loss: torch.Tensor,
                             ref_model: nn.Module, opt_model: nn.Module) -> Dict[str, float]:
    """
    Computes exact numerical error bounds between Reference and Optimized implementations.
    """
    abs_diff = (opt_out.float() - ref_out.float()).abs()
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()

    ref_denom = ref_out.float().abs() + 1e-7
    max_rel_error = (abs_diff / ref_denom).max().item()

    loss_diff = (opt_loss.float() - ref_loss.float()).abs().item()

    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(ref_model.named_parameters(), opt_model.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            g_diff = (p2.grad.float() - p1.grad.float()).abs().max().item()
            if g_diff > max_grad_diff:
                max_grad_diff = g_diff

    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'loss_diff': loss_diff,
        'max_grad_diff': max_grad_diff
    }

def run_sequence_length_sweep(device: torch.device):
    print("\n=================== 1. SEQUENCE LENGTH SWEEP & NUMERICAL ACCURACY ===================")
    seq_lengths = [16, 64, 128, 256, 512]
    
    header = f"{'Seq Len':<8} | {'MoE Base (ms)':<14} | {'Sparse Opt (ms)':<15} | {'Speedup':<9} | {'Max Abs Err':<12} | {'Max Grad Err':<12}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        m_ref = MoELayer(d_model=1408, d_ff=3516, num_experts=32, top_k=2).to(device=device, dtype=torch.bfloat16)
        m_opt = SparseMoEDispatcher(d_model=1408, d_ff=3516, num_experts=32, top_k=2).to(device=device, dtype=torch.bfloat16)
        m_opt.load_state_dict(m_ref.state_dict())

        x_ref = torch.randn(1, seq_len, 1408, device=device, dtype=torch.bfloat16, requires_grad=True)
        x_opt = x_ref.clone().detach().requires_grad_(True)

        # Numerical Equivalence & Gradient check
        out_ref, _ = m_ref(x_ref)
        out_opt, _ = m_opt(x_opt)
        loss_ref = out_ref.pow(2).sum()
        loss_opt = out_opt.pow(2).sum()
        loss_ref.backward()
        loss_opt.backward()

        errs = compute_numerical_errors(out_ref, out_opt, loss_ref, loss_opt, m_ref, m_opt)

        # Latency Benchmark (100 iterations)
        m_ref.eval()
        m_opt.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(30):
                _ = m_ref(x_ref)
                _ = m_opt(x_opt)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = m_ref(x_ref)
        torch.cuda.synchronize(device)
        time_ref = (time.perf_counter() - t0) * 10.0 # ms per call (100 calls total -> 1000/100 = * 10)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = m_opt(x_opt)
        torch.cuda.synchronize(device)
        time_opt = (time.perf_counter() - t0) * 10.0

        speedup = (time_ref - time_opt) / time_ref * 100.0
        print(f"{seq_len:<8} | {time_ref:14.2f} | {time_opt:15.2f} | {speedup:8.1f}% | {errs['max_abs_error']:12.2e} | {errs['max_grad_diff']:12.2e}")

def run_statistical_timing_benchmark(device: torch.device):
    print("\n=================== 2. STATISTICAL SUB-STAGE TIMING (100 RUNS) ===================")
    m_ref = MoELayer(d_model=1408, d_ff=3516, num_experts=32, top_k=2).to(device=device, dtype=torch.bfloat16)
    m_opt = SparseMoEDispatcher(d_model=1408, d_ff=3516, num_experts=32, top_k=2).to(device=device, dtype=torch.bfloat16)
    m_opt.load_state_dict(m_ref.state_dict())

    x = torch.randn(1, 64, 1408, device=device, dtype=torch.bfloat16)

    # 1. Base MoELayer
    prof_ref = DispatchTimingProfiler(device)
    set_global_timing_profiler(prof_ref)
    with torch.no_grad():
        for _ in range(30): _ = m_ref(x) # Warmup
        prof_ref.reset()
        for _ in range(100): _ = m_ref(x)
    sum_ref = prof_ref.get_summary()["averages_ms"]

    # 2. Sparse MoE Dispatcher
    prof_opt = DispatchTimingProfiler(device)
    set_global_timing_profiler(prof_opt)
    with torch.no_grad():
        for _ in range(30): _ = m_opt(x) # Warmup
        prof_opt.reset()
        for _ in range(100): _ = m_opt(x)
    sum_opt = prof_opt.get_summary()["averages_ms"]

    header = f"{'Sub-stage':<12} | {'MoE Base (ms)':<14} | {'Sparse Opt (ms)':<15} | {'Reduction':<10}"
    print(header)
    print("-" * len(header))
    for k in ["router", "dispatch", "experts", "combine"]:
        b_val = sum_ref.get(k, 0.0)
        o_val = sum_opt.get(k, 0.0)
        red = (b_val - o_val) / max(b_val, 1e-5) * 100.0
        print(f"{k.capitalize():<12} | {b_val:14.2f} | {o_val:15.2f} | {red:9.1f}%")

def run_full_m2_end_to_end_benchmark(device: torch.device):
    print("\n=================== 3. FULL M2 MODEL END-TO-END TRAINING STEP BENCHMARK ===================")
    config = get_model_config('M2')
    input_ids = torch.randint(0, 100, (1, 128), device=device)
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # --- Part A: M2 Base ---
    m2_base = MoETransformer(
        vocab_size=100, d_model=1408, n_layers=5, num_heads=8,
        d_ff=config['d_ff'], moe=True, num_experts=32, top_k=2,
        use_sparse_dispatcher=False
    ).to(device=device)
    m2_base = MemoryManager.convert_precision_selective(m2_base, "bf16-storage")
    opt_base = create_optimizer(m2_base, opt_name="adam8bit", lr=1e-3)

    # Warmup
    for _ in range(3):
        opt_base.zero_grad()
        with amp_ctx:
            out, _ = m2_base(input_ids)
            loss = out.sum()
        loss.backward()
        opt_base.step()

    times_base = []
    for _ in range(10):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        opt_base.zero_grad()
        with amp_ctx:
            out, _ = m2_base(input_ids)
            loss = out.sum()
        loss.backward()
        opt_base.step()
        torch.cuda.synchronize(device)
        times_base.append((time.perf_counter() - t0) * 1000.0)

    # Clean VRAM before Part B
    del m2_base, opt_base
    torch.cuda.empty_cache()

    # --- Part B: M2 Sparse ---
    m2_sparse = MoETransformer(
        vocab_size=100, d_model=1408, n_layers=5, num_heads=8,
        d_ff=config['d_ff'], moe=True, num_experts=32, top_k=2,
        use_sparse_dispatcher=True
    ).to(device=device)
    m2_sparse = MemoryManager.convert_precision_selective(m2_sparse, "bf16-storage")
    opt_sparse = create_optimizer(m2_sparse, opt_name="adam8bit", lr=1e-3)

    # Warmup
    for _ in range(3):
        opt_sparse.zero_grad()
        with amp_ctx:
            out, _ = m2_sparse(input_ids)
            loss = out.sum()
        loss.backward()
        opt_sparse.step()

    times_sparse = []
    for _ in range(10):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        opt_sparse.zero_grad()
        with amp_ctx:
            out, _ = m2_sparse(input_ids)
            loss = out.sum()
        loss.backward()
        opt_sparse.step()
        torch.cuda.synchronize(device)
        times_sparse.append((time.perf_counter() - t0) * 1000.0)

    del m2_sparse, opt_sparse
    torch.cuda.empty_cache()

    base_mean = np.mean(times_base)
    base_std = np.std(times_base)
    sparse_mean = np.mean(times_sparse)
    sparse_std = np.std(times_sparse)

    e2e_speedup = (base_mean - sparse_mean) / base_mean * 100.0

    print(f"M2 Full Base Step Latency:   {base_mean:.2f} ms ± {base_std:.2f} ms (p50: {np.percentile(times_base, 50):.2f} ms, p95: {np.percentile(times_base, 95):.2f} ms)")
    print(f"M2 Full Sparse Step Latency: {sparse_mean:.2f} ms ± {sparse_std:.2f} ms (p50: {np.percentile(times_sparse, 50):.2f} ms, p95: {np.percentile(times_sparse, 95):.2f} ms)")
    print(f"End-to-End M2 Step Speedup:  {e2e_speedup:.2f}% reduction in training step latency!")
    print("=========================================================================================\n")

def main():
    if not torch.cuda.is_available():
        print("CUDA is required for this experiment.")
        sys.exit(1)
    device = torch.device("cuda")

    run_sequence_length_sweep(device)
    run_statistical_timing_benchmark(device)
    run_full_m2_end_to_end_benchmark(device)

if __name__ == "__main__":
    main()
