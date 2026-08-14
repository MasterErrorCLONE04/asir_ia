"""
experiments/measure_prefetch_overhead.py — Profiling I/O Disk Latency, CUDA Streams, and Async Prefetching
"""

import os
import sys
import time
import json
import torch
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 80)
    print("ASIR MICRO-BENCHMARK: DISK I/O LATENCY & ASYNC PREFETCH OVERHEAD")
    print("=" * 80)

    # 1. Locate existing expert weights file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expert_path = os.path.join(script_dir, "..", "results", "TR9", "nvme_store", "layer_0_expert_0.pt")
    
    if not os.path.exists(expert_path):
        print(f"Error: Expert weights file not found at {expert_path}.")
        print("Please run benchmark_local_inference.py once first to generate weights.")
        sys.exit(1)
        
    size_mb = os.path.getsize(expert_path) / (1024 * 1024)
    print(f"Target Expert File: {os.path.basename(expert_path)}")
    print(f"File Size: {size_mb:.2f} MB")

    # 2. Test A: Single Threaded Synchronous torch.load (CPU)
    print("\n[Test A] Synchronous Disk Read (torch.load)...", flush=True)
    latencies_sync = []
    for i in range(5):
        start = time.perf_counter()
        sd = torch.load(expert_path, map_location='cpu')
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies_sync.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms", flush=True)
        
    avg_sync = sum(latencies_sync) / len(latencies_sync)
    print(f"  Average: {avg_sync:.2f} ms", flush=True)

    # 3. Test B: torch.load inside torch.cuda.stream (Simulating current engine block)
    print("\n[Test B] Disk Read wrapped in torch.cuda.stream...", flush=True)
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        latencies_stream = []
        for i in range(5):
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                sd = torch.load(expert_path, map_location='cpu')
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies_stream.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f} ms", flush=True)
        avg_stream = sum(latencies_stream) / len(latencies_stream)
        print(f"  Average: {avg_stream:.2f} ms", flush=True)
    else:
        print("  CUDA not available, skipping Stream test. (Identical to Test A on CPU)", flush=True)
        avg_stream = avg_sync

    # 4. Test C: Asynchronous Prefetching using ThreadPoolExecutor
    print("\n[Test C] Asynchronous Prefetch (ThreadPoolExecutor)...", flush=True)
    executor = ThreadPoolExecutor(max_workers=1)
    
    # We will simulate a loop where the main thread does fake compute (e.g. sleep/matrix mult)
    # while the background thread fetches the expert.
    # We measure overlap and total elapsed time.
    def bg_load(path):
        return torch.load(path, map_location='cpu')

    # Case C1: Synchronous loading + Compute
    print("  Case C1: Synchronous Load + Compute (No Overlap)...", flush=True)
    start = time.perf_counter()
    # Load (blocking)
    sd = torch.load(expert_path, map_location='cpu')
    # Compute (simulated 50ms FFN pass)
    time.sleep(0.05)
    elapsed_sync = (time.perf_counter() - start) * 1000.0
    print(f"    Total Time (Load + Compute): {elapsed_sync:.2f} ms", flush=True)

    # Case C2: Asynchronous Load overlapping with Compute
    print("  Case C2: Asynchronous Load + Compute (Overlapped)...", flush=True)
    start = time.perf_counter()
    # Submit async load
    future = executor.submit(bg_load, expert_path)
    # Compute in main thread immediately (simulated 50ms FFN pass)
    time.sleep(0.05)
    # Join / get result
    sd = future.result()
    elapsed_async = (time.perf_counter() - start) * 1000.0
    print(f"    Total Time (Overlapped): {elapsed_async:.2f} ms", flush=True)
    
    speedup = ((elapsed_sync - elapsed_async) / elapsed_sync) * 100.0
    print(f"  Overlap Savings / Speedup: {speedup:.2f}%", flush=True)

    # 5. Save results to JSON
    output_dir = os.path.join(script_dir, "..", "results", "TR9")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "prefetch_latency_decomposition.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'expert_file_size_mb': size_mb,
            'avg_sync_read_ms': avg_sync,
            'avg_stream_read_ms': avg_stream,
            'case_c1_blocking_total_ms': elapsed_sync,
            'case_c2_overlapped_total_ms': elapsed_async,
            'savings_pct': speedup
        }, f, indent=2)
        
    print(f"\nMeasurement results saved to: {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
