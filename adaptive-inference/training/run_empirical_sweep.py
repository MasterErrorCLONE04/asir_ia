"""
training/run_empirical_sweep.py — Orchestrates reproducible multi-seed sweeps.

Calls training/run_experiments.py in isolated subprocesses for a list of model
configurations and seeds.

Usage:
    python training/run_empirical_sweep.py --models M2,M3 --seeds 42,43 --epochs 5
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Orchestrates reproducible training loops and seeds.")
    parser.add_argument("--models", type=str, default="M2,M3", help="Comma-separated list of models to sweep.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated list of seeds (or range like 42-51).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--controlled", action="store_true", help="Enable Controlled MoE routing.")
    parser.add_argument("--aux_coef", type=float, default=0.01, help="Auxiliary load-balancing loss coefficient.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only.")
    
    args = parser.parse_args()

    # Parse models
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # Parse seeds
    seed_list = []
    for s_part in args.seeds.split(","):
        s_part = s_part.strip()
        if not s_part:
            continue
        if "-" in s_part:
            start, end = s_part.split("-")
            seed_list.extend(range(int(start), int(end) + 1))
        else:
            seed_list.append(int(s_part))

    print("=" * 80)
    print("EMPIRICAL RUN SWEEP ORCHESTRATOR")
    print(f"  Models: {model_list}")
    print(f"  Seeds:  {seed_list} (Total: {len(seed_list)})")
    print(f"  Epochs: {args.epochs}")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_script = os.path.join(script_dir, "run_experiments.py")

    total_runs = len(model_list) * len(seed_list)
    current_run = 0

    for model in model_list:
        for seed in seed_list:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running model={model} | seed={seed} | epochs={args.epochs}...")
            
            # Prepare arguments for subprocess call
            cmd = [
                sys.executable,
                experiment_script,
                "--model", model,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--aux_coef", str(args.aux_coef),
            ]
            if args.controlled:
                cmd.append("--controlled")
            if args.eval_only:
                cmd.append("--eval_only")

            # Run in isolated subprocess to prevent memory accumulation and crashes
            try:
                result = subprocess.run(cmd, check=True, text=True)
                print(f"[{model} | Seed {seed}] Completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: [{model} | Seed {seed}] Run failed with returncode {e.returncode}")
                # We do not crash the orchestrator, we continue with the sweep to collect other seeds
                continue

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
