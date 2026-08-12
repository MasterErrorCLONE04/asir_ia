import os
import json
import argparse
from generator import SyntheticTaskGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training and evaluation datasets.")
    parser.add_argument("--num_train", type=int, default=10000, help="Number of training samples.")
    parser.add_argument("--num_val", type=int, default=1000, help="Number of validation samples.")
    parser.add_argument("--num_test", type=int, default=1000, help="Number of test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory for output datasets.")
    
    args = parser.parse_args()

    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir if args.output_dir else script_dir

    train_dir = os.path.join(output_dir, "train_set")
    eval_dir = os.path.join(output_dir, "eval_set")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Initializing SyntheticTaskGenerator with seed {args.seed}...")
    generator = SyntheticTaskGenerator(seed=args.seed)

    splits = [
        ("train", args.num_train, os.path.join(train_dir, "train.jsonl")),
        ("val", args.num_val, os.path.join(eval_dir, "val.jsonl")),
        ("test", args.num_test, os.path.join(eval_dir, "test.jsonl"))
    ]

    for name, count, filepath in splits:
        print(f"Generating {count} samples for '{name}' split...")
        dataset = generator.generate_dataset(count)
        
        # Write to JSONL
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        
        # Print actual distribution generated
        domain_counts = {}
        for item in dataset:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
        print(f"Saved to {filepath}")
        print("Distribution:")
        for domain, d_count in domain_counts.items():
            percentage = (d_count / len(dataset)) * 100
            print(f"  - {domain}: {d_count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
