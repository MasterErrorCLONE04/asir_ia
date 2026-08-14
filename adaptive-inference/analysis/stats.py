"""
analysis/stats.py — Pairwise statistical contrast calculator (§11 spec v0.7)

Aggregates experiment JSON files, matches them by seed, and calculates:
  - mean(ΔQ) and median(ΔQ)
  - 95% Confidence Interval for ΔQ (Student-t)
  - p-value of Wilcoxon signed-rank test
  - Cohen's d (paired)
  - Mean differences of routing mechanics: ΔN_eff, Δη_cap, ΔSD, Δunassigned_mass

Usage:
    python analysis/stats.py --baseline M2 --model M3
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import scipy.stats as stats

# Config order for matching
ROLES = ['arithmetic', 'logic', 'language']

def get_config_hash(model_name: str, args) -> str:
    """
    Computes config hash matching training hyperparameters.
    """
    # Import train configuration helpers
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.train import get_model_config
    cfg = get_model_config(model_name)
    
    config_dict = {
        'model':       model_name,
        'moe':         cfg['moe'],
        'num_experts': cfg['num_experts'],
        'top_k':       cfg['top_k'],
        'd_ff':        cfg['d_ff'],
        'lr':          args.lr,
        'batch_size':  args.batch_size,
        'aux_coef':    args.aux_coef,
        'controlled':  args.controlled,
        'epochs':      args.epochs,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]


def load_results(model_name: str, hash_val: str, results_dir: str) -> dict:
    """
    Loads all seed files for a given config hash from results/R1.0/<hash>/.
    Returns dict mapping seed -> results_dict.
    """
    path = os.path.join(results_dir, "R1.0", hash_val)
    if not os.path.exists(path):
        print(f"WARNING: Results path does not exist: {path}")
        return {}
    
    seeds_data = {}
    for filename in os.listdir(path):
        if filename.startswith("seed_") and filename.endswith("_results.json"):
            # seed_42_results.json
            parts = filename.split("_")
            try:
                seed = int(parts[1])
            except ValueError:
                continue
            
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                seeds_data[seed] = json.load(f)
                
    return seeds_data


def calculate_contrasts(base_data: dict, model_data: dict):
    # Match seeds
    common_seeds = sorted(list(set(base_data.keys()).intersection(set(model_data.keys()))))
    
    if not common_seeds:
        print("ERROR: No common seeds found between baseline and model configurations.")
        sys.exit(1)
        
    N = len(common_seeds)
    print(f"Found {N} common seeds: {common_seeds}")

    # Metrics collections
    q_base = []
    q_model = []
    n_eff_base = []
    n_eff_model = []
    eta_cap_base = []
    eta_cap_model = []
    
    # Layer 0 metrics for specialisation sensitivity
    sd_base = []
    sd_model = []
    un_mass_base = []
    un_mass_model = []

    for seed in common_seeds:
        b = base_data[seed]
        m = model_data[seed]
        
        q_base.append(b['metrics']['quality_q'])
        q_model.append(m['metrics']['quality_q'])
        
        n_eff_base.append(b['metrics']['avg_n_eff'])
        n_eff_model.append(m['metrics']['avg_n_eff'])
        
        eta_cap_base.append(b['metrics']['avg_eta_cap'])
        eta_cap_model.append(m['metrics']['avg_eta_cap'])

        # Fetch layer_0 (first MoE layer) spec metrics under primary tau=0.50 if MoE
        if b['hyperparameters']['moe']:
            l0_b = b['metrics']['layers']['layer_0']['sensitivity']['tau_0.50']
            sd_base.append(l0_b['specialization_density'])
            un_mass_base.append(l0_b['unassigned_routing_mass'])
        else:
            sd_base.append(0.0)
            un_mass_base.append(0.0)
            
        if m['hyperparameters']['moe']:
            l0_m = m['metrics']['layers']['layer_0']['sensitivity']['tau_0.50']
            sd_model.append(l0_m['specialization_density'])
            un_mass_model.append(l0_m['unassigned_routing_mass'])
        else:
            sd_model.append(0.0)
            un_mass_model.append(0.0)

    q_base = np.array(q_base)
    q_model = np.array(q_model)
    delta_q = q_model - q_base
    
    n_eff_base = np.array(n_eff_base)
    n_eff_model = np.array(n_eff_model)
    delta_n_eff = n_eff_model - n_eff_base

    eta_cap_base = np.array(eta_cap_base)
    eta_cap_model = np.array(eta_cap_model)
    delta_eta_cap = eta_cap_model - eta_cap_base

    # Quality Q statistics
    mean_diff = np.mean(delta_q)
    median_diff = np.median(delta_q)
    
    # 95% Confidence Interval (Student-t)
    if N > 1:
        sem = stats.sem(delta_q)
        ci_lower, ci_upper = stats.t.interval(0.95, df=N-1, loc=mean_diff, scale=sem)
        
        # Wilcoxon signed-rank (p-value)
        # Handle case where all differences are zero (causes Wilcoxon to raise ValueError)
        if np.all(delta_q == 0):
            p_wilcoxon = 1.0
        else:
            _, p_wilcoxon = stats.wilcoxon(q_model, q_base)
            
        # Paired Cohen's d
        std_diff = np.std(delta_q, ddof=1)
        cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
    else:
        ci_lower, ci_upper = mean_diff, mean_diff
        p_wilcoxon = 1.0
        cohen_d = 0.0

    # Report contrast
    print("\n" + "=" * 80)
    print("STATISTICAL CONTRAST REPORT (§11 v0.7)")
    print("=" * 80)
    print(f"  Contraste: {base_data[common_seeds[0]]['model_name']} ──> {model_data[common_seeds[0]]['model_name']}")
    print(f"  Semillas:  {N} (Seeds: {common_seeds})")
    print("-" * 80)
    print(f"  Métrica Primaria: Calidad Q (Exact Match)")
    print(f"    Baseline Q (mean): {np.mean(q_base):.2f}%")
    print(f"    Model Q (mean):    {np.mean(q_model):.2f}%")
    print(f"    Media de ΔQ:       {mean_diff:+.2f} pp")
    print(f"    Mediana de ΔQ:     {median_diff:+.2f} pp")
    print(f"    IC 95% (ΔQ):       [{ci_lower:+.4f}, {ci_upper:+.4f}] pp")
    print(f"    Cohen's d:         {cohen_d:.4f}")
    print(f"    p-Wilcoxon:        {p_wilcoxon:.6f}")
    
    # Check Acceptance Criteria
    rej_null = p_wilcoxon < 0.05
    excl_zero = not (ci_lower <= 0.0 <= ci_upper)
    pract_rel = abs(mean_diff) >= 2.0
    
    print("\n  Criterio de Aceptación (§11 v0.7):")
    print(f"    [1] IC 95% excluye 0?                     {excl_zero}   (Intervalo: [{ci_lower:+.2f}, {ci_upper:+.2f}])")
    print(f"    [2] p_Wilcoxon < 0.05?                    {rej_null}   (p-valor: {p_wilcoxon:.4f})")
    print(f"    [3] Relevancia práctica |ΔQ| >= 2.0 pp?   {pract_rel}   (Magnitud: {abs(mean_diff):.2f} pp)")
    
    satisfied = excl_zero and rej_null and pract_rel
    status = "EVIDENCIA ESTADÍSTICA Y PRÁCTICAMENTE RELEVANTE (Nivel 3)" if satisfied else \
             "EVIDENCIA NOMINAL SOLAMENTE (Nivel 2)" if (mean_diff > 0 and np.mean(delta_n_eff) > 0) else \
             "EVIDENCIA DE CAPACIDAD SOLAMENTE (Nivel 1)" if (np.mean(delta_n_eff) > 0) else "SIN RESPALDO EXPERIMENTAL"
             
    print(f"    ESTADO DE EVIDENCIA: {status}")
    print("-" * 80)
    print(f"  Mecánicas Explicativas (Diferencias promedio):")
    print(f"    Baseline N_eff:   {np.mean(n_eff_base):.2f}  | Model N_eff:   {np.mean(n_eff_model):.2f}  | Δ: {np.mean(delta_n_eff):+.2f}")
    print(f"    Baseline η_cap:   {np.mean(eta_cap_base):.4f}  | Model η_cap:   {np.mean(eta_cap_model):.4f}  | Δ: {np.mean(delta_eta_cap):+.4f}")
    if sd_base and sd_model:
        delta_sd = np.array(sd_model) - np.array(sd_base)
        delta_un = np.array(un_mass_model) - np.array(un_mass_base)
        print(f"    Baseline SD (L0): {np.mean(sd_base):.3f}  | Model SD (L0): {np.mean(sd_model):.3f}  | Δ: {np.mean(delta_sd):+.3f}")
        print(f"    Baseline Un (L0): {np.mean(un_mass_base):.3f}  | Model Un (L0): {np.mean(un_mass_model):.3f}  | Δ: {np.mean(delta_un):+.3f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Computes R1.0 statistical contrasts pareados por seed.")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model configuration name (e.g., M2).")
    parser.add_argument("--model", type=str, required=True, help="Model configuration name (e.g., M3).")
    
    # Sweep hyperparams to match hash
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--controlled", action="store_true", help="Controlled MoE routing enabled.")
    parser.add_argument("--aux_coef", type=float, default=0.01, help="Aux loss coef.")
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    # Compute hashes
    base_hash = get_config_hash(args.baseline, args)
    model_hash = get_config_hash(args.model, args)
    
    print(f"Matching baseline hash '{base_hash}' vs model hash '{model_hash}'...")

    # Load results
    base_results = load_results(args.baseline, base_hash, results_dir)
    model_results = load_results(args.model, model_hash, results_dir)
    
    calculate_contrasts(base_results, model_results)


if __name__ == "__main__":
    main()
