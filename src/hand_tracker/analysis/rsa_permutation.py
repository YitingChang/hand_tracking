import os
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

# Paths
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SHAPE_RDM_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_RDM_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
PERCEPT_RDM_DIR = ANALYSIS_ROOT / "percept_analysis" 
RSA_DIR = ANALYSIS_ROOT / "rsa_comparison"
os.makedirs(RSA_DIR, exist_ok=True)

IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

HAND_PATH = HAND_RDM_DIR / f"hand_rdms_{TRIAL_TYPE}_{ori_str}.pkl"
ALEX_PATH = SHAPE_RDM_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl"
PERCEPT_PATH = PERCEPT_RDM_DIR / f"percept_rdms_for_hand_{TRIAL_TYPE}_{ori_str}.pkl"

# Load hand data to get the EXACT order of shape_ids
hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"

hand_feat_path = HAND_RDM_DIR / hand_feat_csv
df_hand = pd.read_csv(hand_feat_path)
valid_ids = df_hand['shape_id'].astype(str).str.strip().tolist() # Keep order

# --- FUNCTIONS ---
def load_rdm(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['rdm'] if isinstance(data, dict) else data

def get_upper_tri(matrix):
    """Extracts the upper triangle of an RDM and flattens it, ignoring NaNs."""
    # Ensure it's a square matrix
    if len(matrix.shape) == 1:
        return matrix
    mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[mask]

def partial_corr(r_xy, r_xz, r_yz):
    num = r_xy - (r_xz * r_yz)
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return num / den if den > 0 else 0

def get_p_value(actual_rho, null_distribution):
    """Calculates p-value based on how many null rhos are greater than actual."""
    return np.sum(null_distribution >= actual_rho) / len(null_distribution)

def run_hierarchical_rsa(n_permutations=1000):
    # Load Primary RDMs
    hand_rdm = load_rdm(HAND_PATH)
    percept_rdm = load_rdm(PERCEPT_PATH)
    with open(ALEX_PATH, 'rb') as f: alex_data = pickle.load(f)
    
    h_vec = get_upper_tri(hand_rdm)
    p_vec = get_upper_tri(percept_rdm)
    
    layers = ['low', 'mid', 'high']
    results = []

    # 1. Constant Baseline: Hand vs Perception
    rho_hp, _ = spearmanr(h_vec, p_vec, nan_policy='omit')

    for layer in layers:
        print(f"Analyzing Layer: {layer}...")
        a_vec = get_upper_tri(alex_data[layer]['rdm'])
        
        # Create mask for valid pairs across all three matrices
        mask = ~np.isnan(a_vec) & ~np.isnan(h_vec) & ~np.isnan(p_vec)
        v_a, v_h, v_p = a_vec[mask], h_vec[mask], p_vec[mask]

        # Actual RSA
        r_ah, _ = spearmanr(v_a, v_h)
        r_ap, _ = spearmanr(v_a, v_p)
        r_hp_layer, _ = spearmanr(v_h, v_p)
        p_ah = partial_corr(r_ah, r_ap, r_hp_layer)
        p_ap = partial_corr(r_ap, r_ah, r_hp_layer)

        # 2. Permutation Testing (Shuffle AlexNet RDM)
        null_ah = []
        for _ in range(n_permutations):
            # Shuffle the vector (shuffling the RDM labels is equivalent)
            shuffled_a = np.random.permutation(v_a)
            r_null, _ = spearmanr(shuffled_a, v_h)
            null_ah.append(r_null)
        
        pval = get_p_value(r_ah, np.array(null_ah))

        results.append({
            'Layer': layer,
            'Hand-Alex': r_ah,
            'Hand-Alex (Partial)': p_ah,
            'Percept-Alex': r_ap,
            'Percept-Alex (Partial)': p_ap,
            'p-value': pval
        })

    df_res = pd.DataFrame(results)
    plot_hierarchy_results(df_res, rho_hp)
    return df_res

def plot_hierarchy_results(df, rho_hp):
    plt.figure(figsize=(10, 6))
    layers = df['Layer'].values
    
    # Standard RSA Lines
    plt.plot(layers, df['Hand-Alex'], label='Hand-Alex', color='tab:blue', marker='o', linewidth=2.5)
    plt.plot(layers, df['Percept-Alex'], label='Percept-Alex', color='tab:orange', marker='o', linewidth=2.5)
    
    # Partial RSA Lines (Dashed)
    plt.plot(layers, df['Hand-Alex (Partial)'], label='Hand-Alex (Partial)', color='tab:blue', linestyle='--', alpha=0.6)
    plt.plot(layers, df['Percept-Alex (Partial)'], label='Percept-Alex (Partial)', color='tab:orange', linestyle='--', alpha=0.6)

    # Significance Stars
    for i, row in df.iterrows():
        if row['p-value'] < 0.001:
            plt.text(i, row['Hand-Alex'] + 0.01, '***', ha='center', color='black', fontsize=12)
        elif row['p-value'] < 0.01:
            plt.text(i, row['Hand-Alex'] + 0.01, '**', ha='center', color='black', fontsize=12)
        elif row['p-value'] < 0.05:
            plt.text(i, row['Hand-Alex'] + 0.01, '*', ha='center', color='black', fontsize=12)

    plt.title("Functional Specialization Across AlexNet Hierarchy", fontsize=14)
    plt.ylabel("Spearman Rho (ρ)", fontsize=12)
    plt.ylim(0, max(df['Hand-Alex'].max(), df['Percept-Alex'].max()) + 0.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RSA_DIR / "hierarchy_significance_plot.png")

if __name__ == "__main__":
    df = run_hierarchical_rsa(n_permutations=1000)

    save_path = RSA_DIR / "rsa_permutation.csv"
    df.to_csv(save_path, index=False)