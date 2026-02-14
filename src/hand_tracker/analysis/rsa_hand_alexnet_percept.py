import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import is_valid_y
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import seaborn as sns

# --- CONFIGURATION ---

# Paths
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
PERCEPT_RDM_SAVE_DIR = ANALYSIS_ROOT / "percept_analysis" 
RSA_SAVE_DIR = ANALYSIS_ROOT / "rsa_comparison"
os.makedirs(RSA_SAVE_DIR, exist_ok=True)

ALEXNET_LAYER = 'mid'  # Options: 'low', 'mid', 'high'
IMAGE_TYPE = 'depth'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

HAND_PATH = HAND_RDM_SAVE_DIR / f"hand_rdms_{TRIAL_TYPE}_{ori_str}.pkl"
ALEX_PATH = SHAPE_RDM_SAVE_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl"
PERCEPT_PATH = PERCEPT_RDM_SAVE_DIR / f"percept_rdms_for_hand_{TRIAL_TYPE}_{ori_str}.pkl"

# Load hand data to get the EXACT order of shape_ids
hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"

hand_feat_path = HAND_RDM_SAVE_DIR / hand_feat_csv
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

def plot_mds_comparison(rdm_list, titles, shape_ids, save_dir=None):
    fig, axes = plt.subplots(1, len(rdm_list), figsize=(18, 6))
    
    # Use a consistent color palette for shapes
    colors = sns.color_palette("husl", len(shape_ids))

    for i, rdm in enumerate(rdm_list):
        # Initialize MDS: precomputed means we provide a distance matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        
        # Handle NaNs by filling with mean distance (simple fix for visualization)
        clean_rdm = np.nan_to_num(rdm, nan=np.nanmean(rdm))
        coords = mds.fit_transform(clean_rdm)
        
        axes[i].scatter(coords[:, 0], coords[:, 1], c=colors, edgecolors='k', alpha=0.7)
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].axis('off')

    plt.suptitle("Representational Geometry: AlexNet vs. Hand vs. Perception", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_dir:
        plt.savefig(save_dir / f"mds_alex-{ALEXNET_LAYER}_hand_percept_{TRIAL_TYPE}_{ori_str}.png")

def main():
    # Load RDMs
    hand_rdm = load_rdm(HAND_PATH)
    percept_rdm = load_rdm(PERCEPT_PATH)
    with open(ALEX_PATH, 'rb') as f: alex_data = pickle.load(f)
    # We use the 'mid' layer as it had the highest Rho
    alex_rdm = alex_data[ALEXNET_LAYER]['rdm']

    # --- PREPARE DATA ---
    hand_vec = get_upper_tri(hand_rdm)
    percept_vec = get_upper_tri(percept_rdm)
    alex_vec = get_upper_tri(alex_rdm)

    # Handle NaNs (important if some pairs were never visited by the monkey)
    valid_mask = ~np.isnan(alex_vec) & ~np.isnan(hand_vec) & ~np.isnan(percept_vec)
    v_alex = alex_vec[valid_mask]
    v_hand = hand_vec[valid_mask]
    v_percept = percept_vec[valid_mask]

    # --- COMPUTE STANDARD RSA ---
    rho_ah, _ = spearmanr(v_alex, v_hand)
    rho_ap, _ = spearmanr(v_alex, v_percept)
    rho_hp, _ = spearmanr(v_hand, v_percept)

    print(f"Standard RSA (Spearman Rho):")
    print(f"  AlexNet ({ALEXNET_LAYER}) <-> Hand: {rho_ah:.3f}")
    print(f"  AlexNet ({ALEXNET_LAYER}) <-> Perception: {rho_ap:.3f}")
    print(f"  Hand <-> Perception: {rho_hp:.3f}")

    # --- COMPUTE PARTIAL CORRELATION ---
    rho_ah_partial_p = partial_corr(rho_ah, rho_ap, rho_hp)
    rho_ap_partial_h = partial_corr(rho_ap, rho_ah, rho_hp)
    rho_hp_partial_a = partial_corr(rho_hp, rho_ah, rho_ap)

    print(f"\nPartial RSA Results:")
    print(f"  AlexNet <-> Hand (controlling for Perception): {rho_ah_partial_p:.3f}")
    print(f"  AlexNet <-> Perception (controlling for Hand): {rho_ap_partial_h:.3f}")
    print(f"  Hand <-> Perception (controlling for AlexNet): {rho_hp_partial_a:.3f}")

    # Plot MDS comparison of the three RDMs
    plot_mds_comparison(
        [alex_rdm, hand_rdm, percept_rdm], 
        [f"AlexNet ({ALEXNET_LAYER})", "Hand Conformation", "Perceptual Choice"],
        valid_ids,
        save_dir=RSA_SAVE_DIR
    )


if __name__ == "__main__":
    main()
