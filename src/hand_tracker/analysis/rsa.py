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
CONTACT_RDM_SAVE_DIR = ANALYSIS_ROOT / "contact_analysis" / "contact_rdms" 
RSA_SAVE_DIR = ANALYSIS_ROOT / "rsa_comparison"
os.makedirs(RSA_SAVE_DIR, exist_ok=True)

ALEXNET_LAYER = 'high'  # Options: 'low', 'mid', 'high'
IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

HAND_PATH = HAND_RDM_SAVE_DIR / f"hand_rdms_{TRIAL_TYPE}_{ori_str}.pkl"
ALEX_PATH = SHAPE_RDM_SAVE_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl"
PERCEPT_PATH = PERCEPT_RDM_SAVE_DIR / f"percept_rdms_for_hand_{TRIAL_TYPE}_{ori_str}.pkl"
CONTACT_PATH = CONTACT_RDM_SAVE_DIR / f"contact_rdms_{TRIAL_TYPE}_{ori_str}.pkl" 

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
    if len(matrix.shape) == 1:
        return matrix
    mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[mask]

def compute_precision_partial_corrs(corr_matrix):
    """
    Computes a clean matrix of partial correlations for any number of 
    modalities using the inverse matrix (Precision Matrix) trick.
    """
    # Inverse of the standard correlation matrix
    precision = np.linalg.inv(corr_matrix)
    diag = np.diag(precision)
    
    # Calculate partial correlation matrix elements: -P_ij / sqrt(P_ii * P_jj)
    partial_corrs = -precision / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(partial_corrs, 1.0)
    return partial_corrs

def plot_mds_comparison(rdm_list, titles, shape_ids, save_dir=None):
    fig, axes = plt.subplots(1, len(rdm_list), figsize=(24, 6)) 
    colors = sns.color_palette("husl", len(shape_ids))

    for i, rdm in enumerate(rdm_list):
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        clean_rdm = np.nan_to_num(rdm, nan=np.nanmean(rdm))
        coords = mds.fit_transform(clean_rdm)
        
        axes[i].scatter(coords[:, 0], coords[:, 1], c=colors, edgecolors='k', alpha=0.7)
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].axis('off')

    plt.suptitle("Representational Geometry Comparison (4 Modalities)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_dir:
        plt.savefig(save_dir / f"mds_alex-{ALEXNET_LAYER}_hand_percept_contact_{TRIAL_TYPE}_{ori_str}.png")

def main():
    # Load RDMs
    hand_rdm = load_rdm(HAND_PATH)
    percept_rdm = load_rdm(PERCEPT_PATH)
    contact_rdm = load_rdm(CONTACT_PATH) 
    
    with open(ALEX_PATH, 'rb') as f: 
        alex_data = pickle.load(f)
    alex_rdm = alex_data[ALEXNET_LAYER]['rdm']

    # --- PREPARE DATA ---
    hand_vec = get_upper_tri(hand_rdm)
    percept_vec = get_upper_tri(percept_rdm)
    alex_vec = get_upper_tri(alex_rdm)
    contact_vec = get_upper_tri(contact_rdm) 

    # Handle NaNs dynamically across all 4 vector spaces simultaneously
    valid_mask = (~np.isnan(alex_vec) & 
                  ~np.isnan(hand_vec) & 
                  ~np.isnan(percept_vec) & 
                  ~np.isnan(contact_vec))
    
    v_alex = alex_vec[valid_mask]
    v_hand = hand_vec[valid_mask]
    v_percept = percept_vec[valid_mask]
    v_contact = contact_vec[valid_mask]

    # --- COMPUTE STANDARD RSA ---
    # Stack columns to compute a clean cross-correlation profile matrix
    data_matrix = np.column_stack([v_alex, v_hand, v_percept, v_contact])
    labels = [f"AlexNet ({ALEXNET_LAYER})", "Hand Conformation", "Perceptual Choice", "Contact Profiles"]
    
    # Generate the standard Spearman rank correlation matrix
    rho_matrix, _ = spearmanr(data_matrix)

    print("Standard RSA Matrix (Spearman Rho):")
    for i in range(4):
        for j in range(i+1, 4):
            print(f"  {labels[i]} <-> {labels[j]}: {rho_matrix[i, j]:.3f}")

    # --- COMPUTE PARTIAL RSA ---
    # Compute the scaled partial correlations controlling for ALL other alternatives
    partial_matrix = compute_precision_partial_corrs(rho_matrix)

    print("\nPartial RSA Matrix (Controlling for remaining elements):")
    print(f"  AlexNet <-> Hand       (cond.): {partial_matrix[0, 1]:.3f}")
    print(f"  AlexNet <-> Perception (cond.): {partial_matrix[0, 2]:.3f}")
    print(f"  AlexNet <-> Contact    (cond.): {partial_matrix[0, 3]:.3f}")
    print(f"  Hand    <-> Perception (cond.): {partial_matrix[1, 2]:.3f}")
    print(f"  Hand    <-> Contact    (cond.): {partial_matrix[1, 3]:.3f}")
    print(f"  Percept <-> Contact    (cond.): {partial_matrix[2, 3]:.3f}")

    # Plot MDS expanded frame comparison grid
    plot_mds_comparison(
        [alex_rdm, hand_rdm, percept_rdm, contact_rdm], 
        labels,
        valid_ids,
        save_dir=RSA_SAVE_DIR
    )

if __name__ == "__main__":
    main()