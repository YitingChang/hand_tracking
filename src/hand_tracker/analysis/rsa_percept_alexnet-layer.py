import pickle
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Paths
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
PERCEPT_RDM_SAVE_DIR = ANALYSIS_ROOT / "percept_analysis" 
SAVE_DIR = ANALYSIS_ROOT / "rsa_comparison" / "percept_alexnet-layer"
os.makedirs(SAVE_DIR, exist_ok=True)
RDM_FIG_DIR = SAVE_DIR / "rdm_figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RDM_FIG_DIR, exist_ok=True)

IMAGE_TYPE = 'depth'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 

ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
PERCEPT_PATH = PERCEPT_RDM_SAVE_DIR / f"percept_rdms_for_hand_{TRIAL_TYPE}_{ori_str}.pkl"
ALEX_PATH = SHAPE_RDM_SAVE_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl"

# --- FUNCTIONS ---
def get_upper_tri(matrix):
    """Extracts the upper triangle of an RDM and flattens it, ignoring NaNs."""
    # Ensure it's a square matrix
    if len(matrix.shape) == 1:
        return matrix
    mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[mask]

def plot_rdm(rdm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(rdm, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    with open(PERCEPT_PATH, 'rb') as f: percept_data = pickle.load(f)
    with open(ALEX_PATH, 'rb') as f: alex_data = pickle.load(f)

    percept_rdm = percept_data['rdm']
    percept_vec = get_upper_tri(percept_rdm)

    results = {}
    for label, data in alex_data.items():

        alex_vec = get_upper_tri(data['rdm'])

        # Handle NaNs (important if some pairs were never visited by the monkey)
        valid_mask = ~np.isnan(alex_vec) & ~np.isnan(percept_vec)
        v_alex = alex_vec[valid_mask]
        v_percept = percept_vec[valid_mask]

        rho, p = spearmanr(v_percept, v_alex)
        results[label] = rho
        print(f"Correlation with {label} ({data['layer']}): Rho = {rho:.3f}, p = {p:.4e}")

        # Scatter plot of the RDM vectors 
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(v_percept, v_alex, s=2, alpha=0.5)

        # Formatting
        ax.set_xlabel("Perception RDM")
        ax.set_ylabel(f"AlexNet {label} RDM")
        # Using LaTeX for scientific notation in title
        ax.set_title(f"Spearman $\\rho$: {rho:.3f}, $p$-value: {p:.4e}")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(SAVE_DIR, f"correlation_percept_alexnet_{label}_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png")
        plt.savefig(save_path)
        plt.close()

    # Plot the RDMs for visual inspection
    plot_rdm(percept_rdm, "Perception RDM", RDM_FIG_DIR / f"percept_rdms_{TRIAL_TYPE}_{ori_str}.png")
    for label, data in alex_data.items():
        plot_rdm(data['rdm'], f"AlexNet {label} RDM", RDM_FIG_DIR / f"alexnet_{label}_rdms_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png")

    # Plotting the Hierarchy
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color='teal')
    plt.ylabel("Spearman Correlation (Rho)")
    plt.title("Perception vs AlexNet Hierarchy")
    plt.ylim(0, max(results.values()) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(SAVE_DIR, f"percept_alexnet_correlation_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png"))

if __name__ == "__main__":
    main()