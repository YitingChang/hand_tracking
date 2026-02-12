import pickle
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Paths
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
SAVE_DIR = ANALYSIS_ROOT / "hand_shape_comparison"
RDM_FIG_DIR = SAVE_DIR / "rdm_figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RDM_FIG_DIR, exist_ok=True)

IMAGE_TYPE = 'depth'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 

ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
HAND_PATH = HAND_RDM_SAVE_DIR / f"hand_rdms_{TRIAL_TYPE}_{ori_str}.pkl"
ALEX_PATH = SHAPE_RDM_SAVE_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl"

def plot_rdm(rdm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(rdm, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    with open(HAND_PATH, 'rb') as f: hand_data = pickle.load(f)
    with open(ALEX_PATH, 'rb') as f: alex_data = pickle.load(f)

    hand_rdm = hand_data['rdm']
    tri_idx = np.triu_indices(hand_rdm.shape[0], k=1)
    hand_vec = hand_rdm[tri_idx]

    results = {}
    for label, data in alex_data.items():
        alex_vec = data['rdm'][tri_idx]
        rho, p = spearmanr(hand_vec, alex_vec)
        results[label] = rho
        print(f"Correlation with {label} ({data['layer']}): Rho = {rho:.3f}, p = {p:.4e}")

        # Plot the scatter with regression line
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(hand_vec, alex_vec, s=2, alpha=0.5)

        # Calculate a linear fit: y = mx + b
        m, b = np.polyfit(hand_vec, alex_vec, 1)
        # Generate points for the line based on the range of the x-axis
        line_x = np.array([hand_vec.min(), hand_vec.max()])
        ax.plot(line_x, m * line_x + b, color='red', linewidth=1, label='Linear Fit')

        # Formatting
        ax.set_xlabel("Hand Conformation RDM")
        ax.set_ylabel(f"AlexNet {label} RDM")
        # Using LaTeX for scientific notation in title
        ax.set_title(f"Spearman $\\rho$: {rho:.3f}, $p$-value: {p:.4e}")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(SAVE_DIR, f"correlation_hand_alexnet_{label}_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png")
        plt.savefig(save_path)
        plt.close()

    # Plot the RDMs for visual inspection
    plot_rdm(hand_rdm, "Hand Conformation RDM", RDM_FIG_DIR / f"hand_rdm_{TRIAL_TYPE}_{ori_str}.png")
    for label, data in alex_data.items():
        plot_rdm(data['rdm'], f"AlexNet {label} RDM", RDM_FIG_DIR / f"alexnet_{label}_rdm_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png")

    # Plotting the Hierarchy
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color='teal')
    plt.ylabel("Spearman Correlation (Rho)")
    plt.title("Hand Conformation vs AlexNet Hierarchy")
    plt.ylim(0, max(results.values()) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(SAVE_DIR, f"hand_alexnet_correlation_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.png"))

if __name__ == "__main__":
    main()