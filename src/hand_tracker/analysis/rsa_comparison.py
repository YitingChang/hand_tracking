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
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION = ['02', '0', '2'] 

HAND_PATH = HAND_RDM_SAVE_DIR / f"hand_rdms_{TRIAL_TYPE}_ori{ORIENTATION[0]}.pkl"
ALEX_PATH = SHAPE_RDM_SAVE_DIR / f"alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_ori{ORIENTATION[0]}.pkl"


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

    # Plotting the Hierarchy
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color='teal')
    plt.ylabel("Spearman Correlation (Rho)")
    plt.title("Hand Conformation vs AlexNet Hierarchy")
    plt.ylim(0, max(results.values()) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(SAVE_DIR, f"hand_alexnet_correlation_{IMAGE_TYPE}_{TRIAL_TYPE}_ori{ORIENTATION[0]}.png"))

if __name__ == "__main__":
    main()