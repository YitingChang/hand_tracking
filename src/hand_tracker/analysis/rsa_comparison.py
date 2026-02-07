import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Paths
HAND_PATH = "/media/yiting/NewVolume/Analysis/hand_analysis/hand_rdm_correct_aligned.pkl"
ALEX_PATH = "/media/yiting/NewVolume/Analysis/shape_analysis/alexnet_rdms_aligned.pkl"

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
    plt.show()

if __name__ == "__main__":
    main()