from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
RSA_SAVE_DIR = ANALYSIS_ROOT / "rsa_comparison"

# --- DATA ENTRY ---
# Using the values you provided
data = {
    'Layer': ['Low', 'Mid', 'High'],
    'Hand_Standard': [0.253, 0.475, 0.450],
    'Hand_Partial': [0.250, 0.451, 0.411],
    'Percept_Standard': [0.039, 0.218, 0.360],
    'Percept_Partial': [-0.014, 0.139, 0.305]
}

df = pd.DataFrame(data)

# --- PLOTTING ---
plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")

# Plot Hand Correlations
plt.plot(df['Layer'], df['Hand_Standard'], marker='o', linestyle='-', color='teal', 
         linewidth=2.5, label='Hand (Standard RSA)')
plt.plot(df['Layer'], df['Hand_Partial'], marker='o', linestyle='--', color='teal', 
         linewidth=1.5, alpha=0.7, label='Hand (Partial - control Percept)')

# Plot Perception Correlations
plt.plot(df['Layer'], df['Percept_Standard'], marker='s', linestyle='-', color='crimson', 
         linewidth=2.5, label='Perception (Standard RSA)')
plt.plot(df['Layer'], df['Percept_Partial'], marker='s', linestyle='--', color='crimson', 
         linewidth=1.5, alpha=0.7, label='Perception (Partial - control Hand)')

# Formatting
plt.title('Representational Hierarchy: Hand Conformation vs. Perception', fontsize=16, pad=20)
plt.ylabel('Spearman Correlation (Rho)', fontsize=14)
plt.xlabel('AlexNet Hierarchy', fontsize=14)
plt.ylim(-0.1, 0.6) # Standardizing Y-axis to see the differences clearly
plt.legend(frameon=True, fontsize=11, loc='upper left')

# Annotate the Peaks
# plt.annotate(f'Peak Hand: {df.iloc[1]["Hand_Standard"]}', xy=('Mid', 0.475), xytext=('Mid', 0.52),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), ha='center')

# plt.annotate(f'Peak Percept: {df.iloc[2]["Percept_Standard"]}', xy=('High', 0.360), xytext=('High', 0.28),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), ha='center')

plt.tight_layout()
plt.savefig(RSA_SAVE_DIR / "representational_hierarchy_comparison.png")