import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration ---
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
HAND_RDM_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
SHAPE_RDM_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_TSNE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_tsne"
ALEXNET_TSNE_DIR = ANALYSIS_ROOT / "shape_analysis" / "alexnet_tsne"
os.makedirs(HAND_TSNE_DIR, exist_ok=True)

# Condition setting
ALEXNET_LAYER = 'mid'  # Options: 'low', 'mid', 'high'
IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
FRAME_NUMBER = 300

alexnet_tsne_path = ALEXNET_TSNE_DIR / f"alexnet_{ALEXNET_LAYER}_tsne_{IMAGE_TYPE}_{ori_str}.csv"
# Model parameters
TSNE_PERPLEXITY = 30

# --- Functions ---

def compute_tsne_pca(df, feature_names):

    # Extract shape_type and orientation safely
    df['shape_type'] = df['shape_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    df['orientation'] = df['shape_id'].apply(lambda x: x.split('_')[1] if '_' in x else '0')

    # Prepare features for t-SNE using explicit column names
    df_features = df[feature_names]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42,
                learning_rate='auto', init='random', perplexity=TSNE_PERPLEXITY)
    tsne_features = tsne.fit_transform(X_scaled)

    # Attach t-SNE coords to df_sorted
    df['tsne-d1'] = tsne_features[:, 0]
    df['tsne-d2'] = tsne_features[:, 1]

    # Compute PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Attach PCA coords to df_scaled
    df['pca-d1'] = principal_components[:, 0]
    df['pca-d2'] = principal_components[:, 1]

    
    return df, X_scaled, tsne_features, principal_components

def get_shape_analysis(df):
    '''
    Load shape tsne results and map shape_type to tsne-d1 order
    '''
    # Load shape analysis results
    df_shape = pd.read_csv(alexnet_tsne_path)

    for row_idx, row in df.iterrows():
        shape_type_trial = row['shape_id']
        shape_tsne_d1 = df_shape.loc[df_shape['shape_id'] == shape_type_trial, 'TSNE1'].values
        shape_tsne_d2 = df_shape.loc[df_shape['shape_id'] == shape_type_trial, 'TSNE2'].values
        if shape_tsne_d1.size > 0:
            df.at[row_idx, 'shape_tsne-d1'] = shape_tsne_d1[0]
        else:
            df.at[row_idx, 'shape_tsne-d1'] = np.nan
        if shape_tsne_d2.size > 0:
            df.at[row_idx, 'shape_tsne-d2'] = shape_tsne_d2[0]
        else:
            df.at[row_idx, 'shape_tsne-d2'] = np.nan
    return df

def plot_dim_red(df, method="tsne", color_by=None):
    # ---- Setup Filenames ----
    if method == "tsne":
        dim1 = 'tsne-d1'
        dim2 = 'tsne-d2'
        base_name = f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_{method}_perplexity{TSNE_PERPLEXITY}"
    elif method == "pca":
        dim1 = 'pca-d1'
        dim2 = 'pca-d2'
        base_name = f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_{method}"

    save_fname = f"{base_name}_{color_by}.png" if color_by else f"{base_name}.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))

    # ---- 1. Color Mapping Logic ----
    df = get_shape_analysis(df)
    shape_types = sorted(df['shape_type'].unique())
    
    # Check if we are doing continuous coloring
    is_continuous = color_by in ["shape_tsne1", "shape_tsne2"]
    
    if is_continuous:
        target_col = 'shape_tsne-d1' if color_by == "shape_tsne1" else 'shape_tsne-d2'
        cmap = plt.get_cmap('viridis')
        
        vmin, vmax = df[target_col].min(), df[target_col].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        color_map = {}
        for st in shape_types:
            val = df.loc[df['shape_type'] == st, target_col].mean()
            color_map[st] = cmap(norm(val))
            
    else: 
        # Default: Categorical coloring (Spectral)
        # With 622 shapes, we use a cyclical colormap or high-contrast map, 
        # though duplicates will occur.
        cmap = plt.get_cmap('Spectral') 
        if len(shape_types) > 1:
            color_map = {st: cmap(i / (len(shape_types) - 1)) for i, st in enumerate(shape_types)}
        else:
            color_map = {shape_types[0]: cmap(0.5)}

    # ---- 2. Marker & Edge Logic ----
    marker_map = {"0": "o", "2": "D", "02": "*"}
    df['marker'] = df['orientation'].astype(str).map(lambda x: marker_map.get(x, "o"))

    # Edge Logic
    # edge_label_map = {}
    # if TRIAL_TYPE == "all":
    #     edge_color_map = {True: 'silver', False: 'black'}
    #     edge_label_map = {True: 'Correct', False: 'Error'}
    #     df['edgecolor'] = df['correct'].map(edge_color_map)
    # elif TRIAL_TYPE == "correct":
    #     edge_color_map = {True: 'black', False: 'silver'}
    #     edge_label_map = {True: 'Long', False: 'Short'}
    #     df['edgecolor'] = df['is_holdlong'].map(edge_color_map)
    # else:
    #     df['edgecolor'] = 'silver'
    edge_label_map = {}
    df['edgecolor'] = 'silver'

    # ---- Plotting ----
    # Iterate through shapes to plot
    for st in shape_types:
        sub = df[df['shape_type'] == st]
        for m in sub['marker'].unique():
            tmp = sub[sub['marker'] == m]
            if tmp.empty: continue
            
            ax.scatter(
                tmp[dim1], tmp[dim2],
                c=[color_map[st]] * len(tmp),
                marker=m,
                s=50,
                linewidths=0.5,
                edgecolors=tmp['edgecolor'],
                alpha=1,
                label=f"{st}"
            )

    ax.set_title(f"Hand_conformation_{TRIAL_TYPE}_f{FRAME_NUMBER}\nColor: {color_by if color_by else 'ShapeType'}", fontsize=14)
    ax.set_xlabel(dim1, fontsize=12)
    ax.set_ylabel(dim2, fontsize=12)

    # ---- Legends / Colorbar ----
    
    # A. Handle Color Legend
    if is_continuous:
        # Continuous -> Colorbar
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Value: {color_by}", fontsize=10)
    else:
        # Discrete -> Check count
        if len(shape_types) > 20:
            # TOO MANY SHAPES: Do NOT plot shape legend
            print(f"Skipping shape legend (n={len(shape_types)} types)")
        else:
            color_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map[st], markersize=8,
                           label=str(st))
                for st in shape_types
            ]
            leg1 = ax.legend(handles=color_handles, title='Shape Type',
                             loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=8, fontsize='small', title_fontsize='small',
                             markerscale=0.7, frameon=False)
            ax.add_artist(leg1)

    # B. Marker & Edge Legends (Bottom Row)
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color='k', linestyle='',
                   markersize=8, label=ori)
        for ori, m in marker_map.items() if str(ori) in df['orientation'].astype(str).unique()
    ]

    edge_handles = []
    if edge_label_map:
        for key, label_text in edge_label_map.items():
            ec = edge_color_map[key]
            edge_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='white', markeredgecolor=ec, markeredgewidth=2, markersize=8,
                           label=label_text)
            )

    # Calculate positions
    # If we skipped the shape legend, we can put the others higher up
    has_shape_legend = (not is_continuous) and (len(shape_types) <= 20)
    bottom_anchor_y = -0.15 if not has_shape_legend else -0.25
    
    if marker_handles:
        leg2 = ax.legend(handles=marker_handles, title='Orientation',
                         loc='upper center', bbox_to_anchor=(0.35, bottom_anchor_y), 
                         ncol=len(marker_handles), fontsize='small', title_fontsize='small', frameon=False)
        ax.add_artist(leg2)

    if edge_handles:
        title_text = "Trial Outcome" if TRIAL_TYPE == "all" else "Hold Type"
        ax.legend(handles=edge_handles, title=title_text,
                  loc='upper center', bbox_to_anchor=(0.65, bottom_anchor_y), 
                  ncol=len(edge_handles), fontsize='small', title_fontsize='small', frameon=False)

    # Adjust layout
    # Less bottom padding needed if no shape legend
    plt.subplots_adjust(bottom=0.2 if not has_shape_legend else 0.3, right=0.9)
    
    save_path = os.path.join(HAND_TSNE_DIR, save_fname)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {save_path}")

def main():
    # --- 1. Load Data ---
    hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
    hand_feat_path = HAND_RDM_DIR / hand_feat_csv
    df_hand = pd.read_csv(hand_feat_path)

    feature_names = list(df_hand.keys()[1:]) # the first column is shape_id

    # Compute tSNE (Pass feature_names explicitly)
    df_tsne_pca, X_scaled, tsne_features, principal_components = compute_tsne_pca(df_hand, feature_names)

    # Plot tSNE and PCA
    plot_dim_red(df_tsne_pca, method="tsne")
    plot_dim_red(df_tsne_pca, method="tsne", color_by="shape_tsne1")
    plot_dim_red(df_tsne_pca, method="tsne", color_by="shape_tsne2")
    plot_dim_red(df_tsne_pca, method="pca")
    plot_dim_red(df_tsne_pca, method="pca", color_by="shape_tsne1")
    plot_dim_red(df_tsne_pca, method="pca", color_by="shape_tsne2")

    # tSNE clustering analysis
    # kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=42)
    # df_tsne_pca['cluster'] = kmeans.fit_predict(tsne_features)

    # Save tSNE results
    # df_tsne_pca.to_csv(os.path.join(SAVE_ROOT, f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_pca_tsne_perplexity{TSNE_PERPLEXITY}.csv"), index=False)
    # Get feature contribution
    # get_feature_contribution(X_scaled, tsne_features, feature_names)

if __name__ == "__main__":
    main()
