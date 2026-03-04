import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


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
NUM_CLUSTER = 5

# --- Functions ---

def compute_tsne_pca(df, feature_names):

    # Extract object id and orientation safely
    df['object_id'] = df['shape_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
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

    df_shape = pd.read_csv(alexnet_tsne_path)
    
    # Pre-map for performance
    mapping_d1 = dict(zip(df_shape['shape_id'], df_shape['TSNE1']))
    mapping_d2 = dict(zip(df_shape['shape_id'], df_shape['TSNE2']))
    mapping_path = dict(zip(df_shape['shape_id'], df_shape['OverlayPath']))

    df['shape_tsne-d1'] = df['shape_id'].map(mapping_d1)
    df['shape_tsne-d2'] = df['shape_id'].map(mapping_d2)
    df['overlay_path'] = df['shape_id'].map(mapping_path)
    
    return df

def plot_dim_red(df, method="tsne", scatter_color_mode="monochrome", overlay=False, zoom=0.25):
    """
    scatter_color_mode: "monochrome", "obj_id", "shape_tsne1", "shape_tsne2"
    overlay: bool, if True, adds original color images
    """
    # ---- Setup Filenames ----
    dim1, dim2 = ('tsne-d1', 'tsne-d2') if method == "tsne" else ('pca-d1', 'pca-d2')
    base_name = f"hand_{TRIAL_TYPE}_{FRAME_NUMBER}_{method}_{scatter_color_mode}"
    save_fname = f"{base_name}_overlay.png" if overlay else f"{base_name}.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    df = get_shape_analysis(df)
    obj_ids = sorted(df['object_id'].unique())
    
    # ---- 1. Four-Way Color Logic ----
    is_continuous = False
    sm = None

    if scatter_color_mode == "monochrome":
        # Version 1: All points one neutral color
        color_map = {oi: "slategray" for oi in obj_ids}
        
    elif scatter_color_mode == "obj_id":
        # Version 2: Each object identity gets a unique categorical color
        cmap = plt.get_cmap('Spectral')
        color_map = {oi: cmap(i / len(obj_ids)) for i, oi in enumerate(obj_ids)}
        
    elif scatter_color_mode in ["shape_tsne1", "shape_tsne2"]:
        # Versions 3 & 4: Continuous mapping based on AlexNet feature space
        is_continuous = True
        target_col = 'shape_tsne-d1' if scatter_color_mode == "shape_tsne1" else 'shape_tsne-d2'
        
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=df[target_col].min(), vmax=df[target_col].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        color_map = {}
        for oi in obj_ids:
            val = df.loc[df['object_id'] == oi, target_col].mean()
            color_map[oi] = cmap(norm(val))

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
    for oi in obj_ids:
        sub = df[df['object_id'] == oi]
        for m in sub['marker'].unique():
            tmp = sub[sub['marker'] == m]
            if tmp.empty: continue
            
            ax.scatter(
                tmp[dim1], tmp[dim2],
                c=[color_map[oi]] * len(tmp),
                marker=m,
                s=50,
                linewidths=0.5,
                edgecolors=tmp['edgecolor'],
                alpha=0.7,
                label=f"{oi}"
            )
    if overlay:
        print(f"Adding image overlays...")
        for i, row in df.iterrows():
            if i % 15 == 0 and pd.notna(row['overlay_path']):
                if os.path.exists(row['overlay_path']):
                    img = Image.open(row['overlay_path']).convert('RGB')
                    imagebox = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(imagebox, (row[dim1], row[dim2]), frameon=True, 
                                        bboxprops=dict(edgecolor='black', alpha=0.3))
                    ax.add_artist(ab)
    ax.set_title(f"Hand Conformation Space\nScatter Color: {scatter_color_mode}", fontsize=16)
    ax.set_xlabel(dim1, fontsize=12)
    ax.set_ylabel(dim2, fontsize=12)

    # ---- Legends / Colorbar ----
    
    # A. Handle Color Legend
    if is_continuous and sm:
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"AlexNet {ALEXNET_LAYER} {scatter_color_mode}", fontsize=12)
    else:
        # Discrete -> Check count
        if len(obj_ids) > 20:
            # TOO MANY SHAPES: Do NOT plot shape legend
            print(f"Skipping shape legend (n={len(obj_ids)} types)")
        else:
            color_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map[oi], markersize=8,
                           label=str(oi))
                for oi in obj_ids
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
    has_shape_legend = (not is_continuous) and (len(obj_ids) <= 20)
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


def get_feature_contribution(X_scaled, tsne_features, feature_names):
    # Train a Surrogate Model (Random Forest)
    # Multi-output regression: X (features) -> Y (tSNE coord 1, tSNE coord 2)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, tsne_features)

    # 4. Extract Feature Importances
    importances = rf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 5. Visualization (Feature Importance)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        hue='Feature', 
        data=feature_importance_df.head(10), 
        palette='viridis', 
        legend=False
    )
    plt.title('Top 10 Features Driving the t-SNE Structure')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Original Feature Name')
    plt.tight_layout()
    
    save_path = os.path.join(HAND_TSNE_DIR, f"feature_importance_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTop 5 Most Influential Features:")
    print(feature_importance_df.head(5))

    # 6. SHAP Analysis
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_df)

    # Case 1: Old SHAP behavior (List of 2D arrays)
    if isinstance(shap_values, list):
        shap_d1 = shap_values[0]
        shap_d2 = shap_values[1]
    # Case 2: New SHAP behavior (Single 3D array)
    # Shape is (n_samples, n_features, n_outputs)
    elif len(shap_values.shape) == 3:
        shap_d1 = shap_values[:, :, 0]  # All samples, All features, Output 0
        shap_d2 = shap_values[:, :, 1]  # All samples, All features, Output 1
    # Case 3: Fallback (Single output model)
    else:
        print("Warning: Model appears to be single-output.")
        shap_d1 = shap_values
        shap_d2 = None

    # 7. Plot SHAP results 
    # Plotting Dimension 1 (X-axis) drivers
    plt.figure()
    plt.title("What drives separation along the Horizontal Axis (Dim 1)?")
    shap.summary_plot(shap_d1, X_df, show=False)
    plt.savefig(os.path.join(HAND_TSNE_DIR, f"feature_importance_d1_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting Dimension 2 (Y-axis) drivers
    if shap_d2 is not None:
        plt.figure()
        plt.title("What drives separation along the Vertical Axis (Dim 2)?")
        shap.summary_plot(shap_d2, X_df, show=False)
        plt.savefig(os.path.join(HAND_TSNE_DIR, f"feature_importance_d2_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png"), dpi=300, bbox_inches='tight')
        plt.close()



def main():
    # --- 1. Load Data ---
    hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
    hand_feat_path = HAND_RDM_DIR / hand_feat_csv
    df_hand = pd.read_csv(hand_feat_path)

    feature_names = list(df_hand.keys()[1:]) # the first column is shape_id

    # Compute tSNE (Pass feature_names explicitly)
    df_tsne_pca, X_scaled, tsne_features, principal_components = compute_tsne_pca(df_hand, feature_names)

    # Generate the 4 requested versions
    modes = ["monochrome", "obj_id", "shape_tsne1", "shape_tsne2"]
    
    for mode in modes:
        # Without overlay
        plot_dim_red(df_tsne_pca, method="tsne", scatter_color_mode=mode, overlay=False)
        # With overlay
        plot_dim_red(df_tsne_pca, method="tsne", scatter_color_mode=mode, overlay=True)

    for mode in modes:
        # Without overlay
        plot_dim_red(df_tsne_pca, method="pca", scatter_color_mode=mode, overlay=False)
        # With overlay
        plot_dim_red(df_tsne_pca, method="pca", scatter_color_mode=mode, overlay=True)

    # tSNE clustering analysis
    kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=42)
    df_tsne_pca['cluster'] = kmeans.fit_predict(tsne_features)

    # Save tSNE results
    df_tsne_pca.to_csv(os.path.join(HAND_TSNE_DIR, f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_pca_tsne_perplexity{TSNE_PERPLEXITY}.csv"), index=False)
    # Get feature contribution
    get_feature_contribution(X_scaled, tsne_features, feature_names)

if __name__ == "__main__":
    main()
