import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from hand_tracker.utils.file_io import get_trialname, find_matching_log

# ------ CONFIGURATION ------
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SAVE_ROOT = os.path.join(ANALYSIS_ROOT, "hand_conf_dim_red")
# Parameters
FRAME_NUMBER = 300
TSNE_PERPLEXITY = 30
NUM_CLUSTER = 4
TRIAL_TYPE = "correct-short" # "all", "correct", "correct-long", "correct-short"
# ---------------------------   

def get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames):
    """
    Reads feature CSVs and corresponding JSON logs.
    Returns a combined dataframe and the list of feature column names.
    """
    df_list = []
    feature_names = []

    # Ensure we only process if we have matching file counts, 
    # though zip will just stop at the shortest list.
    for log_fname, feature_fname in zip(log_fnames, feature_fnames):
        feature_path = os.path.join(feature_dir, feature_fname)
        log_path = os.path.join(log_dir, log_fname)

        feature_df = pd.read_csv(feature_path)
        
        # Store feature names from the first file we encounter
        if not feature_names:
            feature_names = list(feature_df.columns)

        if log_fname == "nan":
            continue
        else:
            with open(log_path, 'r') as file:
                json_data = json.load(file)

            # Check if FRAME_NUMBER is within bounds
            if FRAME_NUMBER in feature_df.index:
                new_df = feature_df.loc[[FRAME_NUMBER]].copy()
                new_df["trial_name"] = get_trialname(log_fname)
                new_df["shape_id"] = json_data.get("shape_id", "unknown_0") # .get avoids key errors
                new_df["correct"] = json_data.get("has_played_success_tone", False)
                new_df["is_holdshort"] = json_data.get("object_released", False)
                new_df["is_holdlong"] = json_data.get("object_held", False)
                new_df["is_sameShape"] = json_data.get("reward_direction") == 'holdshort'

                df_list.append(new_df)
        
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()

    return df, feature_names

def get_shape_analysis(df):
    '''
    Load shape tsne results and map shape_type to tsne-d1 order
    '''
    # Load shape analysis results
    df_shape = pd.read_csv(SHAPE_ANALYSIS_PATH)

    for row_idx, row in df.iterrows():
        shape_type_trial = row['shape_type']
        shape_tsne_d1 = df_shape.loc[df_shape['shape_type'] == shape_type_trial, 'TSNE1'].values
        shape_tsne_d2 = df_shape.loc[df_shape['shape_type'] == shape_type_trial, 'TSNE2'].values
        if shape_tsne_d1.size > 0:
            df.at[row_idx, 'shape_tsne-d1'] = shape_tsne_d1[0]
        else:
            df.at[row_idx, 'shape_tsne-d1'] = np.nan
        if shape_tsne_d2.size > 0:
            df.at[row_idx, 'shape_tsne-d2'] = shape_tsne_d2[0]
        else:
            df.at[row_idx, 'shape_tsne-d2'] = np.nan
    return df

def compute_tsne_pca(df, feature_names, trial_type=None):
    # Filter trials
    if trial_type == "correct":
        df_filtered = df.dropna().query('correct == True').copy()
    elif trial_type == "correct-long":
        df_filtered = df.dropna().query('correct == True and is_holdlong == True').copy()
    elif trial_type == "correct-short":
        df_filtered = df.dropna().query('correct == True and is_holdshort == True').copy()
    elif trial_type == "all":
        df_filtered = df.dropna().copy()
    else:
        df_filtered = df.dropna().copy()

    # Rank by shape_id, sort, reset
    # Handling cases where shape_id might not be a string
    df_filtered['shape_id'] = df_filtered['shape_id'].astype(str)
    
    df_sorted = (
        df_filtered.assign(rank=df_filtered['shape_id'].str.lower().rank(method='dense'))
                .sort_values('rank')
                .drop(columns='rank')
                .reset_index(drop=True)
    )

    # Extract shape_type and orientation safely
    df_sorted['shape_type'] = df_sorted['shape_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    df_sorted['orientation'] = df_sorted['shape_id'].apply(lambda x: x.split('_')[1] if '_' in x else '0')

    # Prepare features for t-SNE using explicit column names
    df_features = df_sorted[feature_names]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42,
                learning_rate='auto', init='random', perplexity=TSNE_PERPLEXITY)
    tsne_features = tsne.fit_transform(X_scaled)

    # Attach t-SNE coords to df_sorted
    df_sorted['tsne-d1'] = tsne_features[:, 0]
    df_sorted['tsne-d2'] = tsne_features[:, 1]

    # Compute PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Attach PCA coords to df_scaled
    df_sorted['pca-d1'] = principal_components[:, 0]
    df_sorted['pca-d2'] = principal_components[:, 1]

    
    return df_sorted, X_scaled, tsne_features, principal_components

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
    edge_label_map = {}
    if TRIAL_TYPE == "all":
        edge_color_map = {True: 'silver', False: 'black'}
        edge_label_map = {True: 'Correct', False: 'Error'}
        df['edgecolor'] = df['correct'].map(edge_color_map)
    elif TRIAL_TYPE == "correct":
        edge_color_map = {True: 'black', False: 'silver'}
        edge_label_map = {True: 'Long', False: 'Short'}
        df['edgecolor'] = df['is_holdlong'].map(edge_color_map)
    else:
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
                s=30,
                linewidths=0.5,
                edgecolors=tmp['edgecolor'],
                alpha=0.8,
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
    
    save_path = os.path.join(SAVE_ROOT, save_fname)
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
    
    save_path = os.path.join(SAVE_ROOT, f"feature_importance_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png")
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
    plt.savefig(os.path.join(SAVE_ROOT, f"feature_importance_d1_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting Dimension 2 (Y-axis) drivers
    if shap_d2 is not None:
        plt.figure()
        plt.title("What drives separation along the Vertical Axis (Dim 2)?")
        shap.summary_plot(shap_d2, X_df, show=False)
        plt.savefig(os.path.join(SAVE_ROOT, f"feature_importance_d2_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Get features and logs from all sessions
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20", 
                     "2025-12-08", "2025-12-09", "2025-12-18"]
    
    df_all_list = []
    final_feature_names = None

    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")
        
        # Check if dirs exist to avoid crash
        if not os.path.exists(feature_dir) or not os.path.exists(log_dir):
            print(f"Skipping session {session_name}: directory not found.")
            continue

        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))

        # Find corresponding log files

        log_fnames = find_matching_log(feature_fnames, log_dir)
        
        # Basic integrity check
        if len(feature_fnames) != len(log_fnames):
            print(f"Warning: Session {session_name} has mismatched file counts ({len(feature_fnames)} features vs {len(log_fnames)} logs). Truncating to shortest.")

        df_session, feature_names = get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames)
        
        if not df_session.empty:
            df_all_list.append(df_session)
            final_feature_names = feature_names # Assume consistent features across sessions
    
    if not df_all_list:
        print("No data found. Exiting.")
        return

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Compute tSNE (Pass feature_names explicitly)
    df_tsne_pca, X_scaled, tsne_features, principal_components = compute_tsne_pca(df_all, final_feature_names, trial_type=TRIAL_TYPE)

    # Plot tSNE and PCA
    plot_dim_red(df_tsne_pca, method="tsne")
    plot_dim_red(df_tsne_pca, method="tsne", color_by="shape_tsne1")
    plot_dim_red(df_tsne_pca, method="tsne", color_by="shape_tsne2")
    plot_dim_red(df_tsne_pca, method="pca")
    plot_dim_red(df_tsne_pca, method="pca", color_by="shape_tsne1")
    plot_dim_red(df_tsne_pca, method="pca", color_by="shape_tsne2")
    # tSNE clustering analysis
    kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=42)
    df_tsne_pca['cluster'] = kmeans.fit_predict(tsne_features)

    # Save tSNE results
    df_tsne_pca.to_csv(os.path.join(SAVE_ROOT, f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_pca_tsne_perplexity{TSNE_PERPLEXITY}.csv"), index=False)
    # Get feature contribution
    # get_feature_contribution(X_scaled, tsne_features, feature_names)

if __name__ == "__main__":
    main()