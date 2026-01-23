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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from hand_tracker.utils.file_io import get_trialname, find_matching_log

# ------ CONFIGURATION ------
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
FRAME_NUMBER = 300
TSNE_PERPLEXITY = 30
NUM_CLUSTER = 4
SAVE_ROOT = os.path.join(ANALYSIS_ROOT, "hand_conformation")
TRIAL_TYPE = "correct" # "all", "correct", "correct-long", "correct-short"

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

def plot_dim_red(df, method="tsne"):

    if method == "tsne":
        dim1 = 'tsne-d1'
        dim2 = 'tsne-d2'
        save_fname = f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_{method}_perplexity{TSNE_PERPLEXITY}.png"
    elif method == "pca":
        dim1 = 'pca-d1'
        dim2 = 'pca-d2'
        save_fname = f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_{method}.png"

    fig, ax = plt.subplots(figsize=(10, 8))

    # ---- Color mapping (shape_type) ----
    shape_types = sorted(df['shape_type'].unique())
    cmap = plt.get_cmap('Spectral')
    
    # Avoid division by zero if only 1 shape type
    if len(shape_types) > 1:
        color_map = {st: cmap(i / (len(shape_types) - 1)) for i, st in enumerate(shape_types)}
    else:
        color_map = {shape_types[0]: cmap(0.5)}

    # ---- Marker mapping (orientation) ----
    marker_map = {
        "0": "o",      # circle
        "2": "D",      # diamond
        "02": "*",     # star
    }
    # Default to 'o' if orientation not in map
    df['marker'] = df['orientation'].astype(str).map(lambda x: marker_map.get(x, "o"))

    # ---- Edge mapping ----
    
    if TRIAL_TYPE == "all":
        # For all trials, edge mapping is correct vs error
        edge_map = {
            True: 'none',
            False: 'black'
        }
        df['edgecolor'] = df['correct'].map(edge_map)

    elif TRIAL_TYPE == "correct":
        # For correct trials, edge mapping is long vs short
        edge_map = {
            True: 'none',
            False: 'black'
        }
        df['edgecolor'] = df['is_holdlong'].map(edge_map)


    # ---- Plot grouped by color + marker + edge ----
    for st in shape_types:
        sub = df[df['shape_type'] == st]
        for m in sub['marker'].unique():
            tmp = sub[sub['marker'] == m]
            # Ensure we have data to plot
            if tmp.empty: continue
            
            ax.scatter(
                tmp[dim1], tmp[dim2],
                color=color_map[st],
                marker=m,
                s=30,
                edgecolors=tmp['edgecolor'],
                label=f"{st}, {m}"
            )

    ax.set_title(f"Hand_conformation_{TRIAL_TYPE}_f{FRAME_NUMBER}")
    ax.set_xlabel(dim1)
    ax.set_ylabel(dim2)
    plt.tight_layout()

    # ---- Legends ----
    color_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color_map[st], markersize=8,
                label=str(st))
        for st in shape_types
    ]

    marker_handles = [
        plt.Line2D([0], [0], marker=m, color='k', linestyle='',
                markersize=8, label=ori)
        for ori, m in marker_map.items() if ori in df['orientation'].unique()
    ]

    if TRIAL_TYPE == "all":
        edge_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='gray', markeredgecolor=ec, markersize=8,
                    label='error' if ec == 'black' else 'correct')
            for ec in edge_map.values()
        ]
    elif TRIAL_TYPE == "correct":
        edge_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='gray', markeredgecolor=ec, markersize=8,
                    label='short' if ec == 'black' else 'long')
            for ec in edge_map.values()
        ]

    leg1 = ax.legend(handles=color_handles, title='shape_type',
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=8)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=marker_handles, title='orientation',
            loc='upper center', bbox_to_anchor=(0.2, -0.075), ncol=5)
    ax.add_artist(leg2)

    ax.legend(handles=edge_handles, title='trial outcome',
            loc='upper center', bbox_to_anchor=(0.8, -0.075), ncol=2)

    save_path = os.path.join(SAVE_ROOT, save_fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

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
    plot_dim_red(df_tsne_pca, method="pca")

    # tSNE clustering analysis
    kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=42)
    df_tsne_pca['cluster'] = kmeans.fit_predict(tsne_features)

    # Save tSNE results
    df_tsne_pca.to_csv(os.path.join(SAVE_ROOT, f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_pca_tsne_perplexity{TSNE_PERPLEXITY}.csv"), index=False)

    # Get feature contribution
    get_feature_contribution(X_scaled, tsne_features, feature_names)

if __name__ == "__main__":
    main()