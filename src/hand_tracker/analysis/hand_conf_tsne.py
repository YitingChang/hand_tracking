import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
# from sklearn.manifold import TSNE
# from openTSNE import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from hand_tracker.utils.file_io import get_trialname

# ------ CONFIGURATION ------
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
FRAME_NUMBER = 300
TSNE_PERPLEXITY = 10
SAVE_ROOT = os.path.join(ANALYSIS_ROOT,"hand_conformation")
TRIAL_TYPE = "all"

def get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames):
    # Create a table
    # 1. Extract angles from the holding time
    # 2. Add trial name and shape id
    df = pd.DataFrame()
    for log_fname, feature_fname in zip(log_fnames, feature_fnames):
        
        feature_df = pd.read_csv(os.path.join(feature_dir, feature_fname))
        # Open and read the JSON file
        with open(os.path.join(log_dir, log_fname), 'r') as file:
            json_data = json.load(file)

        new_df = feature_df.loc[[FRAME_NUMBER]]
        new_df["trial_name"] = get_trialname(log_fname)
        new_df["shape_id"] = json_data["shape_id"]
        new_df["correct"] = json_data["has_played_success_tone"]
        new_df["is_holdshort"] = json_data["object_released"]
        new_df["is_holdlong"] = json_data["object_held"]
        new_df["is_sameShape"] = json_data["reward_direction"] == 'holdshort'

        df = pd.concat([df, new_df])
        
    df.reset_index(drop=True, inplace=True)

    return df

def compute_tsne(df, trial_type=None):
    # Prepare data for t-SNE
    # Filter trials
    if trial_type == "correct":
        df_filtered = df.dropna().query('correct == True').copy()
    elif trial_type == "correct-long":
        df_filtered = df.dropna().query('correct == True and is_holdlong == True').copy()
    elif trial_type == "correct-short":
        df_filtered = df.dropna().query('correct == True and is_holdshort == True').copy()
    elif trial_type == "all":
        df_filtered = df.dropna().copy()

    # Rank by shape_id, sort, reset
    df_sorted = (
        df_filtered.assign(rank=df_filtered['shape_id'].str.lower().rank(method='dense'))
                .sort_values('rank')
                .drop(columns='rank')
                .reset_index(drop=True)
    )

    # Extract shape_type and orientation
    df_sorted['shape_type'] = df_sorted['shape_id'].apply(lambda x: x.split('_')[0])
    df_sorted['orientation'] = df_sorted['shape_id'].apply(lambda x: x.split('_')[1])

    # Prepare features for t-SNE
    df_features = df_sorted.iloc[:, 0:-8]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42,
                learning_rate='auto', init='random', perplexity=TSNE_PERPLEXITY)
    tsne_features = tsne.fit_transform(X_scaled)

    # Attach t-SNE coords to df_scaled
    df_sorted['tsne-d1'] = tsne_features[:, 0]
    df_sorted['tsne-d2'] = tsne_features[:, 1]
    
    return df_sorted


def plot_tsne(df):
    # -----------------------------------------
    # Plot: color by shape_type, marker by orientation
    # -----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # ---- Color mapping (shape_type) ----
    shape_types = df['shape_type'].unique()
    # Get the colormap
    cmap = plt.get_cmap('Spectral')
    # Sample evenly spaced colors from 0 to 1
    color_map = {st: cmap(i / (len(shape_types) - 1)) for i, st in enumerate(shape_types)}

    # cmap = cm.get_cmap('Spectral', len(shape_types))
    # color_map = {st: cmap(i) for i, st in enumerate(shape_types)}

    # ---- Marker mapping (orientation) ----
    marker_map = {
        "0": "o",      # circle
        "2": "D",      # diamond
        "02": "*",     # star
    }
    df['marker'] = df['orientation'].astype(str).map(marker_map)

    # # ---- Edge mapping (correct vs error) ----
    # edge_map = {
    #     True: 'none',
    #     False: 'black'
    # }
    # df_sorted['edgecolor'] = df_sorted['correct'].map(edge_map)

    # ---- Edge mapping (long vs short) ----
    edge_map = {
        True: 'none',
        False: 'black'
    }
    df['edgecolor'] = df['correct'].map(edge_map)

    # ---- Plot grouped by color + marker + edge ----
    for st in shape_types:
        sub = df[df['shape_type'] == st]
        for m in sub['marker'].unique():
            tmp = sub[sub['marker'] == m]
            ax.scatter(
                tmp['tsne-d1'], tmp['tsne-d2'],
                color=color_map[st],
                marker=m,
                s=30,
                edgecolors=tmp['edgecolor'],
                label=f"{st}, {m}"
            )

    # ---- Labels ----
    ax.set_title(f"Hand_conformation_{TRIAL_TYPE}_f{FRAME_NUMBER}")
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    plt.tight_layout()

    # ---- Three separate legends ----
    # 1. Legend for shape_type (colors)
    color_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color_map[st], markersize=8,
                label=str(st))
        for st in shape_types
    ]

    # 2. Legend for orientation (markers)
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color='k', linestyle='',
                markersize=8, label=ori)
        for ori, m in marker_map.items()
    ]
    # 3. Legend for correct vs error or hold long vs hold short (edgecolors)
    edge_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor='gray', markeredgecolor=ec, markersize=8,
                label='error' if ec == 'black' else 'correct')
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

    # Save the figure
    plt.savefig(os.path.join(SAVE_ROOT, f"hand_conf_{TRIAL_TYPE}_f{FRAME_NUMBER}_tsne_perplexity{TSNE_PERPLEXITY}.png"), dpi=300, bbox_inches='tight')

def main():
    # Get features and logs from all sessions
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20", 
                     "2025-12-08", "2025-12-09", "2025-12-18"]
    df_all = pd.DataFrame()
    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")
        log_fnames = sorted(glob(os.path.join(log_dir, "*.json")))

        df_session = get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames)
        df_all = pd.concat([df_all, df_session])
        
    df_all.reset_index(drop=True, inplace=True)

    # Compute tSNE
    df_tsne = compute_tsne(df_all, trial_type=TRIAL_TYPE)

    # Plot tSNE
    plot_tsne(df_tsne)

if __name__ == "__main__":
    main()









