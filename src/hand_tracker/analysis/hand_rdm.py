import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from hand_tracker.utils.file_io import get_trialname, find_matching_log

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
HAND_RDM_SAVE_DIR = os.path.join(ANALYSIS_ROOT, "hand_analysis")
SHAPE_RDM_SAVE_DIR = os.path.join(ANALYSIS_ROOT, "shape_analysis")
SHAPE_ID_SAVE_PATH = os.path.join(SHAPE_RDM_SAVE_DIR, 'shape_ids.pkl')

FRAME_NUMBER = 300
TRIAL_TYPE = "correct" 
ORIENTATION = ['0', '2', '02'] 

def get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames):
    df_list = []
    for feature_fname, log_fname in zip(feature_fnames, log_fnames):
        if log_fname == "nan": continue
        log_path = os.path.join(log_dir, log_fname)
        feature_path = os.path.join(feature_dir, feature_fname)
        feature_df = pd.read_csv(feature_path)

        with open(log_path, 'r') as file:
            json_data = json.load(file)

        if FRAME_NUMBER < len(feature_df):
            new_df = feature_df.iloc[[FRAME_NUMBER]].copy()
            new_df["trial_name"] = get_trialname(log_fname)
            new_df["shape_id"] = json_data.get("shape_id", "unknown_0")
            new_df["correct"] = json_data.get("has_played_success_tone", False)
            new_df["is_holdshort"] = json_data.get("object_released", False)
            new_df["is_holdlong"] = json_data.get("object_held", False)
            df_list.append(new_df)
            
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        metadata_cols = ["trial_name", "shape_id", "correct", "is_holdshort", "is_holdlong"]
        feature_names = [c for c in df.columns if c not in metadata_cols]
        return df, feature_names
    return pd.DataFrame(), []

def main():
    os.makedirs(HAND_RDM_SAVE_DIR, exist_ok=True)
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20", "2025-12-08", "2025-12-09", "2025-12-18"]
    
    df_all_list = []
    all_feature_names = []

    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")
        if not os.path.exists(feature_dir): continue

        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))
        log_fnames = find_matching_log(feature_fnames, log_dir)
        feature_fnames_base = [os.path.basename(f) for f in feature_fnames]

        df_session, feature_names = get_feature_log(feature_dir, feature_fnames_base, log_dir, log_fnames)
        if not df_session.empty:
            df_all_list.append(df_session)
            all_feature_names = feature_names 

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Filtering
    df_filtered = df_all[df_all["correct"] == True].copy()
    if "short" in TRIAL_TYPE: df_filtered = df_filtered[df_filtered["is_holdshort"]]
    
    ori_suffixes = tuple(f"_{ori}" for ori in ORIENTATION)
    df_filtered = df_filtered[df_filtered["shape_id"].str.endswith(ori_suffixes)]

    # Averaging
    feature_cols = [col for col in all_feature_names if col in df_filtered.columns]
    df_filtered[feature_cols] = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')

    df_avg = df_filtered.groupby("shape_id")[feature_cols].mean().reset_index()

    # Aligning with Master List
    with open(SHAPE_ID_SAVE_PATH, 'rb') as f:
        original_master_list = pickle.load(f)

    data_ids = df_avg['shape_id'].unique()
    final_aligned_ids = []
    for base_id in original_master_list:
        for suffix in ['_0', '_2', '_02']:
            combined_id = f"{base_id}{suffix}"
            if combined_id in data_ids:
                final_aligned_ids.append(combined_id)

    # Create the final ordered dataframe
    df_master_aligned = pd.DataFrame({"shape_id": final_aligned_ids})
    df_avg_ordered = df_master_aligned.merge(df_avg, on="shape_id", how="left")
    
    # Drop any shapes that ended up with NaNs before saving
    df_avg_ordered = df_avg_ordered.dropna().reset_index(drop=True)

    # 4. SAVE THE CSV (Ensuring the filename is exact)
    # We force the name to 'hand_avg_features_correct.csv' for the next script
    save_filename = "hand_avg_features_correct.csv"
    save_path = os.path.join(HAND_RDM_SAVE_DIR, save_filename)
    df_avg_ordered.to_csv(save_path, index=False)
    
    print(f"Success! Saved {len(df_avg_ordered)} shapes to {save_path}")

    # RDM Calculation
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    
    hand_matrix = StandardScaler().fit_transform(df_avg_ordered[all_feature_names])
    hand_rdm = squareform(pdist(hand_matrix, metric='correlation'))
    
    output = {'rdm': hand_rdm, 'shape_ids': df_avg_ordered['shape_id'].tolist(), 'trial_type': TRIAL_TYPE}
    save_name = f"hand_rdm_{TRIAL_TYPE}_aligned.pkl"
    with open(os.path.join(HAND_RDM_SAVE_DIR, save_name), 'wb') as f:
        pickle.dump(output, f)
    print(f"Hand RDM saved for {len(output['shape_ids'])} conditions.")

if __name__ == "__main__":
    main()