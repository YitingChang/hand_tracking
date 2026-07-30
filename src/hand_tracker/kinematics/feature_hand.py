import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from hand_tracker.utils.file_io import get_trialname, find_log_or_robot
from hand_tracker.utils.analysis_window import load_window_lookup

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")

TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 

def get_feature_log(feature_dir, feature_fnames, log_dir, log_fnames, window_lookup):
    df_list = []
    skipped_no_window = 0
    skipped_bad_window = 0

    for feature_fname, log_fname in zip(feature_fnames, log_fnames):
        if log_fname == "nan": continue
        log_path = os.path.join(log_dir, log_fname)
        feature_path = os.path.join(feature_dir, feature_fname)
        feature_df = pd.read_csv(feature_path)

        with open(log_path, 'r') as file:
            json_data = json.load(file)

        trial_name_video = get_trialname(feature_fname)
        window = window_lookup.get(trial_name_video)
        if window is None:
            skipped_no_window += 1
            continue

        start_frame, end_frame = window
        if pd.isna(start_frame) or pd.isna(end_frame):
            skipped_no_window += 1
            continue

        start_frame = max(int(start_frame), 0)
        end_frame = min(int(end_frame), len(feature_df) - 1)
        if start_frame > end_frame:
            skipped_bad_window += 1
            continue

        window_df = feature_df.iloc[start_frame:end_frame + 1]
        if window_df.empty:
            skipped_bad_window += 1
            continue

        # Average all numeric feature columns over the hold window
        new_df = window_df.mean(numeric_only=True).to_frame().T
        new_df["trial_name_log"] = get_trialname(log_fname)
        new_df["trial_name_video"] = trial_name_video
        new_df["shape_id"] = json_data.get("shape_id", "unknown_0")
        new_df["correct"] = json_data.get("has_played_success_tone", False)
        new_df["is_holdshort"] = json_data.get("object_released", False)
        new_df["is_holdlong"] = json_data.get("object_held", False)
        df_list.append(new_df)

    if skipped_no_window or skipped_bad_window:
        print(f"  Skipped {skipped_no_window} trials with no matching hold window, "
              f"{skipped_bad_window} trials with an empty/invalid window.")

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        metadata_cols = ["trial_name_log", "trial_name_video", "shape_id", "correct", "is_holdshort", "is_holdlong"]
        feature_names = [c for c in df.columns if c not in metadata_cols]
        return df, feature_names
    return pd.DataFrame(), []

def main():
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20",
                      "2025-12-08", "2025-12-09", "2025-12-18"]
    
    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")
        if not os.path.exists(feature_dir): continue

        window_lookup = load_window_lookup(session_name)
        if window_lookup is None:
            print(f"Warning: no min_holding_window.csv found for {session_name}, skipping session.")
            continue

        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))
        log_fnames = find_log_or_robot(feature_fnames, log_dir)
        feature_fnames_base = [os.path.basename(f) for f in feature_fnames]

        df_session, feature_names = get_feature_log(feature_dir, feature_fnames_base, log_dir, log_fnames, window_lookup)
        if not df_session.empty:

            save_dir = os.path.join(ANALYSIS_ROOT, session_name, "hand")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"hand_features_{session_name}_holdwindow.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(df_session, f)
            print(f"Saved hand features for session {session_name} to {save_path}")

            meta_save_path = os.path.join(save_dir, f"hand_feature_names_{session_name}_holdwindow.pkl")
            with open(meta_save_path, 'wb') as f:
                pickle.dump(feature_names, f)
            print(f"Saved feature names for session {session_name} to {meta_save_path}")

if __name__ == "__main__":
    main()