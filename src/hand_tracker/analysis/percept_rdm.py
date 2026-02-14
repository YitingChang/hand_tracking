import os
from pathlib import Path
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
PERCEPT_DIS_SAVE_DIR = ANALYSIS_ROOT / "percept_analysis"
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
os.makedirs(PERCEPT_DIS_SAVE_DIR, exist_ok=True)

# Parameters to match the hand and alexnet RDMs that we want to compare with
# Note: We calculate behavioral/perceptual RDMs from engaged trials (which could be either correct or incorrect).
# But for the hand and alexnet RDMs, we typically only calculate them from correct trials to get a cleaner signal. 
# So the trial type for the perceptual RDM can be different from the hand/alexnet RDMs, but we should specify it clearly here.
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

def construct_log_dataframe(log_dir, log_fnames):
    log_data = []
    for fname in log_fnames:
        if not fname.endswith('.json'): continue
        log_path = os.path.join(log_dir, fname)
        with open(log_path, 'r') as file:
            try:
                data = json.load(file)
                log_data.append(data)
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(log_data)
    if df.empty: return df

    # --- DECISION LOGIC ---
    df["decision_nograsp"] = df["wait_start_timestamp_s"].isnull()
    # Reports 'Same'
    df["decision_holdshort"] = (df["object_released"] == True) & (df["wait_start_timestamp_s"].notnull())
    # Reports 'Different'
    df["decision_holdlong"] = (df["object_held"] == True) & (df["wait_start_timestamp_s"].notnull())
    
    df["was_engaged"] = df["decision_holdshort"] | df["decision_holdlong"]
    
    # --- CONSECUTIVE PAIRING ---
    df["prev_shape_id"] = df["shape_id"].shift(1)
    df["was_prev_engaged"] = df["was_engaged"].shift(1)
    
    # The first trial of a session cannot be a comparison trial
    df.loc[0, "was_prev_engaged"] = False 
    return df

def main():
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20",
                      "2025-12-08", "2025-12-09", "2025-12-18"]
    # 0. Construct a master dataframe from all logs
    df_list = []
    for session in session_names:
        log_dir = RAW_DATA_ROOT / session / "trial_logs"
        if log_dir.exists():
            df_session = construct_log_dataframe(log_dir, sorted(os.listdir(log_dir)))
            if df_session.empty: continue
            
            # Filter: Both current and previous trials must have a clear decision
            df_engaged = df_session[df_session["was_engaged"] & df_session["was_prev_engaged"]].copy()
            df_list.append(df_engaged)

    df_all = pd.concat(df_list, ignore_index=True)

    # 1. Align with Hand RDM Order
    hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
    df_hand = pd.read_csv(HAND_RDM_SAVE_DIR / hand_feat_csv)
    valid_ids = df_hand['shape_id'].astype(str).str.strip().tolist()

    # 2. Build Perceptual/Behavioral RDM
    n_shapes = len(valid_ids)
    percept_rdm = np.full((n_shapes, n_shapes), np.nan)

    # Convert df to dictionary for faster lookup
    # We aggregate (A,B) and (B,A) pairs to ensure symmetry and boost trial counts
    # But (A,B) and (B,A) could be different. We will investigate this later. 
    # For now, we will just combine them to get a more stable estimate of perceptual dissimilarity.
    for i, id_i in enumerate(tqdm(valid_ids, desc="Building Perceptual RDM")):
        for j in range(i, n_shapes):
            id_j = valid_ids[j]
            
            if i == j:
                percept_rdm[i, j] = 0 
                # Same shape pairs should have 0 dissimilarity in RDM, but in perceptual RDM, 
                # it might not be exactly 0 due to behavioral/perceptual errors. Here we set it to 0 for simplicity.
                continue
            
            # Find trials for both directions of the pair
            mask = ((df_all["shape_id"] == id_i) & (df_all["prev_shape_id"] == id_j)) | \
                   ((df_all["shape_id"] == id_j) & (df_all["prev_shape_id"] == id_i))
            
            pair_df = df_all[mask]
            
            if not pair_df.empty:
                # Dissimilarity = Probability of reporting 'Different'
                dist = pair_df["decision_holdlong"].mean()
                percept_rdm[i, j] = dist
                percept_rdm[j, i] = dist # Enforce symmetry

    # 3. Save
    save_data = {
        'rdm': percept_rdm,
        'shape_ids': valid_ids,
        'hand_metadata': {'trial_type': TRIAL_TYPE, 'orientations': ORIENTATION_LIST}
    }

    save_path = PERCEPT_DIS_SAVE_DIR / f"percept_rdm_for_hand_{TRIAL_TYPE}_{ori_str}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Perceptual RDM saved. Matched {len(valid_ids)} shapes. Final shape: {percept_rdm.shape}")

if __name__ == "__main__":
    main()