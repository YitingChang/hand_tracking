import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pickle
import pandas as pd
import ast
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from hand_tracker.utils.file_io import get_trialname, find_matching_log


# --- CONFIGURATION ---
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
CONTACT_RDM_SAVE_DIR = ANALYSIS_ROOT / "contact_analysis" / "contact_rdms"
# For aligning with master list of shapes (from shape RDM analysis)
SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
SHAPE_ID_SAVE_PATH = ANALYSIS_ROOT / "shape_analysis" / 'shape_ids.pkl'

TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['0', '2', '02'] 

def load_contact_features(file_path):
    """
    Load contact features from a CSV file and return a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def feature_reduction(contact_features, n_components=50):  
    scaler = StandardScaler()
    contact_features_scaled = scaler.fit_transform(contact_features)
    
    # Use standard solver for stability when n_components is small
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(contact_features_scaled)
    
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var_ratio)
    print(f"Total variance explained by {n_components} components: {cum_exp_var[-1]*100:.2f}%")
    
    return reduced_features

def main():

    os.makedirs(CONTACT_RDM_SAVE_DIR, exist_ok=True)
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20",
                      "2025-12-08", "2025-12-09", "2025-12-18"]
    
    df_all_list = []

    for session_name in session_names:
        contact_feature_path = ANALYSIS_ROOT / session_name / "contact" / f"contact_features_{session_name}_holdwindow.csv"
        df = load_contact_features(contact_feature_path)
        if not df.empty:
            df_all_list.append(df)

    # Combine all DataFrames
    df_all = pd.concat(df_all_list, ignore_index=True)

    # Filtering
    df_filtered = df_all[df_all["correct"] == True].copy()
    if "short" in TRIAL_TYPE: df_filtered = df_filtered[df_filtered["is_holdshort"]]
    
    ori_suffixes = tuple(f"_{ori}" for ori in ORIENTATION_LIST)
    df_filtered = df_filtered[df_filtered["shape_id"].str.endswith(ori_suffixes)]

    # Averaging
    df_filtered['contact_vector_numeric'] = df_filtered['contact_vector'].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
    )

    df_avg = df_filtered.groupby('shape_id')['contact_vector_numeric'].apply(
        lambda x: np.mean(np.stack(x.values), axis=0)
    ).reset_index()

    df_avg = df_avg.rename(columns={'contact_vector_numeric': 'contact_vector'})

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

    # 4. SAVE THE PICKLE
    ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
    save_name_pkl = f"contact_avg_features_{TRIAL_TYPE}_{ori_str}.pkl"
    save_path = CONTACT_RDM_SAVE_DIR / save_name_pkl

    # Save to pickle
    df_avg_ordered.to_pickle(save_path)
    
    print(f"✅ Successfully saved {len(df_avg_ordered)} shapes to {save_path}")

    # 4. SAVE THE CSV
    # ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

    # save_name_csv = f"contact_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
    # save_path = CONTACT_RDM_SAVE_DIR / save_name_csv

    # save_path = CONTACT_RDM_SAVE_DIR / save_name_csv
    # df_avg_ordered.to_csv(save_path, index=False)
    
    # print(f"Success! Saved {len(df_avg_ordered)} shapes to {save_path}")

    # RDM Calculation
    # Stack the Series of 1D vectors into a unified 2D matrix shape (n_shapes, resolution^2)
    contact_features = np.stack(df_avg_ordered["contact_vector"].values)
    # Perform feature reduction (PCA) before computing the RDM
    contact_matrix = feature_reduction(contact_features)

    contact_rdm = squareform(pdist(contact_matrix, metric='correlation'))
    
    output = {'rdm': contact_rdm, 'shape_ids': df_avg_ordered['shape_id'].tolist(), 'trial_type': TRIAL_TYPE}
    save_name_rdm = f"contact_rdms_{TRIAL_TYPE}_{ori_str}.pkl"
    with open(CONTACT_RDM_SAVE_DIR / save_name_rdm, 'wb') as f:
        pickle.dump(output, f)
    print(f"Contact RDM saved for {len(output['shape_ids'])} conditions.")

if __name__ == "__main__":
    main()