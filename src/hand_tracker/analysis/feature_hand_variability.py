import os
from pathlib import Path
from glob import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import cv2

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_variability"

FRAME_NUMBER = 300
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 

CAMERA_NAMES = ["camTo", "camTL", "camTR", "camBR", "camBL"]  # List of camera identifiers

def get_frame_from_video(session_name, trial_name, cam_name, frame_number, save_dir):
    video_path = ANALYSIS_ROOT / session_name / "litpose" / "video_preds" / f"{trial_name}_{cam_name}_labeled.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_number} from video: {video_path}")
    
    # Save the extracted frame as an image

    frame_output_path = save_dir / f"{trial_name}_{cam_name}_labeled_f{frame_number}.png"
    cv2.imwrite(str(frame_output_path), frame)
    cap.release()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    session_names = ["2025-08-19", "2025-08-22", "2025-11-20",
                      "2025-12-08", "2025-12-09", "2025-12-18"]
    
    df_all_list = []
    for session_name in session_names:
        feature_path = ANALYSIS_ROOT / session_name / "hand" / f"hand_features_{session_name}_f{FRAME_NUMBER}.pkl" 
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                df_all_list.append(pd.read_pickle(f))

    # Load feature names from the first session metadata
    meta_path = ANALYSIS_ROOT / session_names[0] / "hand" / f"hand_feature_names_{session_names[0]}_f{FRAME_NUMBER}.pkl"
    with open(meta_path, 'rb') as f:
        feature_names = pickle.load(f)

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Filtering
    df_filtered = df_all[df_all["correct"] == True].copy()
    if "short" in TRIAL_TYPE: df_filtered = df_filtered[df_filtered["is_holdshort"]]
    ori_suffixes = tuple(f"_{ori}" for ori in ORIENTATION_LIST)
    df_filtered = df_filtered[df_filtered["shape_id"].str.endswith(ori_suffixes)]

    # Calculations
    feature_cols = [col for col in feature_names if col in df_filtered.columns]
    df_filtered[feature_cols] = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')

    df_avg = df_filtered.groupby("shape_id")[feature_cols].mean().reset_index()
    df_merged = df_filtered.merge(df_avg, on="shape_id", suffixes=('', '_avg'))
    
    # Calculate Euclidean distance to centroid
    diffs = df_merged[[f"{c}_avg" for c in feature_cols]].values - df_merged[feature_cols].values
    df_merged['trial_variability_score'] = np.sqrt(np.sum(diffs**2, axis=1))

    # Identify most variable shapes
    variability_by_shape = df_merged.groupby("shape_id")['trial_variability_score'].mean().sort_values(ascending=False)
    
    # Save variability scores
    variability_by_shape.to_csv(SAVE_DIR / "shape_variability_scores.csv")

    # Plot variability scores
    plt.figure(figsize=(10, 6))
    plt.hist(variability_by_shape.values, bins=20, alpha=0.7)
    plt.xlabel("Variability Score")
    plt.ylabel("Number of Shapes")
    plt.title("Distribution of Variability of Hand Features by Shape")
    plt.savefig(SAVE_DIR / "variability_distribution.png")
    
    # Report generation for top 5 most variable shapes
    print("Generating variability driver reports for top 5 shapes...")
    for shape_id in variability_by_shape.head(10).index:
        print(f"{shape_id}")
        subset = df_merged[df_merged["shape_id"] == shape_id]
        residuals = subset[feature_cols].values - subset[[f"{c}_avg" for c in feature_cols]].values
        
        # PCA to find driving features
        pca = PCA(n_components=1)
        pca.fit(residuals)
        loadings = pd.Series(np.abs(pca.components_[0]), index=feature_cols).sort_values(ascending=False)
        
        # Save report
        loadings.to_csv(SAVE_DIR / f"drivers_{shape_id}.csv")

        # Get example frames with highest variability
        top_indices = subset['trial_variability_score'].nlargest(3).index
        example_trials = subset.loc[top_indices, ['trial_name_video']]
        for trial_name in example_trials['trial_name_video']:
            session_name = trial_name.split('_')[0]

            for cam_name in CAMERA_NAMES:
                example_frame_dir = SAVE_DIR / f"{shape_id}_examples"
                example_frame_dir.mkdir(parents=True, exist_ok=True)
                get_frame_from_video(session_name, trial_name, cam_name, FRAME_NUMBER, example_frame_dir)
        
    print(f"Analysis complete. Reports saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()