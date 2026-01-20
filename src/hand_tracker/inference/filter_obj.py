import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# --- Settings ---
CAMERA_VIEWS = ["To", "TL", "TR", "BL", "BR"]

# We can use "obj_roi_test.py" to get ROIs
ROIs = {
    "camTo": {
        "x": [0, 210],
        "y": [525, 650]
    },
    "camTR": {
        "x": [1000, 1200],
        "y": [375, 500]
    },
    "camTL": {
        "x": [0, 250],
        "y": [330, 450]
    },
    "camBR": {
        "x": [100, 240],
        "y": [520, 620]
    },
    "camBL": {
        "x": [850, 1010],
        "y": [730, 830]
    }
}

OBJECT_KPTs = [
    "Dot_t1", "Dot_t2", "Dot_t3", "Dot_b1", "Dot_b2", "Dot_b3",
    "Dot_l1", "Dot_l2", "Dot_l3", "Dot_r1", "Dot_r2", "Dot_r3" 
]

CONF_THR = 0.95

# Interpolation
INTERP_METHOD = 'linear'  # 'linear' is standard; 'spline' or 'time' are alternatives
MAX_GAP_TO_FILL = 5       # Max number of consecutive NaNs to fill. Adjust based on your FPS.

# Smooth
WINDOW_LEN = 9   # Must be ODD. Higher = smoother but might cut corners. (e.g., 5, 7, 9)
POLY_ORDER = 3    # Polynomial order. 2 or 3 is standard for position data.

def filter_obj(df, roi):

    # 1. Filter columns to only keep relevant keypoints
    # mask = df.columns.get_level_values("bodyparts").isin(OBJECT_KPTs)

    # Create an explicit copy to avoid SettingWithCopyWarning
    # df_filtered = df.loc[:, mask].copy()

    df_filtered = df.copy()

    # 2. Iterate through keypoints and apply ROI/Likelihood logic
    for kpt in OBJECT_KPTs:
        # Define column accessors for this keypoint
        x_col = (kpt, "x")
        y_col = (kpt, "y")
        l_col = (kpt, "likelihood")
        
        # Skip if keypoint not found in CSV
        if x_col not in df_filtered.columns:
            continue

        # Create Boolean Masks
        valid_x = (df_filtered[x_col] >= roi["x"][0]) & (df_filtered[x_col] <= roi["x"][1])
        valid_y = (df_filtered[y_col] >= roi["y"][0]) & (df_filtered[y_col] <= roi["y"][1])
        valid_conf = (df_filtered[l_col] >= CONF_THR)

        is_valid_point = valid_x & valid_y & valid_conf

        # Set INVALID points to NaN
        df_filtered.loc[~is_valid_point, [x_col, y_col, l_col]] = np.nan
    
    return df_filtered

def interpolate_obj(df):

    # 1. Apply interpolation
    # We use limit_direction='both' to handle NaNs at the very start or end of the trial if needed,
    # though usually 'forward' is safer for strict causality.
    df_interp = df.interpolate(
        method=INTERP_METHOD, 
        limit=MAX_GAP_TO_FILL, 
        limit_direction='forward'
    )

    # 2. (Optional) Backfill remaining NaNs slightly if you trust the start/end
    # df_interp = df_interp.fillna(method='bfill', limit=2)

    return df_interp

def smooth_obj(df):
    # --- Smooth ---
    df_smoothed = df.copy()

    # Apply filter to each coordinate
    for kpt in OBJECT_KPTs:
        for coord in ['x', 'y']:
            # Get the series
            series = df_smoothed[kpt][coord]
            
            # 1. Identify where data is missing (the "long gaps" we purposely kept)
            is_missing = series.isna()
            
            # 2. Temporarily fill NaNs so savgol_filter allows the math
            # We use 'ffill' + 'bfill' to handle edges, then linear for the rest
            temp_series = series.interpolate(method='linear').ffill().bfill()
            
            # If the series is still containing NaNs (meaning the point was NEVER seen), skip it.
            if temp_series.isna().any():
                continue

            # Check if we have enough data to smooth (series length > window length)
            if len(temp_series) > WINDOW_LEN:
                # 3. Apply Savitzky-Golay Filter
                smoothed_values = savgol_filter(temp_series, window_length=WINDOW_LEN, polyorder=POLY_ORDER)
                
                # 4. Store smoothed values
                df_smoothed.loc[:, (kpt, coord)] = smoothed_values
                
                # 5. Re-apply the NaN mask (restore the long gaps)
                # This ensures we don't invent data where the object was missing for a long time
                df_smoothed.loc[is_missing, (kpt, coord)] = np.nan
    return df_smoothed

def process_trial(trial_name, lp_2d_dir, lp_2d_filter_dir):

    for cam_view in CAMERA_VIEWS:
        
        file_name = trial_name + "_cam" + cam_view + ".csv"
        pred_file_path = lp_2d_dir / file_name

        # --- SAFETY CHECK: Skip if file doesn't exist ---
        if not pred_file_path.exists():
            print(f"Warning: File not found: {pred_file_path}")
            continue
        
        # --- Load Data ---
        try: # Load CSV with MultiIndex headers (Level 0: bodyparts, Level 1: coords)
            df = pd.read_csv(pred_file_path, header=[1, 2], index_col=0)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
        
        camera_name = "cam" + cam_view

        # Check if we have ROIs for this camera
        if camera_name not in ROIs:
             print(f"Warning: No ROI defined for {camera_name}, skipping.")
             continue
        
        roi_cam = ROIs[camera_name]

        # --- Process ---
        df_filtered = filter_obj(df, roi_cam)
        df_interp = interpolate_obj(df_filtered)
        df_smoothed = smooth_obj(df_interp)

        # --- Save ---
        # output_path = lp_2d_filter_dir / f"{file_name.replace('.csv', '_filtered_interp_smooth.csv')}"
        output_path = lp_2d_filter_dir / file_name
        df_smoothed.to_csv(output_path)
        print(f"Saved processed data to: {output_path}")

