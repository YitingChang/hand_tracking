import pandas as pd
from pathlib import Path

# --- Settings ---
cam_check = "camTo"   # Change this to the camera you are debugging (e.g., camTR, camTL)
trial_name = "2025-11-20_07-58-20"
file_name = f"{trial_name}_{cam_check}.csv"

analysis_dir = Path("/media/yiting/NewVolume/Analysis")
session_name = "2025-11-20"
pred_file_path = analysis_dir / session_name / "litpose" / "video_preds" / file_name

OBJECT_KPTs = [
    "Dot_t1", "Dot_t2", "Dot_t3", "Dot_b1", "Dot_b2", "Dot_b3",
    "Dot_l1", "Dot_l2", "Dot_l3", "Dot_r1", "Dot_r2", "Dot_r3" 
]

# --- Load Data ---
print(f"Loading {file_name}...")
df = pd.read_csv(pred_file_path, header=[1, 2], index_col=0)

# --- Slice Frames 300-400 ---
subset = df.loc[300:400]

print(f"\n--- DIAGNOSTICS: {cam_check} (Frames 300-400) ---")

if subset.empty:
    print("Error: No data found in frames 300-400.")
else:
    # 1. Check each keypoint individually
    print(f"{'Keypoint':<10} | {'Conf (Avg)':<10} | {'X Range':<15} | {'Y Range':<15}")
    print("-" * 60)
    
    valid_x_all = []
    valid_y_all = []

    for kpt in OBJECT_KPTs:
        # Check if this keypoint exists in the CSV headers
        if (kpt, "x") not in subset.columns:
            print(f"{kpt:<10} | {'NOT FOUND':<10}")
            continue
            
        # Get data for this dot
        x_vals = subset[(kpt, "x")]
        y_vals = subset[(kpt, "y")]
        conf_vals = subset[(kpt, "likelihood")]
        
        # Calculate stats (ignoring NaNs)
        avg_conf = conf_vals.mean()
        min_x, max_x = x_vals.min(), x_vals.max()
        min_y, max_y = y_vals.min(), y_vals.max()
        
        # Store for global ROI calculation later
        if not pd.isna(min_x):
            valid_x_all.extend([min_x, max_x])
            valid_y_all.extend([min_y, max_y])

        # Print row
        # If data is all NaN, handle gracefully
        if pd.isna(avg_conf):
            print(f"{kpt:<10} | {'ALL NaN':<10} | {'-':<15} | {'-':<15}")
        else:
            print(f"{kpt:<10} | {avg_conf:<10.4f} | {int(min_x)}-{int(max_x):<9} | {int(min_y)}-{int(max_y):<9}")

    # 2. Suggest ROI based on ALL visible dots
    if valid_x_all:
        global_min_x, global_max_x = min(valid_x_all), max(valid_x_all)
        global_min_y, global_max_y = min(valid_y_all), max(valid_y_all)
        
        padding = 30
        print("\n--- SUGGESTED ROI (All dots combined) ---")
        print(f'"x": [{int(global_min_x - padding)}, {int(global_max_x + padding)}]')
        print(f'"y": [{int(global_min_y - padding)}, {int(global_max_y + padding)}]')
    else:
        print("\nAll keypoints are NaN in this range. Check if the object is visible.")