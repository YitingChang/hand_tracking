import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Settings ---
CAMERA_VIEWS = ["To", "TL", "TR", "BL", "BR"]

OBJECT_KPTs = [
    "Dot_t1", "Dot_t2", "Dot_t3", "Dot_b1", "Dot_b2", "Dot_b3",
    "Dot_l1", "Dot_l2", "Dot_l3", "Dot_r1", "Dot_r2", "Dot_r3" 
]

# Color
cmap = plt.get_cmap('jet', len(OBJECT_KPTs)) # 'jet', 'hsv', or 'viridis' work well

KPT_COLORS = {}
for i, kpt in enumerate(OBJECT_KPTs):
    rgba = cmap(i)
    # Convert RGBA (0-1) to BGR (0-255) for OpenCV
    b = int(rgba[2] * 255)
    g = int(rgba[1] * 255)
    r = int(rgba[0] * 255)
    KPT_COLORS[kpt] = (b, g, r)

DOT_RADIUS = 5
THICKNESS = -1           # -1 fills the circle

# Legend Settings
LEGEND_X = 20          # Starting X position
LEGEND_Y = 30          # Starting Y position
LINE_HEIGHT = 25       # Vertical spacing between items
FONT_SCALE = 0.6
TEXT_COLOR = (255, 255, 255) # White text


# --- Funcitons ---
def get_labeled_video(pred_path, video_path, output_path):

    # --- Load predictions ---
    df = pd.read_csv(pred_path, header=[0, 1], index_col=0)

    # --- Video Processing ---
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID' for .avi
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        # A. Draw Predictions
        if frame_idx in df.index:
            for kpt in OBJECT_KPTs:
                x = df.loc[frame_idx, (kpt, "x")]
                y = df.loc[frame_idx, (kpt, "y")]
                
                if not np.isnan(x) and not np.isnan(y):
                    center = (int(x), int(y))
                    cv2.circle(frame, center, DOT_RADIUS, KPT_COLORS[kpt], THICKNESS)

        # B. Draw Legend
        # (Optional: Draw a semi-transparent background box for readability)
        # overlay = frame.copy()
        # cv2.rectangle(overlay, (5, 5), (150, 30 + len(OBJECT_KPTs)*LINE_HEIGHT), (0, 0, 0), -1)
        # cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        for i, kpt in enumerate(OBJECT_KPTs):
            # Calculate position for this item
            y_pos = LEGEND_Y + (i * LINE_HEIGHT)
            
            # 1. Draw color swatch (small rectangle)
            color = KPT_COLORS[kpt]
            cv2.rectangle(frame, (LEGEND_X, y_pos - 10), (LEGEND_X + 20, y_pos + 5), color, -1)
            
            # 2. Draw Text
            cv2.putText(frame, kpt, (LEGEND_X + 30, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Video saved to: {output_path}")

def main():
    analysis_dir = Path("/media/yiting/NewVolume/Analysis")
    session_name = "2025-11-20"
    trial_name = "2025-11-20_07-58-20"

    lp_dir = analysis_dir / session_name / "litpose"
    lp_2d_filter_dir = lp_dir / "lp_2d_filter"
    video_dir = lp_dir / "new_videos"

    output_dir = lp_dir / "videos_filtered_2d"
    os.makedirs(output_dir, exist_ok=True)

    for cam_view in CAMERA_VIEWS:
        # --- Get prediction and video paths ---
        # Prediction (.csv)
        pred_name = trial_name + "_cam" + cam_view + "_filtered_interp_smooth.csv"
        pred_path = lp_2d_filter_dir / pred_name

        # Video (.mp4)
        video_name = trial_name + "_cam" + cam_view + ".mp4"
        video_path = video_dir / video_name
        output_path = output_dir / f"{video_name.replace('.mp4', '_f2d.mp4')}"

        get_labeled_video(pred_path, video_path, output_path)

if __name__ == "__main__":
    main() 


