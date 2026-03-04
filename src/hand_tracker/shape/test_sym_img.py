from pathlib import Path
import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
IMAGE_ORI0_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori0"
IMAGE_ORI2_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori2"

VIEWS = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
ORIENTATION = 180  # Degrees to rotate around Z-axis (stem).


def compare_images(img_path_a, img_path_b):
    # Load images in grayscale
    img_a = cv2.imread(img_path_a, 0)
    img_b = cv2.imread(img_path_b, 0)

    # Images must be the same size
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    score, diff = ssim(img_a, img_b, full=True)
    return score

# Get object IDs from filenames (assuming format: "objectID_view_type.png")
filenames = [f for f in os.listdir(IMAGE_ORI0_DIR) if f.endswith(f"{VIEWS[0]}_{IMAGE_TYPE}.png")]
object_ids = set(f.split('_')[0] for f in filenames)

results = []
for obj in object_ids:
    for view in VIEWS:
        base_name = f"{obj}_{view}_{IMAGE_TYPE}.png"
        img_path_a = IMAGE_ORI0_DIR / base_name
        img_path_b = IMAGE_ORI2_DIR / base_name
        if img_path_a.exists() and img_path_b.exists():
            similarity_score = compare_images(img_path_a, img_path_b)
            results.append({
                "object_id": obj,
                "similarity_score": similarity_score,
                "view": view
            })
        else:
            print(f"Missing files for {obj} view {view}. Skipping.")


# Save to CSV
df = pd.DataFrame(results)
save_path = DATA_ROOT / "Shapes" / f"symmetry_results_img_ori-{ORIENTATION}.csv"
df.to_csv(save_path, index=False)
print(f"Processing complete. Results saved to {save_path}")

