from pathlib import Path
import numpy as np
import pandas as pd
import shap
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration ---
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"

# Condition setting
ALEXNET_LAYER = 'mid'  # Options: 'low', 'mid', 'high'
IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
HAND_TARGET_FEATURE = "pair_Small_MCP_Thumb_MCP"

# --- 1. PREPARE DATA ---
# Load AlexNet feature vectors (X) and hand tracking dfeatures (y)
# X: AlexNet features (e.g., [N_shapes, 24576] for concatenated views)
# y: A specific hand metric (e.g., Thumb-Index aperture or first PC of hand posture)

alexnet_fname = f"alexnet_{ALEXNET_LAYER}_features_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.npy"
alexnet_save_path = SHAPE_RDM_SAVE_DIR / alexnet_fname
X = np.load(alexnet_save_path)

hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
hand_feat_path = HAND_RDM_SAVE_DIR / hand_feat_csv
df_hand = pd.read_csv(hand_feat_path)
y = df_hand[HAND_TARGET_FEATURE]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. TRAIN REGRESSION MODEL ---
# Random Forest is robust for high-dimensional feature attribution
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 3. COMPUTE SHAP VALUES ---
# This identifies which of the 24,576 features contribute most to the grasp
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- 4. VISUALIZE CONTRIBUTIONS ---
# Summary plot shows the top features that drive the hand conformation
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=[f"Feat_{i}" for i in range(X.shape[1])])

# Save the top feature indices for further investigation (e.g., back-mapping to images)
top_inds = np.argsort(np.abs(shap_values).mean(0))[-20:]
print(f"Top 20 AlexNet features driving the hand metric ({HAND_TARGET_FEATURE}): {top_inds}")