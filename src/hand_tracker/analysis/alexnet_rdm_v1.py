import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from glob import glob
import pickle

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
IMAGE_ORI0_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori0"
IMAGE_ORI2_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori2"

SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
IMAGE_TYPE = 'rgb'  # Options: 'rgb' or 'depth'
VIEW_ORDER = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']

TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 

os.makedirs(SHAPE_RDM_SAVE_DIR, exist_ok=True)

class FeatureExtractor:
    def __init__(self, layer_name):
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
        self.layer_name = layer_name
        self.features = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def hook(self, module, input, output):
        self.features = output.detach().cpu().numpy().flatten()

    def get_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0)
        with torch.no_grad(): self.model(img_t)
        return self.features

def extract_concatenated_shape_features(layer_id):
    extractor = FeatureExtractor(layer_id)
    
    # 1. Load valid IDs from CSV (e.g., A013_0, A013_2, A04_02, etc.)
    
    hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}.csv"
    if len(ORIENTATION_LIST) < 3: 
        hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_ori{ORIENTATION_LIST[0]}.csv"
    hand_feat_path = HAND_RDM_SAVE_DIR / hand_feat_csv
    df_hand = pd.read_csv(hand_feat_path)
    valid_ids = df_hand['shape_id'].astype(str).str.strip().unique().tolist()
    
    # 2. Map Image Files to Base IDs
    all_ori0_images = glob(os.path.join(IMAGE_ORI0_DIR, f"*_{IMAGE_TYPE}.png"))
    all_ori2_images = glob(os.path.join(IMAGE_ORI2_DIR, f"*_{IMAGE_TYPE}.png"))
    all_images = all_ori0_images + all_ori2_images
    base_to_views = {}
    for p in all_images:
        fname = os.path.basename(p)
        parts = fname.split('_')
        base_id = parts[0] # "A013"
        view = parts[1]    # "Front"
        if base_id not in base_to_views: base_to_views[base_id] = {}
        base_to_views[base_id][view] = p

    cache = {}
    final_features = []
    final_shape_ids = []

    for full_id in tqdm(valid_ids, desc=f"Extracting {layer_id}"):
        base_id = full_id.split('_')[0]
        if base_id not in base_to_views: continue
            
        if base_id not in cache:
            obj_views = []
            first_feat = None
            for view in VIEW_ORDER:
                if view in base_to_views[base_id]:
                    feat = extractor.get_features(base_to_views[base_id][view])
                    obj_views.append(feat)
                    if first_feat is None: first_feat = feat
                else: obj_views.append(None)
            
            if first_feat is not None:
                feat_dim = len(first_feat)
                obj_views = [f if f is not None else np.zeros(feat_dim) for f in obj_views]
                cache[base_id] = np.concatenate(obj_views)
        
        if base_id in cache:
            final_features.append(cache[base_id])
            final_shape_ids.append(full_id)

    return np.array(final_features), final_shape_ids

def compute_and_save_rdms():
    target_layers = {'low': 'features.0', 'mid': 'features.10', 'high': 'classifier.1'}
    results = {}

    for label, layer_id in target_layers.items():
        features, shape_ids = extract_concatenated_shape_features(layer_id)
        if len(features) == 0:
            print(f"Warning: No features extracted for {label}. Skipping...")
            continue
            
        print(f"Computing {label} RDM (Shape: {features.shape})...")
        rdm_matrix = squareform(pdist(features, metric='correlation'))
        results[label] = {'rdm': rdm_matrix, 'shape_ids': shape_ids, 'layer': layer_id}

    save_path = os.path.join(SHAPE_RDM_SAVE_DIR, f'alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_ori{ORIENTATION_LIST[0]}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    compute_and_save_rdms()