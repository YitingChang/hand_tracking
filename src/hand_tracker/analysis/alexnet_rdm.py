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
import pickle

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
IMAGE_ORI0_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori0"
IMAGE_ORI2_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori2"

SHAPE_RDM_SAVE_DIR = ANALYSIS_ROOT / "shape_analysis" / "shape_rdms"
HAND_RDM_SAVE_DIR = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms"
IMAGE_TYPE = 'depth'  # Options: 'rgb' or 'depth'
VIEW_ORDER = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']

TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
os.makedirs(SHAPE_RDM_SAVE_DIR, exist_ok=True)

class FeatureExtractor:
    def __init__(self, layer_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(self.device).eval()
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
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): 
            self.model(img_t)
        return self.features

def extract_concatenated_shape_features(layer_id):
    extractor = FeatureExtractor(layer_id)
    
    # 1. Load hand data to get the EXACT order of shape_ids
    hand_feat_csv = f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
    
    hand_feat_path = HAND_RDM_SAVE_DIR / hand_feat_csv
    df_hand = pd.read_csv(hand_feat_path)
    valid_ids = df_hand['shape_id'].astype(str).str.strip().tolist() # Keep order
    
    final_features = []
    final_shape_ids = []
    
    # Cache features by the full unique ID (e.g., A013_0)
    cache = {}

    print(f"Extracting {layer_id} features for {len(valid_ids)} conditions...")
    for full_id in tqdm(valid_ids):
        if full_id in cache:
            final_features.append(cache[full_id])
            final_shape_ids.append(full_id)
            continue

        # Split A013_0 -> base='A013', ori='0'
        parts = full_id.split('_')
        base_id = parts[0]
        ori_suffix = parts[1]

        # Determine which image directory to use
        # Ori 0 and 02 (symmetric) use Ori0 folder; Ori 2 uses Ori2 folder
        current_img_dir = IMAGE_ORI2_DIR if ori_suffix == '2' else IMAGE_ORI0_DIR
        
        obj_views = []
        first_feat = None
        
        for view in VIEW_ORDER:
            # Construct filename: e.g., A013_Front_rgb.png
            img_name = f"{base_id}_{view}_{IMAGE_TYPE}.png"
            img_path = current_img_dir / img_name
            
            feat = extractor.get_features(img_path)
            obj_views.append(feat)
            if feat is not None and first_feat is None:
                first_feat = feat
        
        # Handle missing data or padding
        if first_feat is not None:
            feat_dim = len(first_feat)
            # Fill missing views with zeros if necessary
            obj_views = [f if f is not None else np.zeros(feat_dim) for f in obj_views]
            concatenated = np.concatenate(obj_views)
            
            cache[full_id] = concatenated
            final_features.append(concatenated)
            final_shape_ids.append(full_id)
        else:
            print(f"Warning: No images found for {full_id} in {current_img_dir}")

    return np.array(final_features), final_shape_ids

def compute_and_save_rdms():
    target_layers = {'low': 'features.0', 'mid': 'features.10', 'high': 'classifier.1'}
    results = {}

    for label, layer_id in target_layers.items():
        features, shape_ids = extract_concatenated_shape_features(layer_id)
        
        if len(features) == 0:
            print(f"Error: No features extracted for {label}.")
            continue
            
        print(f"Computing {label} RDM (Matrix size: {len(shape_ids)}x{len(shape_ids)})...")
        # Use correlation distance (1 - pearson)
        rdm_matrix = squareform(pdist(features, metric='correlation'))
        
        results[label] = {
            'rdm': rdm_matrix, 
            'shape_ids': shape_ids, 
            'layer': layer_id,
            'image_type': IMAGE_TYPE
        }

    # Save filename reflects the orientation setup
    ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"
    save_name = f'alexnet_rdms_concatenated_{IMAGE_TYPE}_{TRIAL_TYPE}_{ori_str}.pkl'
    save_path = SHAPE_RDM_SAVE_DIR / save_name
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Successfully saved RDMs to: {save_path}")

if __name__ == "__main__":
    compute_and_save_rdms()