import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --- CONFIGURATION ---
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
IMAGE_DIR = Path("/media/yiting/NewVolume/Data/Shapes/shapes_2026")
PROCESSED_DIR = Path("/media/yiting/NewVolume/Data/Shapes/shapes_2026_alexnet_preprocessed")
SHAPE_RDM_SAVE_DIR = Path("/media/yiting/NewVolume/Analysis/shape_analysis")
SHAPE_ID_SAVE_PATH = os.path.join(SHAPE_RDM_SAVE_DIR, 'shape_ids.pkl')
HAND_RDM_SAVE_DIR = os.path.join(ANALYSIS_ROOT, "hand_analysis")

# Crop logic (top-left origin)
# PIL Crop: (left, top, right, bottom)
CROP_COORDS = (75, 250, 550, 725) 

def preprocess_all_images(input_dir, output_dir):
    """ Clean, crop, and save images locally."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paths = glob(os.path.join(input_dir, '*.png'))
    
    # Define the transform for saving (no normalization yet)
    pipe = transforms.Compose([
        transforms.Lambda(lambda img: img.crop(CROP_COORDS)),
        transforms.Grayscale(num_output_channels=1), # Save as 1-channel to save space
    ])

    print(f"Pre-processing {len(paths)} images...")
    for path in tqdm(paths):
        img = Image.open(path).convert('RGB')
        processed = pipe(img)
        processed.save(os.path.join(output_dir, os.path.basename(path)))


def extract_features(processed_dir, alexnet_layer='classifier.1'):
    paths = sorted(glob(os.path.join(processed_dir, '*.png')))
    # Extract base IDs (e.g., A013) from filenames (A013_0.png)
    master_list = [os.path.basename(p).split('_')[0] for p in paths]
    
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
    activation = {}
    def hook_fn(m, i, o): activation['feat'] = o.detach()
    
    layer_to_hook = dict([*model.named_modules()])[alexnet_layer]
    layer_to_hook.register_forward_hook(hook_fn)

    model_pipe = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    for path in tqdm(paths, desc=f"Layer {alexnet_layer}"):
        img = Image.open(path).convert('RGB')
        tensor = model_pipe(img).unsqueeze(0)
        model(tensor)
        features_list.append(activation['feat'].numpy().flatten())
    
    return np.array(features_list), master_list

def run_layer_hierarchy_analysis():
    hand_data_path = os.path.join(HAND_RDM_SAVE_DIR, "hand_avg_features_correct.csv")
    
    if not os.path.exists(hand_data_path):
        print(f"\nCRITICAL ERROR: {hand_data_path} not found.")
        print("Please run 'hand_rdm.py' first to generate the trial averages.\n")
        return
    
    # Load Hand shape_ids to ensure alignment
    df_hand = pd.read_csv(hand_data_path).dropna()
    valid_ids = df_hand['shape_id'].tolist()

    target_layers = {'low': 'features.0', 'high': 'features.12', 'global': 'classifier.1'}
    rdm_results = {}

    for label, layer_id in target_layers.items():
        feats, master_list = extract_features(PROCESSED_DIR, layer_id)
        
        # Align: For every 'A251_02' in valid_ids, find index of 'A251' in master_list
        mapping_indices = [master_list.index(sid.split('_')[0]) for sid in valid_ids]
        aligned_feats = feats[mapping_indices]
        
        # Compute RDM
        scaler = StandardScaler()
        feats_norm = scaler.fit_transform(aligned_feats)
        rdm = squareform(pdist(feats_norm, metric='correlation'))
        
        rdm_results[label] = {'layer': layer_id, 'rdm': rdm}
        
    with open(os.path.join(SHAPE_RDM_SAVE_DIR, 'alexnet_rdms_aligned.pkl'), 'wb') as f:
        pickle.dump(rdm_results, f)
    print("AlexNet RDMs Aligned and Saved.")

if __name__ == "__main__":
    # Step 1: Preprocess and save images (only need to run once)
    # preprocess_all_images(IMAGE_DIR, PROCESSED_DIR)

    # Step 2: Extract features and compute RDMs for each layer
    run_layer_hierarchy_analysis()