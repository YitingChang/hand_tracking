import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- CONFIGURATION ---
IMAGE_DIR = '/media/yiting/NewVolume/Data/Shapes/shapes_2026'
PROCESSED_DIR = '/media/yiting/NewVolume/Data/Shapes/shapes_2026_alexnet_preprocessed'
OUTPUT_CSV = '/media/yiting/NewVolume/Analysis/hand_conformation/alexnet_tsne_results.csv'
# Crop logic (top-left origin)
# PIL Crop: (left, top, right, bottom)
CROP_COORDS = (75, 250, 550, 725) 

def preprocess_all_images(input_dir, output_dir):
    """Stage 1: Clean, crop, and save images locally."""
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

def extract_and_project(processed_dir):
    """Stage 2: Run AlexNet features -> PCA -> t-SNE."""
    paths = sorted(glob(os.path.join(processed_dir, '*.png')))
    
    # Load Model (Truncated to first FC layer)
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:2])
    model.eval()

    # Pre-processing for the model (Resize + Normalize)
    model_pipe = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # AlexNet needs 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    print("Extracting features...")
    for path in tqdm(paths):
        img = Image.open(path).convert('RGB')
        tensor = model_pipe(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
            features.append(feat.numpy().flatten())
    
    features = np.array(features)
    
    # Dimensionality Reduction
    print("Running PCA and t-SNE...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=50)
    pca_res = pca.fit_transform(scaled)
    
    tsne = TSNE(
            n_components=2, 
            perplexity=min(30, len(pca_res) - 1), 
            max_iter=1000, 
            random_state=42
        )
    tsne_res = tsne.fit_transform(pca_res)
    
    return tsne_res, paths

def plot_tsne(csv_path, save_path=None, img_overlay = False, image_column='OriginalPath', zoom=0.05):
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('AlexNet Feature t-SNE: Shape Clustering', fontsize=15)
    
    x = df['TSNE1'].values
    y = df['TSNE2'].values
    paths = df[image_column].values

    # 1. Plot the background points
    ax.scatter(x, y, alpha=0.3, c='royalblue')

    if img_overlay:
        # 2. Add image overlays
        # Note: We'll plot a subset if there are thousands to keep it readable
        step = max(1, len(df) // 50) 
        
        for i in range(0, len(df), step):
            try:
                img = Image.open(paths[i]).convert('RGB')
                img = img.crop(CROP_COORDS)
                
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x[i], y[i]), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                continue

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path)

def main():
    # 1. Pre-process
    preprocess_all_images(IMAGE_DIR, PROCESSED_DIR)
    
    # 2. Extract and Project
    tsne_results, final_paths = extract_and_project(PROCESSED_DIR)
    
    # 3. Save
    df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df['OriginalPath'] = [os.path.join(IMAGE_DIR, os.path.basename(p)) for p in final_paths]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")

    # 4. Plot
    plot_tsne(OUTPUT_CSV, save_path=OUTPUT_CSV.replace('results.csv', 'plot.png'), img_overlay=False)
    plot_tsne(OUTPUT_CSV, save_path=OUTPUT_CSV.replace('results.csv', 'plot_with_images.png'), img_overlay=True, zoom=0.05)

if __name__ == '__main__':
    main()