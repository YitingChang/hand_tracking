import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
IMAGE_ORI0_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori0"
IMAGE_ORI2_DIR = DATA_ROOT / "Shapes" / "shapes_6views_ori2"

IMAGE_TYPE = 'rgb' 
VIEW_ORDER = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
OVERLAY_VIEW = 'Left' 

TRIAL_TYPE = "correct" 
ORIENTATION_LIST = ['02', '0', '2'] 
ori_str = "all" if len(ORIENTATION_LIST) == 3 else f"ori{ORIENTATION_LIST[0]}"

HAND_FEAT_PATH = ANALYSIS_ROOT / "hand_analysis" / "hand_rdms" / f"hand_avg_features_{TRIAL_TYPE}_{ori_str}.csv"
TSNE_PERPLEXITY = 30

class FeatureExtractor:
    def __init__(self, layer_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(self.device).eval()
        self.layer_name = layer_name
        self.features = None
        
        # Shrinks spatial dimensions (H, W) to (1, 1)
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def hook(self, module, input, output):
        # Handle 4D Conv outputs (Batch, Channels, Height, Width)
        if len(output.shape) == 4:
            # Reduces (1, 256, 13, 13) -> (1, 256, 1, 1)
            pooled = self.pooler(output)
            self.features = pooled.detach().cpu().numpy().flatten()
        else:
            # For 2D Linear layers (like classifier.1)
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
    df_hand = pd.read_csv(HAND_FEAT_PATH)
    valid_ids = df_hand['shape_id'].astype(str).str.strip().tolist()
    
    final_features = []
    final_shape_ids = []
    overlay_image_paths = []
    
    # Store both features and paths in cache to prevent index mismatch
    cache = {} 

    print(f"Extracting {layer_id} features for {len(valid_ids)} conditions...")
    for full_id in tqdm(valid_ids):
        if full_id in cache:
            feat, path = cache[full_id]
            final_features.append(feat)
            overlay_image_paths.append(path)
            final_shape_ids.append(full_id)
            continue

        parts = full_id.split('_')
        base_id, ori_suffix = parts[0], parts[1]
        current_img_dir = IMAGE_ORI2_DIR if ori_suffix == '2' else IMAGE_ORI0_DIR
        
        obj_views = []
        first_feat = None
        overlay_view_path = None

        for view in VIEW_ORDER:
            img_name = f"{base_id}_{view}_{IMAGE_TYPE}.png"
            img_path = current_img_dir / img_name
            
            feat = extractor.get_features(img_path)
            obj_views.append(feat)

            if feat is not None and first_feat is None:
                first_feat = feat
            if view == OVERLAY_VIEW:
                overlay_view_path = str(img_path)
        
        if first_feat is not None:
            feat_dim = len(first_feat)
            obj_views = [f if f is not None else np.zeros(feat_dim) for f in obj_views]
            concatenated = np.concatenate(obj_views)
            
            # Update cache and lists
            cache[full_id] = (concatenated, overlay_view_path)
            final_features.append(concatenated)
            final_shape_ids.append(full_id)
            overlay_image_paths.append(overlay_view_path)
        else:
            print(f"Warning: No images found for {full_id}")

    return np.array(final_features), overlay_image_paths, final_shape_ids

def run_tsne(features):
    print("Running PCA -> t-SNE...")
    scaled = StandardScaler().fit_transform(features)
    n_comp = min(50, features.shape[0])
    pca_res = PCA(n_components=n_comp).fit_transform(scaled)
    tsne_res = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, init='pca', learning_rate='auto').fit_transform(pca_res)
    return tsne_res

def plot_tsne_with_overlay(csv_path, save_path, label, overlay=False, zoom=0.25):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.scatter(df['TSNE1'], df['TSNE2'], alpha=0.3, c='royalblue', s=20)

    if overlay:
        print(f"Adding image overlays for {label}...")
        # Plot every 15th image to avoid overlap
        for i, row in df.iterrows():
            if i % 15 == 0 and pd.notna(row['OverlayPath']):
                if os.path.exists(row['OverlayPath']):
                    img = Image.open(row['OverlayPath']).convert('RGB')
                    imagebox = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(imagebox, (row['TSNE1'], row['TSNE2']), frameon=True, 
                                        bboxprops=dict(edgecolor='black', alpha=0.3))
                    ax.add_artist(ab)
        ax.set_title(f"t-SNE: AlexNet {label} (Concatenated Views, Overlay: {OVERLAY_VIEW})")
    else:
        ax.set_title(f"t-SNE: AlexNet {label} (Concatenated Views)")
    plt.xlabel('tsne-d1')
    plt.ylabel('tsne-d2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    target_layers = {'low': 'features.0', 'mid': 'features.10', 'high': 'classifier.1'}
    
    # Ensure output directory exists
    out_dir = ANALYSIS_ROOT / "shape_analysis" / "alexnet_tsne"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, layer_id in target_layers.items():
        feats, img_paths, shape_ids = extract_concatenated_shape_features(layer_id)
        
        if len(feats) == 0: continue
            
        tsne_results = run_tsne(feats)
        
        df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
        df['shape_id'] = shape_ids
        df['OverlayPath'] = img_paths
        
        csv_path = out_dir / f"alexnet_{label}_tsne_{IMAGE_TYPE}_{ori_str}.csv"
        df.to_csv(csv_path, index=False)
        
        plot_path = str(csv_path).replace('.csv', '.png')
        plot_tsne_with_overlay(csv_path, plot_path, label, overlay=False)
        plot_path = str(csv_path).replace('.csv', '_overlay.png')
        plot_tsne_with_overlay(csv_path, plot_path, label, overlay=True)
        print(f"Finished {label}. Plot saved to {plot_path}")