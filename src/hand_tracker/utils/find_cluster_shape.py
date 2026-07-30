import os
from glob import glob
import pandas as pd

'''
1. Load cluster data from CSV files in a specified directory.
2. For each trial (row) in the cluster data:
   a. Extract the shape name, orientation, and cluster id.
   b. Find the corresponding shape image file based on the shape name.
   c. Rename the shape image file to include the orientation information.
   d. Copy the shape image file to a new directory named after the cluster id.

'''

def find_and_copy_shape_images(cluster_data_path, shape_images_dir, output_dir):
    # Load cluster data from CSV
    cluster_data = pd.read_csv(cluster_data_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in cluster_data.iterrows():
        shape_type = row['shape_type']
        cluster_id = row['cluster']
        shape_id = row['shape_id']

        # Find corresponding shape image file
        shape_image_file = os.path.join(shape_images_dir, f"{shape_type}_A.png")

        if not os.path.exists(shape_image_file):
            print(f"No image found for shape: {shape_type}")
            continue

        # Create cluster directory if it doesn't exist
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Copy shape image to the cluster directory
        new_filename = f"{shape_id}.png"
        destination_file = os.path.join(cluster_dir, new_filename)
        os.system(f'cp "{shape_image_file}" "{destination_file}"')

        print(f"Copied {shape_image_file} to {destination_file}")

if __name__ == "__main__":
    cluster_data_path = '/media/yiting/NewVolume/Analysis/hand_conformation/hand_conf_correct_f300_pca_tsne_perplexity30.csv'
    shape_images_dir = '/media/yiting/NewVolume/Data/Shapes/shapes_2026'
    output_dir = '/media/yiting/NewVolume/Analysis/hand_conformation/cluster_shape_images_tsne30'

    find_and_copy_shape_images(cluster_data_path, shape_images_dir, output_dir)