import os
import sys
import glob
import numpy as np
from tqdm import tqdm

def calculate_and_save_normalizers(save_path):
    # Define the path where you want to save the TensorFlow tensors
    normalizers_path = os.path.join(save_path, "normalizers")

    # Ensure the directory for saving normalizers exists
    os.makedirs(normalizers_path, exist_ok=True)

    # Initialize lists to store the maximum and minimum values
    max_node_feat = []
    min_node_feat = []
    max_node_config_feat = []
    min_node_config_feat = []

    # Iterate through the files in the "train" directory
    for f in tqdm(glob.glob(os.path.join(save_path, "train/*.npz"))):
        info = np.load(f)
        max_node_feat.append(info["node_feat"].max(0))
        min_node_feat.append(info["node_feat"].min(0))
        max_node_config_feat.append(info["node_config_feat"].max(0))
        min_node_config_feat.append(info["node_config_feat"].min(0))

    # Save the TensorFlow tensors to the "normalizers" directory
    np.save(normalizers_path,{
        "max_node_feat": np.vstack(max_node_feat).max(0),
        "min_node_feat": np.vstack(min_node_feat).min(0),
        "max_node_config_feat": np.vstack(max_node_config_feat).max(0),
        "min_node_config_feat": np.vstack(min_node_config_feat).min(0)
    })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_normalizers.py <SAVE_PATH>")
        sys.exit(1)

    save_path = sys.argv[1]
    calculate_and_save_normalizers(save_path)
