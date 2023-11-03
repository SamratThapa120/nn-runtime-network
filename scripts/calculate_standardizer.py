import os
import sys
import glob
import numpy as np
from tqdm import tqdm

def calculate_and_save_normalizers(save_path):
    # Define the path where you want to save the normalizers
    normalizers_path = os.path.join(save_path, "normalizers_std")

    # Ensure the directory for saving normalizers exists
    os.makedirs(normalizers_path, exist_ok=True)

    # Initialize counters and accumulators for sum and squared sum
    count_node_feat = 0
    count_node_config_feat = None
    sum_node_feat = None
    sum_squared_node_feat = None
    sum_node_config_feat = None
    sum_squared_node_config_feat = None

    # Iterate through the files in the "train" directory
    for f in tqdm(glob.glob(os.path.join(save_path, "train/*.npz"))):
        info = np.load(f)

        # Convert data to float128 for higher precision
        node_feat = info["node_feat"].astype(np.float128)
        node_config_feat = info["node_config_feat"].reshape(-1,18).astype(np.float128)

        # Update counters and accumulators for node_feat
        count_node_feat += node_feat.shape[0]
        if sum_node_feat is None:
            sum_node_feat = np.sum(node_feat, axis=0)
            sum_squared_node_feat = np.sum(node_feat**2, axis=0)
        else:
            sum_node_feat += np.sum(node_feat, axis=0)
            sum_squared_node_feat += np.sum(node_feat**2, axis=0)

        # Exclude -1 values for node_config_feat and update counters
        mask = node_config_feat != -1
        if count_node_config_feat is None:
            count_node_config_feat = mask.sum(axis=0)
        else:
            count_node_config_feat += mask.sum(axis=0)

        # Update accumulators for node_config_feat with valid data
        valid_node_config_feat = node_config_feat * mask  # Set padding values to 0
        if sum_node_config_feat is None:
            sum_node_config_feat = np.sum(valid_node_config_feat, axis=0)
            sum_squared_node_config_feat = np.sum(valid_node_config_feat**2, axis=0)
        else:
            sum_node_config_feat += np.sum(valid_node_config_feat, axis=0)
            sum_squared_node_config_feat += np.sum(valid_node_config_feat**2, axis=0)

    # Calculate mean and standard deviation
    mean_node_feat = sum_node_feat / count_node_feat
    std_node_feat = np.sqrt(sum_squared_node_feat / count_node_feat - mean_node_feat**2)
    mean_node_config_feat = sum_node_config_feat / count_node_config_feat
    std_node_config_feat = np.sqrt(sum_squared_node_config_feat / count_node_config_feat - mean_node_config_feat**2)

    # Save the computed values
    np.save(os.path.join(normalizers_path, "normalizers_std"), {
        "mean_node_feat": mean_node_feat,
        "std_node_feat": std_node_feat,
        "mean_node_config_feat": mean_node_config_feat,
        "std_node_config_feat": std_node_config_feat
    })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_standardizer.py <SAVE_PATH>")
        sys.exit(1)

    save_path = sys.argv[1]
    calculate_and_save_normalizers(save_path)
