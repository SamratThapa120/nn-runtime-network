import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import json

def get_unique(array):
    total = array[:,0]
    for i in range(1,array.shape[1]):
        total = total+ (10**i)* array[:,i]
    return total

nf_groups =[
    ("dimension",31,37),
    ("minmaj",134,140),
]
cf_groups =[
    ("output",0,6),
    ("input",6,12),
    ("kernel",12,18),
]
def calculate_and_save_normalizers(save_path):
    # Define the path where you want to save the normalizers
    normalizers_path = os.path.join(save_path, "uniques")

    # Ensure the directory for saving normalizers exists
    os.makedirs(normalizers_path, exist_ok=True)

    uniques_info = {k:set() for k,_,_ in nf_groups+cf_groups}
    counts_info = {k:{} for k,_,_ in nf_groups+cf_groups}
    for f in tqdm(glob.glob(os.path.join(save_path, "train/*.npz"))):
        info = np.load(f)
        for k,i,j in nf_groups:
            uniques,counts = np.unique(get_unique(info["node_feat"][:,i:j]),return_counts=True)
            for u,c in zip(uniques.astype(int).tolist(),counts.tolist()):
                uniques_info[k].add(u)
                if u not in counts_info[k]:
                    counts_info[k][u] = 0
                counts_info[k][u] += c
        for k,i,j in cf_groups:
            uniques,counts = np.unique(get_unique(info["node_config_feat"][:,:,i:j].reshape(-1,6)),return_counts=True)
            for u,c in zip(uniques.astype(int).tolist(),counts.tolist()):
                uniques_info[k].add(u)
                if u not in counts_info[k]:
                    counts_info[k][u] = 0
                counts_info[k][u] += c
                
    uniques_info = {k:list(v) for k,v in uniques_info.items()}

    with open(os.path.join(save_path, "uniques","values.json"), "w") as json_file:
        json.dump(uniques_info, json_file, indent=4)

    with open(os.path.join(save_path, "uniques","counts.json"), "w") as json_file:
        json.dump(counts_info, json_file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_standardizer.py <SAVE_PATH>")
        sys.exit(1)

    save_path = sys.argv[1]
    calculate_and_save_normalizers(save_path)
