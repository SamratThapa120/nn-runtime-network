import torch
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import sys
import glob
import os
import numpy as np
import json
from tqdm import tqdm
sys.path.append("../")
from ml_graph_timer.dataset.layout_dataset import NpzDataset,GraphCollator

from torch.utils.data import DataLoader
from configs.listmle_gsage_xla_fused_embedding import Configs
from ml_graph_timer.callbacks.evaluation import ordered_pair_accuracy,kendalltau


feature_vec_name = {int(i):v for i,v in json.load(open("/app/nn-runtime-network/assets/node_feature_vector.json")).items()}
keep_indices_nf= [  0,   3,   6,   7,  10,  13,  15,  18,  20,  21,  22,  23,  24,
    25,  26,  27,  28,  29,  30, 37,38,  39,  40,  41,  43,  44,  45,  46,  47,  48,  49,  51,  52,
    53,  54,  55,  59,  60,  61,  62,  63,  64,  67,  68,  69,  70,
    71,  72,  73,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
    86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100, 101, 102,
    103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133
]
nf_groups =[31,32,33,34,35,36,134,135,136,137,138,139]
feature_vec_name = {idx:feature_vec_name[i] for idx,i in enumerate(keep_indices_nf+nf_groups)}


CFG = Configs()
CFG.valid_dataset.max_configs = 1024
CFG.valid_dataset.files = CFG.valid_dataset.files
dataloader = DataLoader(CFG.valid_dataset, batch_size=1, shuffle=False, collate_fn=CFG.dataloder_collate_val,num_workers=4,pin_memory=False)

CFG.load_state_dict(os.path.join(CFG.OUTPUTDIR,"bestmodel_opa.pkl"))
model = CFG.model
model.eval()
model.cuda()


feature_conf_name = {int(i):v for i,v in json.load(open("/app/nn-runtime-network/assets/layout_config_feature.json")).items()}

truths = []
predictions = []
modeltype = CFG.valid_dataset.model_types
with torch.no_grad():
    for batch in tqdm(dataloader):
        out = model(batch["node_features"].cuda(), 
                                batch["node_config_features"].cuda(),  
                                batch["node_separation"].cuda(), 
                                batch["node_ops"].cuda(), 
                                batch["edges"].cuda(), 
                                batch["batches"].cuda()
                            ).detach().cpu()
        truths.append(batch["config_runtimes"])
        predictions.append(out)
truths = torch.concat(truths, 0)
predictions = torch.concat(predictions, 0)
opa = ordered_pair_accuracy(truths, predictions,-1).item()
ktau = kendalltau(truths.numpy(),predictions.numpy())
original_metric = [opa,ktau]
print("original opa:",opa)
print("original ktau:",ktau)
# Calculating ordered_pair_accuracy for each unique modeltype
for mt in np.unique(modeltype):
    indices = [i for i,m in enumerate(modeltype) if m==mt]
    opa_modeltype = ordered_pair_accuracy(truths[indices], predictions[indices],-1).item()
    print(f"opa_{mt}:", opa_modeltype)

    kt_modeltype = kendalltau(truths[indices].numpy(), predictions[indices].numpy(),-1).item()
    print(f"ktau_{mt}", kt_modeltype)

# if len(sys.argv)<2 or sys.argv[1]!="1":
#     exit(0)

groups =[
    (9,15),
    (19,24),
    (26,31),
    (33,36),
    (38,42),
    (44,49),
    (51,56),
    (58,61),
    (66,68),
    (70,73),
    (73,74),
    (77,81),
    (81,84),
    (84,88),
    (101,107),
    (107,113),
]
opas = []
names =[]
for i,j in tqdm(groups):

    truths = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch["node_features"][:,i:j]= torch.randint(0,6,batch["node_features"][:, i:j].shape)
            out = model(batch["node_features"].cuda(), 
                                    batch["node_config_features"].cuda(),  
                                    batch["node_separation"].cuda(), 
                                    batch["node_ops"].cuda(), 
                                    batch["edges"].cuda(), 
                                    batch["batches"].cuda()
                                ).detach().cpu()
            truths.append(batch["config_runtimes"])
            predictions.append(out)
    truths = torch.concat(truths, 0)
    predictions = torch.concat(predictions, 0)
    opa = ordered_pair_accuracy(truths, predictions,-1).item()
    ktau = kendalltau(truths.numpy(),predictions.numpy())

    opas.append([opa,ktau])
    names.append(feature_vec_name[i])

opas = {s:i for i,s in zip(opas,names)}
opas["original"] = original_metric
filename = os.path.join(CFG.OUTPUTDIR,'node_importances_group.json')

with open(filename, 'w') as f:
    json.dump(opas, f, indent=4)


    
groups =[
    (0,6),
    (6,12),
    (12,14),
]
opas = []
names =[]
for i,j in tqdm(groups):
    truths = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch["node_config_features"][:,i:j]= torch.randint(0,6,batch["node_config_features"][:, i:j].shape)
            out = model(batch["node_features"].cuda(), 
                                    batch["node_config_features"].cuda(),  
                                    batch["node_separation"].cuda(), 
                                    batch["node_ops"].cuda(), 
                                    batch["edges"].cuda(), 
                                    batch["batches"].cuda()
                                ).detach().cpu()
            truths.append(batch["config_runtimes"])
            predictions.append(out)
    truths = torch.concat(truths, 0)
    predictions = torch.concat(predictions, 0)
    opa = ordered_pair_accuracy(truths, predictions,-1).item()
    ktau = kendalltau(truths.numpy(),predictions.numpy())
    opas.append([opa,ktau])
    names.append(feature_conf_name[i])


opas = {s:i for i,s in zip(opas,names)}
opas["original"] = original_metric
filename = os.path.join(CFG.OUTPUTDIR,'node_conf_importances_group.json')

with open(filename, 'w') as f:
    json.dump(opas, f, indent=4)