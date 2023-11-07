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
from configs.listmle_gsage_nlp import Configs
from ml_graph_timer.callbacks.evaluation import ordered_pair_accuracy


feature_vec_name = {int(i):v for i,v in json.load(open("/app/nn-runtime-network/assets/node_feature_vector.json")).items()}

CFG = Configs()

dataloader = DataLoader(CFG.valid_dataset, batch_size=4, shuffle=False, collate_fn=CFG.dataloder_collate_val,num_workers=8,pin_memory=False)

CFG.load_state_dict(os.path.join(CFG.OUTPUTDIR,"latest_model.pkl"))
model = CFG.model
model.eval()
model.cuda()


feature_conf_name = {int(i):v for i,v in json.load(open("/app/nn-runtime-network/assets/layout_config_feature.json")).items()}

truths = []
predictions = []
with torch.no_grad():
    for batch in dataloader:
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

print("original opa:",opa)

opas =[]
index=0
for i,proceed in tqdm(enumerate(CFG.train_dataset.node_config_feat_norms[0])):
    if not proceed:
        opas.append(-1)
        continue
    truths = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            replace = CFG.train_dataset.node_config_feat_norms[1][0,0,index]
            batch["node_config_features"][:,index]= replace.item()
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
    opas.append(opa)
    index+=1

opas = {s:i for i,s in zip(opas,feature_conf_name.values())}
filename = os.path.join(CFG.OUTPUTDIR,'node_conf_importances.json')

with open(filename, 'w') as f:
    json.dump(opas, f, indent=4)

    
opas =[]
index=0
for i,proceed in tqdm(enumerate(CFG.train_dataset.node_feat_norms[0])):
    if not proceed:
        opas.append(-1)
        continue
    truths = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            replace = CFG.train_dataset.node_feat_norms[1][0,index]
            batch["node_features"][:,index]= replace.item()
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
    opas.append(opa)
    index+=1

opas = {s:i for i,s in zip(opas,feature_vec_name.values())}
filename = os.path.join(CFG.OUTPUTDIR,'node_feature_importances.json')

with open(filename, 'w') as f:
    json.dump(opas, f, indent=4)
