

import torch
from tqdm import tqdm
import os
import numpy as np
from allrank.models.losses import listMLE,lambdaLoss
from scipy import stats
from collections import defaultdict


def ordered_pair_accuracy(y_true, y_pred, padding=-1):
    assert y_true.shape == y_pred.shape, "Shapes of ground truth and prediction must be the same"
    
    batch_size, n_values = y_true.shape
    mask = y_true != padding
    i, j = torch.triu_indices(n_values, n_values, offset=1)
    
    correct_pairs = 0
    total_pairs = 0
    for batch in range(batch_size):
        # Get valid indices for this batch where neither i nor j is padded
        valid_indices = mask[batch][i] & mask[batch][j]
        if valid_indices.any():
            y_true_i = y_true[batch, i[valid_indices]]
            y_true_j = y_true[batch, j[valid_indices]]
            y_pred_i = y_pred[batch, i[valid_indices]]
            y_pred_j = y_pred[batch, j[valid_indices]]
            
            true_order = y_true_i > y_true_j
            pred_order = y_pred_i > y_pred_j
            correct_pairs += (true_order == pred_order).float().sum()
            total_pairs += len(y_true_i)
    
    opa = correct_pairs / total_pairs
    return opa

def kendalltau(y_true, y_pred,padding=-1):
    assert y_true.shape == y_pred.shape, "Shapes of ground truth and prediction must be the same"
    
    return np.mean([stats.kendalltau(a[a!=padding],b[a!=padding]).correlation for a,b in zip(y_true,y_pred)])

class ModelValidationCallback:
    def __init__(self,model,metrics,valid_loader,padding=-1):
        self.model = model
        self.metrics = metrics
        self.valid_loader = valid_loader
        self.opa = -1
        self.padding = padding
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.model.get_state_dict(),
        }, path)
    #@profile
    def __call__(self, current_step):
        self._savemodel(current_step, os.path.join(self.model.OUTPUTDIR, "latest_model.pkl"))
        
        # Initializing storage for overall truths and predictions
        truths = []
        predictions = []
        
        
        modeltype = self.valid_loader.dataset.model_types
        loss = 0
        batch_count = 0
        for batch in tqdm(self.valid_loader, desc=f"Valid step: {current_step}"):
            generated_tokens = self.model.infer(batch)
            loss += self.model.criterion(generated_tokens, batch["config_runtimes"]).item()
            batch_count += 1
            truths.append(batch["config_runtimes"])
            predictions.append(generated_tokens)

        truths = torch.concat(truths, 0)
        predictions = torch.concat(predictions, 0)

        opa = ordered_pair_accuracy(truths, predictions,self.padding).item()
        self.metrics(current_step, "ordered_pair_accuracy", opa)
        self.metrics(current_step, "kendall_tau", kendalltau(truths.numpy(), predictions.numpy(),self.padding))
        self.metrics(current_step, "valid_loss", loss / batch_count)
    
        # Calculating ordered_pair_accuracy for each unique modeltype
        for mt in np.unique(modeltype):
            indices = [i for i,m in enumerate(modeltype) if m==mt]
            opa_modeltype = ordered_pair_accuracy(truths[indices], predictions[indices],self.padding).item()
            self.metrics(current_step, f"opa_{mt}", opa_modeltype)

            kt_modeltype = kendalltau(truths[indices].numpy(), predictions[indices].numpy(),self.padding).item()
            self.metrics(current_step, f"ktau_{mt}", kt_modeltype)

        if opa>=self.opa:
            print(f"saving best model. opa improved from {self.opa} to {opa}")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_opa.pkl"))
            self.opa = opa


            
