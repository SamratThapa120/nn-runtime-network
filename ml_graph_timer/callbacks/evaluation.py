

import torch
from tqdm import tqdm
import os
import numpy as np
from allrank.models.losses import listMLE,lambdaLoss
from scipy import stats

def ordered_pair_accuracy(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Shapes of ground truth and prediction must be the same"
    
    batch_size, n_values = y_true.shape
    i, j = torch.triu_indices(n_values, n_values, offset=1)
    y_true_i, y_true_j = y_true[:, i], y_true[:, j]
    y_pred_i, y_pred_j = y_pred[:, i], y_pred[:, j]
    true_order = y_true_i > y_true_j
    pred_order = y_pred_i > y_pred_j
    correct_pairs = (true_order == pred_order).sum().float()
    total_pairs = batch_size * len(i)
    opa = correct_pairs / total_pairs
    return opa

def kendalltau(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Shapes of ground truth and prediction must be the same"
    
    return np.mean([stats.kendalltau(a,b).correlation for a,b in zip(y_true,y_pred)])

class ModelValidationCallback:
    def __init__(self,model,metrics,valid_loader):
        self.model = model
        self.metrics = metrics
        self.valid_loader = valid_loader
        self.opa = -1
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.model.get_state_dict(),
        }, path)
    #@profile
    def __call__(self, current_step):
        self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"latest_model.pkl"))
        truths = []
        predictions = []
        loss = 0
        batch_count = 0
        for batch in tqdm(self.valid_loader,desc=f"Valid step: {current_step}"):
            generated_tokens = self.model.infer(batch)
            loss += listMLE(generated_tokens,batch["config_runtimes"]).item()
            batch_count +=1
            truths.append(batch["config_runtimes"])
            predictions.append(generated_tokens)

        opa = ordered_pair_accuracy(torch.concat(truths,0),torch.concat(predictions,0)).item()
        self.metrics(current_step,"ordered_pair_accuracy",opa)
        self.metrics(current_step,"kendall_tau",kendalltau(torch.concat(truths,0).numpy(),torch.concat(predictions,0).numpy()))
        self.metrics(current_step,"valid_loss",loss/batch_count)



        if opa>=self.opa:
            print(f"saving best model. opa improved from {self.opa} to {opa}")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_opa.pkl"))
            self.opa = opa


            
