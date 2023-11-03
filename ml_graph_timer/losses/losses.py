import torch 
from allrank.models.losses import listMLE

class CustomMAELoss:
    def __init__(self,padding=-1):
        self.padding = padding
        self.lossf = torch.nn.L1Loss(reduction="mean")
    
    def __call__(self,truth,pred):
        mask = truth !=self.padding
        return self.lossf(truth[mask],pred[mask])

class CustomMSELoss:
    def __init__(self,padding=-1):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
    
    def __call__(self,truth,pred):
        mask = truth !=self.padding
        return self.lossf(truth[mask],pred[mask])

class ListMLEwMSELoss:
    def __init__(self,padding=-1,mse_weight=1):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
        self.mse_weight = mse_weight
    def __call__(self,truth,pred):
        mask = truth !=self.padding
        return torch.mean(torch.abs(truth[mask]-pred[mask])/(truth[mask]))*self.mse_weight + listMLE(pred,truth)