import torch 

class CustomMAELoss:
    def __init__(self,padding=-1):
        self.padding = padding
        self.lossf = torch.nn.L1Loss(reduction="mean")
    
    def __call__(self,truth,pred):
        mask = truth !=self.padding
        return self.lossf(truth[mask],pred[mask])
