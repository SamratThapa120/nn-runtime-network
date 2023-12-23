import torch 
from allrank.models.losses import listMLE,rankNet


def sine_between_vectors(u, v):
    # Compute z-component of the cross product (using a trick for 2D vectors)
    cross_product_z = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    
    # Compute magnitudes of u and v
    u_mag = torch.norm(u, dim=1)
    v_mag = torch.norm(v, dim=1)
    
    # Sine of angle between vectors
    sine_theta = cross_product_z / (u_mag * v_mag + 1e-10)  # Adding a small epsilon to avoid division by zero
    
    return sine_theta

def compute_sine_matrix(tensor):
    # Get tensor shape
    batch_size, configs, _ = tensor.shape
    
    # Resultant matrix
    sine_matrix = torch.zeros(batch_size, configs, configs, device=tensor.device, dtype=tensor.dtype)
    
    for i in range(configs):
        for j in range(configs):
            sine_matrix[:, i, j] = sine_between_vectors(tensor[:, i, :], tensor[:, j, :])
            
    return sine_matrix

def pairwise_hinge_loss_batch(pred, true):
    # pred: (batch_size, num_preds )
    # true: (batch_size, num_preds)
    batch_size = pred.shape[0]
    num_preds = pred.shape[1]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = true[:,i_idx] > true[:,j_idx]
    loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[:,i_idx] - pred[:,j_idx])) * pairwise_true.float()) / batch_size
    return loss

def shuffledListMLE(preds,truth,*args,**kwargs):
    b,s = truth.shape
    truth = truth.view(-1)
    preds = preds.view(-1)
    perm_index = torch.randperm(b*s,device=truth.device)
    # Shuffle both tensors using the same permutation index
    truth = truth[perm_index].view(b,s)
    preds = preds[perm_index].view(b,s)

    return listMLE(preds,truth,*args,**kwargs)
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

class SineLoss:
    def __init__(self,padding=-1):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
        self.ignore_idx = -100
    def calculate_pair_gt_tensor(self,T):

        # Extend dimensions of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Compute F according to the given formula
        F = T_ext1 - T_ext2

        # Masking conditions
        mask_invalid = (T_ext1 == self.padding) | (T_ext2 == self.padding) | (torch.triu(torch.ones_like(F), diagonal=1)==0)
        F[mask_invalid] = self.ignore_idx
        return F
    def __call__(self,pred,truth):
        pred = compute_sine_matrix(pred)
        truth = (truth-truth.min(dim=1,keepdim=True).values)/truth.max(dim=1,keepdim=True).values
        truth = self.calculate_pair_gt_tensor(truth)
        mask = truth != self.ignore_idx
        return self.lossf(truth[mask],pred[mask])

class PairwiseLoss:
    def __init__(self,padding=-1):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
        self.ignore_idx = -100
    def calculate_pair_gt_tensor(self,T, is_truth=True):

        # Extend dimensions of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Compute F according to the given formula
        F = T_ext1 - T_ext2 

        # Masking conditions

        if is_truth:
            mask_invalid = (T_ext1 == self.padding) | (T_ext2 == self.padding) | (torch.triu(torch.ones_like(F), diagonal=1)==0)
            F[mask_invalid] = self.ignore_idx
        return F

    def __call__(self,pred,truth):
        truth = (truth-truth.min(dim=1,keepdim=True).values)/truth.max(dim=1,keepdim=True).values
        pred = (pred-pred.min(dim=1,keepdim=True).values)/pred.max(dim=1,keepdim=True).values

        pred = self.calculate_pair_gt_tensor(pred,is_truth=False)
        truth = self.calculate_pair_gt_tensor(truth)
        mask = truth != self.ignore_idx
        return self.lossf(truth[mask],pred[mask])

class TanhPairwiseLoss:
    def __init__(self,padding=-1,epsilon=1e-5):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
        self.ignore_idx = -100
        self.epsilon = epsilon
    def calculate_pair_gt_tensor(self,T, is_truth=True):

        # Extend dimensions of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Compute F according to the given formula
        F = torch.tanh((T_ext1 - T_ext2)/self.epsilon) 

        # Masking conditions

        if is_truth:
            mask_invalid = (T_ext1 == self.padding) | (T_ext2 == self.padding) | (torch.triu(torch.ones_like(F), diagonal=1)==0)
            F[mask_invalid] = self.ignore_idx
        return F

    def __call__(self,pred,truth):
        pred = self.calculate_pair_gt_tensor(pred,is_truth=False)
        truth = self.calculate_pair_gt_tensor(truth)
        mask = truth != self.ignore_idx
        return self.lossf(truth[mask],pred[mask])
    
class ListMLEwMSELoss:
    def __init__(self,padding=-1,mse_weight=1):
        self.padding = padding
        self.lossf = torch.nn.MSELoss(reduction="mean")
        self.mse_weight = mse_weight
    def __call__(self,truth,pred):
        mask = truth !=self.padding
        return torch.mean(torch.abs(truth[mask]-pred[mask])/(truth[mask]))*self.mse_weight + listMLE(pred,truth)