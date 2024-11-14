import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class MSELoss(torch.nn.Module):
    """MSE loss."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

def check_type(x):
    return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

def acc(pred, targets):
    """Returns accuracy given batch of categorical predictions and targets."""
    pred = check_type(pred)
    targets = check_type(targets)
    return accuracy_score(targets, pred)