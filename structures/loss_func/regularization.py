import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizationLoss(nn.Module):
    def __init__(self, **params):
        super(RegularizationLoss, self).__init__()  
        self.params = params
        self.params.setdefault('p', 1)
        
    def forward(self, layers: nn.ModuleList):
        return sum([layer.weight.norm(p=self.params.get('p', 1)) for layer in layers])