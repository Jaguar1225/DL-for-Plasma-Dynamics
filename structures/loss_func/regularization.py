import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizationLoss(nn.Module):
    def __init__(self):
        super(RegularizationLoss, self).__init__()  
        self.params.setdefault('p', 1)
        
    def forward(self, layer: nn.Module):
        return layer.weight.norm(p=self.params.get('p', 1))