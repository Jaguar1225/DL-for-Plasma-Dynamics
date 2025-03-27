import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizationLoss(nn.Module):
    def __init__(self, **params):
        super(RegularizationLoss, self).__init__()  
        self.params = params
        self.params.setdefault('p', 1)
        
    def forward(self, layers: nn.ModuleList):
        loss = 0
        total_params = 0
        for layer in layers:
            loss += layer.weight.norm(p=self.params.get('p', 1))
            loss += layer.bias.norm(p=self.params.get('p', 1))
            total_params += layer.weight.numel() + layer.bias.numel()
        return loss/total_params**(1/self.params.get('p', 1))