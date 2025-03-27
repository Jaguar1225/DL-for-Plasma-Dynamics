import numpy as np

import torch

class Normalize:
    def __init__(self, **params):
        self.params = params

    def standardize(self, data):
        if data is torch.Tensor:
            return (data - data.mean(dim=-1,keepdim=True))/(data.std(dim=-1,keepdim=True))
        else:
            return (data - data.mean(axis=-1,keepdims=True))/(data.std(axis=-1,keepdims=True))
    
    def minmax(self, data):
        if data is torch.Tensor:
            return (data - data.min(dim=-1,keepdim=True))/(data.max(dim=-1,keepdim=True)-data.min(dim=-1,keepdim=True))
        else:
            return (data - data.min(axis=-1,keepdims=True))/(data.max(axis=-1,keepdims=True)-data.min(axis=-1,keepdims=True))
    
    def maxabs(self, data):
        if data is torch.Tensor:
            return data/data.max(dim=-1,keepdim=True)
        else:
            return data/data.max(axis=-1,keepdims=True)
    
    def partial(self, data):
        if data is torch.Tensor:
            return data/torch.sum(data,dim=-1,keepdims=True)
        else:
            return data/np.sum(data,axis=-1,keepdims=True)
        
    def partial_normalize(self, data, window_size=10):
        if data is torch.Tensor:
            return data/torch.sum(data,dim=-1,keepdims=True)
        else:
            return data/np.sum(data,axis=-1,keepdims=True)


    
    
