import numpy as np
from typing import Union
import torch

class Normalize:
    def __init__(self, **params):
        self.params = params
        self.map_preprocess = {
            'standardize': self.standardize,
            'minmax': self.minmax,
            'maxabs': self.maxabs,
            'partial': self.partial,
            'partial_normalize': self.partial_normalize
        }
        self.params['preprocess'] = self.map_preprocess[self.params['preprocess'].lower()]

    def __call__(self, data: Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
        return self.params['preprocess'](data)
    
    def standardize(self, data: Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return (data - data.mean(dim=0,keepdim=True))/(data.std(dim=0,keepdim=True))
        else:
            return (data - data.mean(axis=0,keepdims=True))/(data.std(axis=0,keepdims=True))
    
    def minmax(self, data: Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return (data - data.min(dim=0,keepdim=True))/(data.max(dim=0,keepdim=True)-data.min(dim=0,keepdim=True))
        else:
            return (data - data.min(axis=0,keepdims=True))/(data.max(axis=0,keepdims=True)-data.min(axis=0,keepdims=True))
    
    def maxabs(self, data: Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return data/data.max(dim=0,keepdim=True)
        else:
            return data/data.max(axis=0,keepdims=True)
    
    def partial(self, data: Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return data/torch.sum(data,dim=0,keepdim=True)
        else:
            return data/np.sum(data,axis=0,keepdims=True)
        
    def partial_normalize(self, data: Union[torch.Tensor, np.ndarray], window_size: int = 10)->Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return data/torch.sum(data,dim=0,keepdim=True)
        else:
            return data/np.sum(data,axis=0,keepdims=True)

    
    
