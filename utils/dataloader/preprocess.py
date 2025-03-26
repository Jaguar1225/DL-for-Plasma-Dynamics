import numpy as np

import torch

class Normalize:
    def __init__(self, **params):
        self.params = params

    def standardize(self, data):
        return (data - data.mean(axis=-1,keepdims=True))/(data.std(axis=-1,keepdims=True))
    
    def minmax(self, data):
        return (data - data.min(axis=-1,keepdims=True))/(data.max(axis=-1,keepdims=True)-data.min(axis=-1,keepdims=True))
    
    def maxabs(self, data):
        return data/data.max(axis=-1,keepdims=True)
    
    def partial(self, data):
        try:
            return data/np.sum(data,axis=-1,keepdims=True)
        except:
            return data/torch.sum(data,dim=-1,keepdims=True)
        
    def partial_normalize(self, data, window_size=10):
        return data/np.sum(data,axis=-1,keepdims=True)


    
    
