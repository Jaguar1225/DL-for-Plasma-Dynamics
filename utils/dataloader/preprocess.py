import numpy as np

import torch

class Normalize:
    def __init__(self, **params):
        self.params = params

    def standardize(self, data):
        return (data - data.mean(axis=-1))/(data.std(axis=-1))
    
    def minmax(self, data):
        return (data - data.min(axis=-1))/(data.max(axis=-1)-data.min(axis=-1))
    
    def maxabs(self, data):
        return data/data.max(axis=-1)
    
    def partial(self, data):
        try:
            return data/np.sum(data,axis=-1)
        except:
            return data/torch.sum(data,dim=-1)
        
    def partial_normalize(self, data, window_size=10):
        return data/np.sum(data,axis=-1)


    
    
