import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_base import Autoencoder

class VariationalAutoencoder(Autoencoder):
    def __init__(self, **params):
        super(VariationalAutoencoder, self).__init__(**params)
        self.params = params
    
    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var
    
    def decode(self, mu, log_var):
        z = mu + torch.randn_like(log_var) * torch.exp(log_var / 2)
        for layer in self.decoder_layers:
            z = layer(z)
        return z
    
    def update_params(self, x):
        pass

