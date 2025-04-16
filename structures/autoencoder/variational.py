import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .ae_base import Autoencoder

class VariationalAutoencoder(Autoencoder):
    def __init__(self, **params):
        super(VariationalAutoencoder, self).__init__(**params)
        self.params = params
    
    def encode(self, x: Tensor)->tuple[Tensor, Tensor]:
        for layer in self.encoder_layers:
            x = layer(x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var
    
    def decode(self, mu: Tensor, log_var: Tensor)->Tensor:
        z = mu + torch.randn_like(log_var) * torch.exp(log_var / 2)
        for layer in self.decoder_layers:
            z = layer(z)
        return z
    
    def update_params(self, x: Tensor)->Tensor:
        pass

