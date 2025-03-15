import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Autoencoder

class VariationalAutoencoder(Autoencoder):
    def __init__(self, **params):
        Autoencoder.__init__(self, **params)


