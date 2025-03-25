import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor
from optimizer import Opt
from loss_func import RegularizationLoss, ReconstructionLoss

class StackingAutoencoderBase(Opt):
    def __init__(self):
        #tensorboard
        self.params.setdefault('log_dir', f'./logs/{self.params["model"].__name__}/hidden_dim_{
            [layer.hidden_dim for layer in self.encoder_layers].join("_")
            }')
        
        super(StackingAutoencoderBase, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

    def add_encoder_layer(self, layer : nn.Module):
        self.encoder_layers.append(layer)

    def add_decoder_layer(self, layer : nn.Module):
        self.decoder_layers.append(layer)

    def delete_encoder_layer(self):
        self.encoder_layers.pop()

    def delete_decoder_layer(self):
        self.decoder_layers.pop()

    def update_encoder_layer(self):
        self.clear_training_layer()
        self.add_training_layer(self.encoder_layers[-1])
        self.add_training_layer(self.decoder_layers[-1])

class Autoencoder(
    StackingAutoencoderBase,
    RegularizationLoss,
    ReconstructionLoss
    ):

    def __init__(self, **params):
        self.params = params
        StackingAutoencoderBase.__init__(self)
        RegularizationLoss.__init__(self)
        ReconstructionLoss.__init__(self)

        if self.params['encoder_layers'] != None:
            for layer in self.params['encoder_layers']:
                self.add_encoder_layer(layer)

        if self.params['decoder_layers'] != None:
            for layer in self.params['decoder_layers']:
                self.add_decoder_layer(layer)

    def forward(self, x : Tensor):
        z= self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def encode(self, x : Tensor):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def decode(self, z : Tensor):
        for layer in self.decoder_layers[::-1]:
            z = layer(z)
        return z
    
    def loss_fn(self, x : Tensor):
        loss = self.reconstruction_loss(self.forward(x), x)
        loss += self.params.get('lambda_reg', 0) * self.regularization_loss(self.training_layers)
        return loss

