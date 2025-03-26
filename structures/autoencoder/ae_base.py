import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor
from ..optimizer import Opt
from ..loss_func import RegularizationLoss, ReconstructionLoss

class StackingAutoencoderBase(Opt):
    def __init__(self):
        #torch default
        if torch.cuda.is_available():
            if self.params['device'] == 'cuda':
                self.params['device'] = torch.device('cuda')
        else:
            self.params['device'] = torch.device('cpu')
        self.params.setdefault('dtype', torch.float32)
        
        #layer init
        super(StackingAutoencoderBase, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.to(self.params['device'])

    def add_encoder_layer(self, layer : nn.Module):
        layer.to(self.params['device'])
        self.encoder_layers.append(layer)

    def add_decoder_layer(self, layer : nn.Module):
        layer.to(self.params['device'])
        self.decoder_layers.append(layer)

    def delete_encoder_layer(self):
        self.encoder_layers.pop()

    def delete_decoder_layer(self):
        self.decoder_layers.pop()

    def update_layer(self):
        self.clear_training_layer()
        self.add_training_layer(self.encoder_layers[-1])
        self.add_training_layer(self.decoder_layers[-1])
        self.update_optimizer()

class Autoencoder(
    StackingAutoencoderBase,
    RegularizationLoss,
    ReconstructionLoss
    ):

    def __init__(self, **params):
        self.params = params
        self.params['model'] = self.__class__.__name__
        self.reconstruction_loss = ReconstructionLoss(**self.params.get('reconstruction_loss', {}))
        self.regularization_loss = RegularizationLoss(**self.params.get('regularization_loss', {}))

        super(Autoencoder, self).__init__()

        try:
            if self.params['encoder_layers'] != None:
                for layer in self.params['encoder_layers']:
                    self.add_encoder_layer(layer)

            if self.params['decoder_layers'] != None:
                for layer in self.params['decoder_layers']:
                    self.add_decoder_layer(layer)

        except KeyError as key_error:
            pass

        except Exception as e:
            print(f"Error: {e}")

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
    
    def loss_fn(self, *inputs):
        intensity, condition = inputs
        loss = self.reconstruction_loss(self.forward(intensity), intensity)
        loss += self.params.get('lambda_reg', 0) * self.regularization_loss(self.training_layers)
        return loss

