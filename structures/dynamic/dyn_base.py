import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

from ..optimizer import Opt
from ..loss_func import RegularizationLoss, ReconstructionLoss
from typing import Union

class DynBase(Opt):
    def __init__(self):
        if torch.cuda.is_available():
            if self.params['device'] == 'cuda':
                self.params['device'] = torch.device('cuda')
        else:
            self.params['device'] = torch.device('cpu')
        self.params.setdefault('dtype', torch.float32)

        super(DynBase, self).__init__()                                   
        self.rnn_layers = nn.ModuleList()
        self.to(self.params['device'])

    def add_rnn_layer(self, layer : nn.Module)->None:
        layer.to(self.params['device'])
        self.rnn_layers.append(layer)

    def delete_rnn_layer(self)->nn.Module:
        if len(self.rnn_layers) > 0:
            removed_layer = self.rnn_layers.pop(-1)
            return removed_layer
        return None
    
    def update_rnn_layer(self)->None:
        self.clear_training_layer()
        for layer in self.rnn_layers:
            self.add_training_layer(layer)
        self.update_optimizer()


class RNN(DynBase):
    def __init__(self, **params):
        self.params = params
        self.params['model'] = self.__class__.__name__

        super(RNN, self).__init__()
        self.reconstruction_loss = ReconstructionLoss(**self.params.get('reconstruction_loss', {}))
        
        try:
            if self.params['rnn_layers'] != None:
                for layer in self.params['rnn_layers']:
                    self.add_rnn_layer(layer)

        except KeyError as key_error:
            pass

        except Exception as e:
            print(f"Error: {e}")

    def forward(self, *inputs : Union[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z, v = inputs

        for layer in self.rnn_layers:
            z = layer(z, v)
        return z
    
    def loss_fn(self, *inputs : Union[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z_pred, z_true = inputs
        return self.reconstruction_loss(z_pred, z_true)
