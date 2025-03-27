import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

from ..act_func import ActivationFunction as act

class UnitTransformer(nn.Module):
    def __init__(self, **params):
        super(UnitTransformer, self).__init__()

        self.params = params

        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']

        self.Q = nn.Linear(self.input_dim, self.output_dim)
        self.K = nn.Linear(self.input_dim, self.output_dim)
        self.V = nn.Linear(self.input_dim, self.output_dim)

        self.activation_function = params['activation_function']

        self.act = act(self.activation_function)

    def forward(self, x: Tensor):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        attn = q@(k.transpose(-2, -1))
        if hasattr(self, 'activation_function'):
            attn = self.act(attn/torch.sqrt(torch.tensor(self.output_dim)))

        return attn@v
    
    def clone(self):
        new_transformer = UnitTransformer(**self.params)
        new_transformer.Q.weight.data.copy_(self.Q.weight.data)
        new_transformer.Q.bias.data.copy_(self.Q.bias.data)

        new_transformer.K.weight.data.copy_(self.K.weight.data)
        new_transformer.K.bias.data.copy_(self.K.bias.data)
        
        new_transformer.V.weight.data.copy_(self.V.weight.data)
        new_transformer.V.bias.data.copy_(self.V.bias.data)
        
        return new_transformer