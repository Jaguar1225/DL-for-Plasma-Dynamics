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
        with torch.no_grad():
            new_transformer.Q.weight.copy_(self.Q.weight)
            new_transformer.Q.bias.copy_(self.Q.bias)

            new_transformer.K.weight.copy_(self.K.weight)
            new_transformer.K.bias.copy_(self.K.bias)

            new_transformer.V.weight.copy_(self.V.weight)
            new_transformer.V.bias.copy_(self.V.bias)
        return new_transformer