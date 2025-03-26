import torch.nn as nn
import torch as Tensor

from ..act_func import ActivationFunction as act

class UnitCoder(nn.Module):
    def __init__(self, **params):
        super(UnitCoder, self).__init__()
        self.params = params
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.activation_function = params['activation_function']
        self.Layer = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            act(activation_function=self.activation_function),
        )
        self.weight = self.Layer[0].weight
        self.bias = self.Layer[0].bias

    def forward(self, x: Tensor):
        return self.Layer(x)