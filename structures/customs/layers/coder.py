import torch.nn as nn
import torch as Tensor
import torch
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

    def clone(self):
        """모듈의 깊은 복사본을 생성합니다."""
        new_layer = UnitCoder(**self.params)
        # 가중치와 편향 복사
        new_layer.Layer[0].weight.data.copy_(self.Layer[0].weight.data)
        new_layer.Layer[0].bias.data.copy_(self.Layer[0].bias.data)
        # weight와 bias 참조 업데이트
        new_layer.weight = new_layer.Layer[0].weight
        new_layer.bias = new_layer.Layer[0].bias
        return new_layer