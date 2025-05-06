import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

from typing import Union

from ..act_func import ActivationFunction as act

class UnitRNN(nn.Module):

    r'''
        Here what I want to construct is a Simple dynamic model for plasma.

        We encode the OES signals with a trained autoencoder into Z space.
        Then the encoded feature at time t is z_t, we want to predict z_{t+1} from z_t and v_{t+1}
        where v_{t+1} is the process condition at time t+1.

        Here we can assume that, the z_{t+1} is determined by z_t and v_{t+1} like below.
        z_{t+1} = f(z_t, v_{t+1}) + \epsilon
        z_{t+1} = \mu * z_t + \alpha * v_{t+1} + \epsilon 

        where \epsilon is the noise. 
        The second assumption is that, \mu and \alpha are constant.

        This is a simple dynamic model, connecting the dynamic model with the simple recurrent neural network.

        We want to evaluate the accuracy of the model by comparing the predicted z_{t+1} and the ground truth z_{t+1} (encoded, measured data).

    '''

    def __init__(self, **params):
        super(UnitRNN, self).__init__()
        self.params = params

        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.activation_function = params['activation_function']

        self.Layer = nn.RNN(
            self.input_dim, self.output_dim, 
                   nonlinearity=self.activation_function,
                   batch_first=True)
        self.weight_ih = self.Layer.weight_ih_l0
        self.bias_ih = self.Layer.bias_ih_l0
        self.weight_hh = self.Layer.weight_hh_l0
        self.bias_hh = self.Layer.bias_hh_l0

    def forward(self, *inputs : Union[torch.Tensor, torch.Tensor])->torch.Tensor:
        return self.Layer(*inputs)

    def clone(self)->nn.Module:
        """모듈의 깊은 복사본을 생성합니다."""
        new_layer = UnitRNN(**self.params)

        # 가중치와 편향 복사
        with torch.no_grad():
            new_layer.Layer.weight_ih_l0.copy_(self.Layer.weight_ih_l0)
            new_layer.Layer.bias_ih_l0.copy_(self.Layer.bias_ih_l0)
            new_layer.Layer.weight_hh_l0.copy_(self.Layer.weight_hh_l0)
            new_layer.Layer.bias_hh_l0.copy_(self.Layer.bias_hh_l0) 

        # weight와 bias 참조 업데이트
        new_layer.weight_ih = new_layer.Layer.weight_ih_l0
        new_layer.bias_ih = new_layer.Layer.bias_ih_l0
        new_layer.weight_hh = new_layer.Layer.weight_hh_l0
        new_layer.bias_hh = new_layer.Layer.bias_hh_l0
        return new_layer