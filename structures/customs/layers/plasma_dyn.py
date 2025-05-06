import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

from typing import Union, Tuple

from ..act_func import ActivationFunction as act


class PlasmaDynamics(nn.Module):
    def __init__(self, **params):
        super(PlasmaDynamics, self).__init__()
        self.params = params
        
        # 기본 파라미터 설정
        self.input_dim = params['input_dim']  # z_t의 차원
        self.condition_dim = params['condition_dim']  # v_{t+1}의 차원
        self.output_dim = params['output_dim']  # z_{t+1}의 차원
        
        # 동적 파라미터 μ와 α를 학습 가능한 파라미터로 정의
        self.mu = nn.Parameter(torch.ones(self.input_dim, self.input_dim))
        self.alpha = nn.Parameter(torch.ones(self.input_dim, self.condition_dim))
        
        # 노이즈 모델링을 위한 파라미터
        self.noise_scale = nn.Parameter(torch.ones(self.output_dim))
        
    def forward(self, z_t: torch.Tensor, v_t1: torch.Tensor) -> torch.Tensor:
        """
        z_t: 현재 상태 (batch_size, input_dim)
        v_t1: 다음 시간의 프로세스 조건 (batch_size, condition_dim)
        """
        # 동적 모델 구현
        z_pred = torch.matmul(z_t, self.mu) + torch.matmul(v_t1, self.alpha.T)
        
        # 노이즈 추가 (학습 중에는 비활성화)
        if self.training:
            noise = torch.randn_like(z_pred) * self.noise_scale
            z_pred = z_pred + noise
            
        return z_pred
    
    def get_dynamic_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """현재 학습된 동적 파라미터 μ와 α를 반환"""
        return self.mu.detach(), self.alpha.detach()
    
    def clone(self) -> nn.Module:
        """모듈의 깊은 복사본을 생성"""
        new_model = PlasmaDynamics(**self.params)
        with torch.no_grad():
            new_model.mu.copy_(self.mu)
            new_model.alpha.copy_(self.alpha)
            new_model.noise_scale.copy_(self.noise_scale)
        return new_model
