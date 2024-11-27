# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 2024

@author: Jaguar

version log;

    1. Beta funcion was fixed in BandKernel, by adding .exp()
    2. Each kernel visualizing was updated
"""
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import matplotlib.pyplot as plt
class BroadeningKernelPattern(nn.Module):
    def __init__(self, wavelengths):
        torch.autograd.set_detect_anomaly(True)
        
        nn.Module.__init__(self)

        self.sigma_g = nn.Parameter(torch.rand(1)*self.kernel_size/30)
        self.sigma_l = nn.Parameter(torch.rand(1)*self.kernel_size/60)
        
        self.params.setdefault('slit_width',
                               nn.Parameter(
                                   torch.tensor(5000),
                                   requires_grad=False
                                   )
                               )#nm
        self.params.setdefault('slit_spacing',
                               nn.Parameter(
                                   torch.tensor(1667),
                                   requires_grad=False)
                               ) #nm
        self.params.setdefault('blaze_angle',
                               nn.Parameter(
                                   torch.tensor(0.090),
                                   requires_grad=False
                                   )
                               ) #nm
        self.params.setdefault('resolution',
                               nn.Parameter(
                                   torch.tensor(0.36),
                                   requires_grad=False
                                   )
                               ) #nm
        
        self.diffraction_pattern = nn.Parameter(
            self.calculate_diffraction_pattern(wavelengths),
            requires_grad=False
            )

        # 고정된 범위의 x 값을 생성합니다.
        self.base = nn.Parameter(
            torch.linspace(
                -self.kernel_size//2, self.kernel_size//2, self.kernel_size
                )*self.params['resolution'],
            requires_grad=False
            )

    def _kernel(self):
        if self.mode == 'gaussian':
            kernel = self.create_gaussian_kernel()
        elif self.mode == 'lorentzian':
            kernel = self.create_lorentzian_kernel()
        elif self.mode == 'voigt':
            kernel = self.create_voigt_kernel()
        else:
            raise ValueError("Invalid mode selected. Choose 'gaussian', 'lorentzian', or 'voigt'.")

        weighted_kernel = kernel * self.diffraction_pattern
        weighted_kernel = weighted_kernel/weighted_kernel.sum()

        return weighted_kernel
    
    def calculate_diffraction_pattern(self,wavelengths):
        beta = torch.pi * self.params['slit_width'] * torch.sin(self.params['blaze_angle']) / wavelengths[:self.kernel_size]
        gamma = torch.pi * self.params['slit_spacing'] * torch.sin(self.params['blaze_angle']) / wavelengths[:self.kernel_size]
        diffraction_pattern = (torch.sin(beta) / beta)**2 * (torch.sin(gamma) / gamma)**2
        return diffraction_pattern
    
    def create_gaussian_kernel(self):
        exponent = -0.5 * ((self.base)/ self.sigma_g) ** 2
        kernel1d = exponent.exp()
        kernel1d = kernel1d/kernel1d.sum()
        return kernel1d.view(1, 1, self.kernel_size)

    def create_lorentzian_kernel(self):
        kernel1d = 1 / (1 + (self.base/ self.sigma_l) ** 2)
        kernel1d = kernel1d/kernel1d.sum()
        return kernel1d.view(1, 1, self.kernel_size)

    def create_voigt_kernel(self):
        # Voigt 커널은 가우시안과 로렌츠 커널의 결합이므로, 각 커널을 생성하고 결합합니다.
        gaussian = self.create_gaussian_kernel()
        lorentzian = self.create_lorentzian_kernel()
        kernel1d = gaussian * lorentzian  # 여기서는 단순한 요소별 곱셈으로 근사
        kernel1d = kernel1d/kernel1d.sum()
        return kernel1d
            
    def visualize_kernels(self,Save=False):
        # 각 커널을 생성합니다.
        gaussian_kernel = self.create_gaussian_kernel()
        lorentzian_kernel = self.create_lorentzian_kernel()
        voigt_kernel = self.create_voigt_kernel()
        weighted_kernel = self._kernel()

        # 시각화를 위한 subplot 설정
        fig, axs = plt.subplots(1, 4, gridspec_kw={'wspace': 1})
        
        axs[0].imshow(gaussian_kernel.squeeze(1).detach().cpu().T,cmap='gray',aspect='auto')
        axs[0].set_title('Gaussian Kernel')
        
        axs[1].imshow(lorentzian_kernel.squeeze(1).detach().cpu().T,cmap='gray',aspect='auto')
        axs[1].set_title('Lorentzian Kernel')
        
        axs[2].imshow(voigt_kernel.squeeze(1).detach().cpu().T,cmap='gray',aspect='auto')
        axs[2].set_title('Voigt Kernel')
        
        axs[3].imshow(weighted_kernel.squeeze(1).detach().cpu().T,cmap='gray',aspect='auto')
        axs[3].set_title('Sin² Kernel')

        for ax in axs:
            ax.set_xticks([])
            plt.colorbar(ax.get_images()[0],ax=ax)
        if Save:
            plt.savefig(f"Fig/Kernels/{time.strftime('%Y-%m-%d', time.localtime())} kernel_size {self.kernel_size}.png")
            
        plt.show()
        plt.close()
    
class BroadeningConvLayer(BroadeningKernelPattern):
    def __init__(self, wavelengths, kernel_size, 
                 stride = 1, mode='voigt', bias = True,
                 **params):
        
        self.params = params
        self.kernel_size = kernel_size
        
        BroadeningKernelPattern.__init__(self,wavelengths)

        self.stride = stride
        self.mode = mode
        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 결합된 커널을 사용하여 컨볼루션 수행
        kernel = self._kernel()
        output = F.conv1d(x, kernel, stride=self.stride, padding=self.kernel_size // 2)
        if self.use_bias:
            output += self.bias.view(1, -1, 1)
        return output

class BroadeningTransConvLayer(BroadeningKernelPattern):
    def __init__(self, wavelengths, kernel_size, stride = 1, 
                 mode='voigt', bias = True,
                 **params):
        
        self.params = params
        self.kernel_size = kernel_size
        
        BroadeningKernelPattern.__init__(self,wavelengths)
        
        torch.autograd.set_detect_anomaly(True)
        

        self.stride = stride
        self.mode = mode
        
        # Transposed Convolution 레이어를 초기화하지 않습니다. 대신, forward에서 직접 커널을 사용합니다.
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # 결합된 커널을 사용하여 Transposed Convolution 수행
        kernel = self._kernel()
        output = F.conv_transpose1d(x, kernel, stride=self.stride)
        if self.use_bias:
            output += self.bias.view(1, -1, 1)
        return output
    
class DiracDeltaKernel(nn.Module):
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
        nn.Module.__init__(self)
        kernel1d = torch.zeros((self.num_filter,1,self.kernel_size))
        kernel1d[:,0,self.kernel_size//2] = 1
        self.kernel1d = nn.Parameter(kernel1d,requires_grad=False)
    def _kernel(self):
        return self.kernel1d
    
class DiracDeltaTransConvLayer(DiracDeltaKernel):
    def __init__(self,kernel_size, num_filter, stride = 1,bias = True):
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        DiracDeltaKernel.__init__(self)

        self.stride = stride
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 결합된 커널을 사용하여 Transposed Convolution 수행
        kernel = self._kernel()
        output = F.conv_transpose1d(x, kernel, stride=self.stride, padding=self.kernel_size // 2, output_padding=4)
        if self.use_bias:
            output += self.bias.view(1, -1, 1)
        return output


class BandKernel(nn.Module):
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
        nn.Module.__init__(self)
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        self.base = nn.Parameter(torch.linspace(0 + 1e-5,1 - 1e-5,self.kernel_size),requires_grad=False)
    def _kernel(self):
        beta_dist = Beta(F.softplus(self.a)+1,F.softplus(self.b)+1)
        kernel1d = beta_dist.log_prob(self.base).exp()
        kernel1d = kernel1d/kernel1d.sum()
        return kernel1d.view(1,1,self.kernel_size).repeat(self.num_filter,1,1)
    def visualize_kernels(self,Save=None):
        fig, axs = plt.subplots(2,1,gridspec_kw={'height_ratios': [4.5, 1]}, figsize=(10,4),sharex=True)
        filter_xtick_indices = np.linspace(1, self.kernel_size,10, dtype=int)
        
        plt.title(f'Band_Kernel {self.kernel_size}')
        
        axs[0].plot(self._kernel().squeeze(0,1).detach().cpu(),color='red')
        axs[0].grid()
        axs[0].set_xticks(filter_xtick_indices)
        axs[0].set_xticklabels([])
        
        axs[1].imshow(self._kernel().squeeze(1).detach().cpu(),cmap='gray',aspect='auto')
        axs[1].set_xticks(filter_xtick_indices)
        axs[1].set_xticklabels(filter_xtick_indices)
        
        plt.tight_layout()
        
        if Save:
            plt.savefig(f"Fig/Kernels/{time.strftime('%Y-%m-%d', time.localtime())} kernel_size {self.kernel_size}.png")
        plt.show()
        plt.close()

class BandConvLayer(BandKernel):
    def __init__(self,kernel_size, num_filter, stride = 1,bias = True):
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        BandKernel.__init__(self)
        self.stride = stride
        self.use_bias = bias
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        kernel = self._kernel()
        output = F.conv1d(x, kernel, stride=self.stride, padding=self.kernel_size // 2 )
        if self.use_bias:
            output += self.bias.view(1, -1, 1)
        return output
    
class BandTransConvLayer(BandKernel):
    def __init__(self,kernel_size, num_filter, stride = 1,bias = True):
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        BandKernel.__init__(self)

        self.stride = stride
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 결합된 커널을 사용하여 Transposed Convolution 수행
        kernel = self._kernel()
        output = F.conv_transpose1d(x, kernel, stride=self.stride, padding=self.kernel_size // 2,output_padding=4)
        if self.use_bias:
            output += self.bias.view(1, -1, 1)
        return output


