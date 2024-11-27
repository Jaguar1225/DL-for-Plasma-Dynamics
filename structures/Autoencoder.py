from utils.trains import AutoencoderTrains

import time
import pickle

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.CustomModules import *

class Autoencoder(nn.Module, AutoencoderTrains):
    def __init__(self, **params):
        super(Autoencoder, self).__init__()
        self.params = params
        
        # 1. date 설정
        self.date = params.get('date', time.strftime('%Y-%m-%d', time.localtime()))
        
        # 2. 기본 파라미터 설정
        self._set_default_params()
        
        # 3. 저장 경로 설정
        self._setup_save_path()
        
        # 4. 나머지 초기화
        self.init_params()

    def _setup_save_path(self):
        """저장 경로 설정"""
        save_path = self.date
        base_path = f'model/{self.__class__.__name__}/{save_path}'
        
        if os.path.exists(f"{base_path}/{self.params['keyword']}"):
            idx = 1
            while os.path.exists(f"{base_path}/{self.params['keyword']}_{idx:03d}"):
                idx += 1
            save_path = f"{self.date}_{idx:03d}"
        
        # save_path를 params에 추가
        self.params['save_path'] = save_path
        
        # 저장 디렉토리 생성
        os.makedirs(f'model/{self.__class__.__name__}/{save_path}', exist_ok=True)

    def _set_default_params(self):
        """기본 파라미터 설정"""
        defaults = {
            'base_dimension': 3648,
            'hidden_dimension': 16,
            'layer_dimension': [3648,3648,16],
            'process_variables': 16,
            'batch_size': 2**14,
            'optimizer_sae': None,
            'dropout_rate': 0,
            'l1_weight': 1e-3,
            'cos_weight': 1,
            'dtype': torch.float32,
            'device': device,
            'random_seed': 0,
            'keyword': "",
            'Writer_dir': f'logs/{self.date}',
            'trial': 0,
            'hidden_dim': 16,
            'layer_depth': 2
        }
        
        # 기본값으로 params 업데이트
        for key, value in defaults.items():
            self.params.setdefault(key, value)

    def forward(self,X):
        Z = self.encode(X)
        X_hat = self.decode(Z)
        return X_hat
    
    def encode(self,X):
        Transpose = False
        if len(X.size()) == 1:
            X = X.unsqueeze(0)
        elif len(X.size()) == 2:
            pass
        else:
            N,T,D = X.shape
            X = X.reshape(-1,self.params['base_dimension'])
            Transpose = True
        
        for n in range(len(self.params['layer_dimension'])-1):
            X = self.EncoderDict[f'E{n}'](X)
        
        Z = self.EncoderDict[f"E{len(self.params['layer_dimension'])-1}"](X)

        if Transpose:
            Z = Z.view(N, T,self.params['hidden_dimension'])
        return Z    
    
    def decode(self,Z):
        Transpose = False
        if len(Z.size()) == 1:
            Z = Z.unsqueeze(0)
        
        elif len(Z.size()) == 2:
            pass

        else:
            N,T,D = Z.shape
            Z = Z.view(N*T,D)
            Tranpose = True
        
        for n in range(len(self.params['layer_dimension'])-1):
            Z = self.EncoderDict[f'D{n}'](Z)

        X_hat = self.EncoderDict[f"D{len(self.params['layer_dimension'])-1}"](Z)
        X_hat = F.relu(X_hat)
        X_hat = X_hat/torch.norm(X_hat,dim=1,keepdim=True)

        if Transpose:
            X_hat = X_hat.view(N, T, self.params['base_dimension'])

        return X_hat
    
    def init_params(self):
        self.init_encoder()
        self.to(self.params['device'])

    def init_encoder(self):
        self.EncoderDict = nn.ModuleDict()
        in_dimension = self.params['base_dimension']
        
        # use_log와 use_residual 파라미터 가져오기
        use_log = self.params.get('use_log', False)
        use_residual = self.params.get('use_residual', False)

        for n, d in enumerate(self.params['layer_dimension']):
            if n == len(self.params['layer_dimension'])-1:
                self.EncoderDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    normalization='batchnorm', 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=use_log,  # use_log 전달
                    use_residual=use_residual  # use_residual 전달
                )
            else:
                self.EncoderDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    normalization='batchnorm', 
                    activation='tanh', 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=use_log,  # use_log 전달
                    use_residual=use_residual  # use_residual 전달
                )
            in_dimension = d

        out_dimension = self.params['base_dimension']
        for n,d in enumerate(self.params['layer_dimension']):
            if n == 0:
                self.EncoderDict[f"D{len(self.params['layer_dimension'])-n-1}"] = UnitCoder(
                    d, out_dimension,
                    activation='relu',
                    dropout_rate=self.params['dropout_rate'],
                    use_log=use_log,  # use_log 전달
                    use_residual=use_residual  # use_residual 전달
                )
            else:
                self.EncoderDict[f"D{len(self.params['layer_dimension'])-n-1}"] = UnitCoder(
                    d, out_dimension, 
                    normalization='batchnorm', 
                    activation='tanh', 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=use_log,  # use_log 전달
                    use_residual=use_residual  # use_residual 전달
                )
            out_dimension = d

    def loss(self, x_hat, x, reduction = 'mean'):
        recon_loss = self.mse_loss(x_hat, x, reduction=reduction) 
        if self.params['cos_weight'] > 0:
            cos_loss = self.params['cos_weight']*self.cos_loss(x_hat,x,reduction=reduction)
        else:
            cos_loss = 0
        if self.params['l1_weight'] > 0:
            l1_loss = self.params['l1_weight']*self.l1_loss()
        else:
            l1_loss = 0
        loss = recon_loss + cos_loss + l1_loss
        return loss, recon_loss, cos_loss, l1_loss
    
    def mse_loss(self, x_hat, x, reduction = 'mean'):
        criterion = nn.MSELoss(reduction=reduction)
        if reduction == 'none':
            return criterion(x_hat,x).mean(dim=1)
        return criterion(x_hat,x)
    
    def cos_loss(self, x_hat, x, reduction = 'mean'):
        target = torch.ones(x.size(0)).to(x.device)
        criterion = nn.CosineEmbeddingLoss(reduction=reduction)
        return criterion(x_hat,x,target)
    
    def l1_loss(self):
        total_params = sum(p.numel() for p in self.parameters())
        l1_sum = sum(p.abs().sum() for p in self.parameters())
        avg_l1 = l1_sum / total_params
        num_layers = len(self.params['layer_dimension'])
        adjusted_l1 = avg_l1 / num_layers
        return torch.log1p(adjusted_l1)
    
class OrthogonalAutoencoder(Autoencoder):
    def __init__(self,**params):
        params['use_orthogonal'] = True
        params['orthogonal_weight'] = 1e-3
        super(OrthogonalAutoencoder, self).__init__(**params)
        
    def orthogonal_loss(self,Z):
        Z = Z.view(Z.size(0),-1)
        Z_T = Z.T
        Z_T_Z = Z_T@Z
        I = torch.eye(Z_T_Z.size(0))
        return torch.norm(Z_T_Z-I,dim=(-2,-1))
    def loss(self,x_hat,x,reduction='mean'):
        Z = self.encode(x)
        loss, recon_loss, cos_loss, l1_loss = super(OrthogonalAutoencoder, self).loss(x_hat,x,reduction)
        return loss + self.params['orthogonal_weight']*self.orthogonal_loss(Z), recon_loss, cos_loss, l1_loss


class ResAutoencoder(Autoencoder):
    def __init__(self,**params):
        params['use_residual'] = True
        super(ResAutoencoder, self).__init__(**params)

class LogAutoencoder(Autoencoder):
    def __init__(self,**params):
        params['use_log'] = True
        super(LogAutoencoder, self).__init__(**params)

class LogResAutoencoder(Autoencoder):
    def __init__(self,**params):
        params['use_log'] = True
        params['use_residual'] = True
        super(LogResAutoencoder, self).__init__(**params)

class VariationalAutoencoder(Autoencoder):
    def __init__(self,**params):
        super(VariationalAutoencoder, self).__init__(**params)
        self.init_params()
        # 인코더의 마지막 레이어에서 평균과 로그 분산을 출력하도록 설정
    def init_params(self):
        super(VariationalAutoencoder, self).init_params()
        self.fc_mu = nn.Linear(self.params['hidden_dimension'], self.params['hidden_dimension'])
        self.fc_logvar = nn.Linear(self.params['hidden_dimension'], self.params['hidden_dimension'])

    def encode(self, X):
        Z = super(VariationalAutoencoder, self).encode(X)
        mu = self.fc_mu(Z)  # Z를 통해 평균 계산
        logvar = self.fc_logvar(Z)  # Z를 통해 로그 분산 계산
        return Z, mu, logvar

    def decode(self, Z):
        return super(VariationalAutoencoder, self).decode(Z)

    def forward(self, X):
        Z, _, _ = self.encode(X)
        X_hat = self.decode(Z)
        return X_hat

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x_hat, x, reduction = 'mean'):
        _, mu, logvar = self.encode(x)
        loss, recon_loss, cos_loss, l1_loss = super(VariationalAutoencoder, self).loss(x_hat, x, reduction)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss + KLD, recon_loss, cos_loss, l1_loss

class VariationalResAutoencoder(VariationalAutoencoder):
    def __init__(self,**params):
        params['use_residual'] = True
        super(VariationalResAutoencoder, self).__init__(**params)

class VariationalLogAutoencoder(VariationalAutoencoder):
    def __init__(self,**params):
        params['use_log'] = True
        super(VariationalLogAutoencoder, self).__init__(**params)

class VariationalLogResAutoencoder(VariationalAutoencoder):
    def __init__(self,**params):
        params['use_log'] = True
        params['use_residual'] = True
        super(VariationalLogResAutoencoder, self).__init__(**params)

    