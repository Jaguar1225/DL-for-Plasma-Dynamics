from utils.trains import PlasDynTrains
from structures.Autoencoder import *

import time
import pickle
import glob
import re
import os

import torch
import torch.nn as nn
import torch.nn.init as init

from utils.CustomModules import *

class AE_PlasDyn(nn.Module, PlasDynTrains):
    def __init__(self, **params):
        super(AE_PlasDyn, self).__init__()
        self.params = params.copy()
        
        # 1. date 설정
        self.date = self.params.get('date', time.strftime('%Y-%m-%d', time.localtime()))
        
        # 2. 기본 파라미터 설정
        self._set_default_params()
        
        # 3. 인코더 모델 로드
        self._load_encoder_model()
        
        # 4. 저장 경로 설정
        self._setup_save_path()
        
        # 5. 파라미터 초기화
        self.init_params()

    def _set_default_params(self):
        """기본 파라미터 설정"""
        defaults = {
            'base_dimension': 3648,
            'layer_dimension': [16,16],
            'hidden_dimension': 16,
            'sequence_length': 8,
            'process_variables': 16,
            'batch_size': 2**14,
            'optimizer_plasdyn': None,
            'dropout_rate': 0,
            'gate_type': 'glu',
            'l1_weight': 0.01,
            'cos_weight': 1,
            'dtype': torch.float32,
            'device': device,
            'random_seed': 0,
            'Writer_dir': f'logs/{self.date}',
            'keyword': "",
            'encoder': 'Autoencoder',
            'encoder_model': None,
            'use_log': False,
            'use_residual': False,
            'save_path': None,
            'trial': 0,
            'hidden_dim': 16,
            'layer_depth': 2
        }
        
        for key, value in defaults.items():
            self.params.setdefault(key, value)

    def _load_encoder_model(self):
        """인코더 모델 로드"""
        if self.params['encoder_model'] is None:
            encoder_file_dir = f"model/{self.params['encoder']}/*"
            encoders = glob.glob(encoder_file_dir)
            
            if encoders:
                latest_encoder_dir = max(encoders)
                latest_encoders = glob.glob(os.path.join(latest_encoder_dir, '*'))
                
                matching_files = []
                for file in latest_encoders:
                    match = re.search(r'_(\d+)_\d+\.pth$', file)
                    if match:
                        hidden_value = int(match.group(1))
                        if hidden_value == self.params['hidden_dimension']:
                            matching_files.append(file)
                
                if matching_files:
                    encoder_model_name = matching_files[0]
                else:
                    encoder_model_name = None
                    print("Warning: No matching files found for the desired hidden value.")
            else:
                print("Warning: No encoder folders found in the specified directory.")
                
            torch.serialization.add_safe_globals([eval(self.params['encoder'])])
            self.encoder_model = torch.load(encoder_model_name, weights_only=False)
                    
            # 파라미터 고정
            for param in self.encoder_model.parameters():
                param.requires_grad = False
            self.encoder_model.eval()

    def _setup_save_path(self):
        """저장 경로 설정 (파일 I/O 작업)"""
        save_path = self.params['date']

        if os.path.exists(f'Result/{self.__class__.__name__}/{save_path}'):
            idx = 1
            while os.path.exists(f"Result/{self.__class__.__name__}/{save_path}_{idx:03d}"):
                idx += 1
                save_path = f"{self.params['date']}_{idx:03d}"
        self.params['save_path'] = save_path
        
        # 저장 디렉토리 생성
        os.makedirs(f'Result/{self.__class__.__name__}/{save_path}', exist_ok=True)
    def forward(self, X, L):
        Z = self.encode(X)
        if torch.isnan(Z).any():
            print("NaN detected after encoding")
            return None
        Z = Z.transpose(0,1)
        Z_hat = self.plasdyn(Z, L)
        if torch.isnan(Z_hat).any():
            print("NaN detected after PlasDyn")
            return None
            
        return Z_hat
    
    def encode(self,X):
        return self.encoder_model.encode(X)
    
    def plasdyn(self,Z,V):
        for n, d in enumerate(self.params['layer_dimension']):
            V = self.PlasDynDict[f'E{n}'](V)
        return self.PlasDynDict['PlasDyn'](Z.squeeze(1),V)
    
    def init_params(self):
        self.init_plasdyn()
        self.to(self.params['device'])

    def init_plasdyn(self):
        self.PlasDynDict = nn.ModuleDict()
        in_dimension = self.params['process_variables']

        for n, d in enumerate(self.params['layer_dimension']):
            if n == len(self.params['layer_dimension'])-1:
                self.PlasDynDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=False,  # use_log 전달
                    use_residual=False  # use_residual 전달
                )
            else:
                self.PlasDynDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    normalization='batchnorm', 
                    activation='tanh', 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=False,  # use_log 전달
                    use_residual=False  # use_residual 전달
                )
            in_dimension = d

        self.PlasDynDict['PlasDyn'] = PlasDyn(
            hidden_dimension=self.params['hidden_dimension'],
            gate_type=self.params['gate_type'],
            batch_first=True
        )
                
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.PlasDynDict.modules():  # .modules()를 사용하여 모든 하위 모듈에 접근
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, PlasDyn):
                for name, param in m.named_parameters():
                    if 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'weight_ih' in name:
                        init.xavier_uniform_(param)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)  # gamma 초기화
                init.constant_(m.bias, 0)    # beta 초기화

    def loss(self, z_hat, z, reduction='mean'):
        if z_hat is None:
            return torch.tensor(1e6, device=self.params['device'])
            
        mse = self.mse_loss(z_hat, z, reduction=reduction)
        loss = mse
        
        if self.params['cos_weight'] > 0:
            cos_loss = self.cos_loss(z_hat, z, reduction=reduction)
            if not torch.isnan(cos_loss).any():
                loss = loss + self.params['cos_weight'] * cos_loss
                
        if self.params['l1_weight'] > 0:
            l1 = self.l1_loss()
            if not torch.isnan(l1).any():
                loss = loss + self.params['l1_weight'] * l1
                
        return loss, mse, cos_loss, l1
    
    def mse_loss(self, z_hat, z, reduction='mean'):
        # gradient clipping 추가
        z_hat = torch.clamp(z_hat, min=-1e6, max=1e6)
        
        # nan 체크
        if torch.isnan(z_hat).any() or torch.isnan(z).any():
            print("NaN detected in mse_loss inputs")
            return torch.tensor(1e6, device=self.params['device'])
            
        criterion = nn.MSELoss(reduction=reduction)
        if reduction == 'none':
            return criterion(z_hat, z).mean(dim=-1)
        return criterion(z_hat, z)
    
    def cos_loss(self, z_hat, z, reduction='mean'):
        try:
            target = torch.ones(z_hat.size(0), device=z_hat.device)
            criterion = nn.CosineEmbeddingLoss(reduction=reduction)
            return criterion(z_hat, z, target)
        except RuntimeError as e:
            print(f"Runtime Error in cos_loss: {str(e)}")
            print(f"z_hat shape: {z_hat.shape}, device: {z_hat.device}")
            print(f"z shape: {z.shape}, device: {z.device}")
            return torch.tensor(0.0, device=self.params['device'])
    
    def l1_loss(self):
        total_params = sum(p.numel() for p in self.PlasDynDict.parameters())
        l1_sum = sum(p.abs().sum() for p in self.PlasDynDict.parameters())
        avg_l1 = l1_sum / total_params
        return torch.log1p(avg_l1)
    
class ResAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'ResAutoencoder'
        super(ResAE_PlasDyn, self).__init__(**params)

class LogAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'LogAutoencoder'
        super(LogAE_PlasDyn, self).__init__(**params)

class LogResAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'LogResAutoencoder'
        super(LogResAE_PlasDyn, self).__init__(**params)

class VariationalAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'VariationalAutoencoder'
        super(VariationalAE_PlasDyn, self).__init__(**params)

class VariationalResAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'VariationalResAutoencoder'
        super(VariationalResAE_PlasDyn, self).__init__(**params)

class VariationalLogAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'VariationalLogAutoencoder'
        super(VariationalLogAE_PlasDyn, self).__init__(**params)    

class VariationalLogResAE_PlasDyn(AE_PlasDyn):
    def __init__(self,**params):
        params['encoder'] = 'VariationalLogResAutoencoder'
        super(VariationalLogResAE_PlasDyn, self).__init__(**params)

