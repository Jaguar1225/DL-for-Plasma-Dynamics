from utils.trains import RNNTrains
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

class AE_RNN(nn.Module, RNNTrains):
    def __init__(self, **params):
        super(AE_RNN, self).__init__()
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
            'hidden_dimension': 32,
            'sequence_length': 8,
            'process_variables': 16,
            'batch_size': 2**14,
            'optimizer_rnn': None,
            'dropout_rate': 0,
            'l1_weight': 0.01,
            'cos_weight': 1,
            'dtype': torch.float32,
            'device': device,
            'random_seed': 0,
            'Writer_dir': f'logs/{self.date}',
            'keyword': "",
            'encoder': 'Autoencoder',
            'encoder_model': None,
            'save_path': None,
            'trial': 0,
            'hidden_dim': 16,
            'layer_depth': 2
        }
        
        # 기존 params 값을 유지하면서 기본값 설정
        for key, value in defaults.items():
            self.params.setdefault(key, value)

    def _load_encoder_model(self):
        """인코더 모델 로드"""
        if self.params['encoder_model'] is None:
            encoder_file_dir = f"model/{self.params['encoder']}/*"  # 모든 인코더 폴더를 포함하는 패턴

            # 해당 디렉토리에서 모든 인코더 폴더 검색
            encoders = glob.glob(encoder_file_dir)

            # 최신 인코더 폴더 찾기
            if encoders:
                latest_encoder_dir = max(encoders)  # 최신 폴더 찾기
                latest_encoders = glob.glob(os.path.join(latest_encoder_dir, '*'))  # 최신 폴더 내의 모든 파일 검색

                # 원하는 hidden 값과 일치하는 파일 찾기
                matching_files = []
                for file in latest_encoders:
                    # 파일 이름에서 hidden 뒤의 숫자를 찾기 위한 정규 표현식
                    match = re.search(r'_(\d+)_\d+\.pth$', file)
                    if match:
                        hidden_value = int(match.group(1))  # 찾은 숫자를 정수로 변환
                        if hidden_value == self.params['hidden_dimension']:
                            matching_files.append(file)  # 원하는 파일을 리스트에 추가

                # 결과 출력
                if matching_files:
                    encoder_model_name = matching_files[0]  # 첫 번째 일치하는 파일 선택
                else:
                    encoder_model_name = None  # 일치하는 파일이 없을 경우 None으로 설정
                    print("Warning: No matching files found for the desired hidden value.")
            else:
                print("Warning: No encoder folders found in the specified directory.")

            torch.serialization.add_safe_globals([eval(self.params['encoder'])])            
            self.encoder_model = torch.load(encoder_model_name, weights_only=False)
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

    def forward(self, X, L):
        Z = self.encode(X)
        if torch.isnan(Z).any():
            print("NaN detected after encoding")
            return None
            
        Z_hat = self.rnn(Z, L)
        if torch.isnan(Z_hat).any():
            print("NaN detected after RNN")
            return None
            
        return Z_hat
    
    def encode(self,X):
        return self.encoder_model.encode(X)
    
    def rnn(self,Z,V):
        N,T,D = V.size()
        V = V.reshape(-1,self.params['process_variables']).contiguous()

        for n in range(len(self.params['layer_dimension'])):
            V = self.RNNDict[f'E{n}'](V)

        V = V.view(-1,T,self.params['hidden_dimension'])
        _, Z_hat = self.RNNDict['RNN'](V,h=Z)
        return Z_hat
    
    def init_params(self):
        self.init_rnn()
        self.to(self.params['device'])

    def init_rnn(self):
        self.RNNDict = nn.ModuleDict()
        in_dimension = self.params['process_variables']

        for n, d in enumerate(self.params['layer_dimension']):
            if n == len(self.params['layer_dimension'])-1:
                self.RNNDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=False,  # use_log 전달
                    use_residual=False  # use_residual 전달
                )
            else:
                self.RNNDict[f'E{n}'] = UnitCoder(
                    in_dimension, d, 
                    normalization='batchnorm', 
                    activation='tanh', 
                    dropout_rate=self.params['dropout_rate'],
                    use_log=False,  # use_log 전달
                    use_residual=False  # use_residual 전달
                )
            in_dimension = d

        self.RNNDict['RNN'] = RNN(
            hidden_size=self.params['hidden_dimension'],
            dropout=self.params['dropout_rate']
        )
                
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.RNNDict.modules():  # .modules()를 사용하여 모든 하위 모듈에 접근
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, RNN):
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
        total_params = sum(p.numel() for p in self.RNNDict.parameters())
        l1_sum = sum(p.abs().sum() for p in self.RNNDict.parameters())
        avg_l1 = l1_sum / total_params
        return torch.log1p(avg_l1)
    
class ResAE_RNN(AE_RNN):
    def __init__(self,**params):
        params['encoder'] = 'ResAutoencoder'
        super(ResAE_RNN, self).__init__(**params)

class LogAE_RNN(AE_RNN):
    def __init__(self,**params):
        params['encoder'] = 'LogAutoencoder'
        super(LogAE_RNN, self).__init__(**params)

class LogResAE_RNN(AE_RNN):
    def __init__(self,**params):
        params['encoder'] = 'LogResAutoencoder'
        super(LogResAE_RNN, self).__init__(**params)