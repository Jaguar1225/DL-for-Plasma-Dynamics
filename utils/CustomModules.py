'''

24-06-04 21:27

Jaguar

RNN update -> Plasma model
    1. Separate plasma feature and equipment feature from Z feature
    2. Used Gated Linear Unit to describe the change of plasma equipment

'''
import pickle
import glob
import re
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def glu(a,b):
    return a * torch.sigmoid(b)

def mish(x):
    return x * torch.tanh(F.softplus(x))

def activation_function(activation):
    if activation is None:
        return nn.Identity()
    if isinstance(activation,str):
        # 활성화 함수 설정
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'mish':
            return nn.Mish() if hasattr(nn, 'Mish') else lambda x: x * torch.tanh(F.softplus(x))
        elif activation == 'glu':
            return lambda x: x * torch.sigmoid(x)
        elif activation == 'swish':  # 또는 SiLU
            return nn.SiLU()
        elif activation == 'none':
            return nn.Identity()
    elif callable(activation):
        return activation
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def normalization_function(normalization,**kwargs):
    if normalization is None:
        return nn.Identity()
    if isinstance(normalization,str):
        if normalization == 'batchnorm':
            return nn.BatchNorm1d(**kwargs)
        elif normalization == 'layernorm':
            return nn.LayerNorm(**kwargs)
        elif normalization == 'instancenorm':
            return nn.InstanceNorm1d(**kwargs)
        elif normalization == 'groupnorm':
            return nn.GroupNorm(**kwargs)
        elif normalization == 'syncbatchnorm':
            return nn.SyncBatchNorm(**kwargs)
        elif normalization == 'none':
            return nn.Identity()
    elif callable(normalization):
        return normalization
    else:
        raise ValueError(f"Unknown normalization function: {normalization}")

class MishGate(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x, y):
        # Mish 게이팅에 스케일링 추가
        gate = self.scale * (1 + F.tanh(F.softplus(y)))  # 0~1 범위로 조정
        return x * gate
    
class GLU(nn.Module):
    def __init__(self,in_dimension,out_dimension):
        super().__init__()
        self.W = nn.Linear(in_dimension,out_dimension)
        self.Wg = nn.Linear(in_dimension,out_dimension)
        self._initialize_weights()
        
    def forward(self,a,b):
        return glu(self.W(a),self.Wg(b))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class TDM(nn.Module):
    def __init__(self, hidden_dimension):
        super().__init__()
        # rho 초기화: 0.05 ~ 2.0 범위로 확장
        self.rho = nn.Parameter(torch.rand(hidden_dimension) * 1.95 + 0.05)
        
    def forward(self, x, t):
        # ReLU나 Softplus로 양수 보장
        positive_rho = F.softplus(self.rho)  # 또는 F.relu(self.rho)
        return x * torch.exp(-positive_rho * t)

class UnitCoder(nn.Module):
    def __init__(self,in_dimension, out_dimension,normalization=None,activation=None,
                 dropout_rate=0.0, use_log=False, use_residual=False):
        super(UnitCoder,self).__init__()

        self.use_log = use_log
        self.use_residual = use_residual

        if use_log:
            self.log = Log()

        self.linear1 = nn.Linear(in_dimension, out_dimension)
        
        if use_residual and in_dimension != out_dimension:
            self.linear2 = nn.Linear(in_dimension, out_dimension)
        else:
            self.linear2 = None

        kwargs = {}
        if normalization is not None:
            if normalization == 'batchnorm':
                kwargs['num_features'] = out_dimension
            elif normalization == 'layernorm':
                kwargs['normalized_shape'] = out_dimension

        # 정규화 레이어 설정
        self.norm = normalization_function(normalization,**kwargs)

        # 활성화 함수 설정
        self.act = activation_function(activation)
        
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.normalization = normalization
        self.activation = activation

        self._initialize_weights()

    def forward(self,X):
        Transpose = False
        if len(X.size()) > 2:
            N,T,D = X.size()
            X = X.reshape(-1,D).contiguous()
            Transpose = True

        if self.use_residual:
            identity = X

        if self.use_log:
            X = self.log(X)
        H = self.linear1(X)
        if self.normalization != None:
            H = self.norm(H)
        if self.activation != None:
            H = self.act(H)
        if self.dropout_rate:
            H = self.dropout(H)

        if self.use_residual:
            identity = self.linear2(identity)
            H += identity

        if Transpose:
            H =  H.reshape(N,T,-1).contiguous()
        return H
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier uniform 초기화
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # Bias는 0으로 초기화

            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)  # BatchNorm의 가중치는 1로 초기화
                init.constant_(m.bias, 0)    # Bias는 0으로 초기화

            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)  # LayerNorm의 가중치는 1로 초기화
                init.constant_(m.bias, 0)    # Bias는 0으로 초기화

class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()
    
    def forward(self, X):
        return (F.relu(X) + 1e-7).log()
    
class RNN(nn.Module):
    def __init__(self, hidden_size,  dropout=0.0,batch_first=True):
        super(RNN, self).__init__()
        # 활성화 함수 선택
        self.rnn_layer = nn.Linear(hidden_size*2, hidden_size)

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.batch_first = batch_first
    def forward(self, input, h=None): #input: (N,T,D) hx Z: (N,D)
        if self.batch_first:
            input = input.transpose(0,1)
        if h is None:
            h = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)

        outs = []
        for t, x in enumerate(input):
            i = torch.cat((x,h),dim=-1)
            h = self.rnn_layer(i)
            if self.training and self.dropout_rate > 0:
                h = self.dropout(h)
            outs.append(h.unsqueeze(0))
        out = torch.stack(outs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, h


class PlasDyn(nn.Module):
    def __init__(self, hidden_dimension, gate_type='glu',use_residual=False,batch_first=True):
        super().__init__()

        self.hidden_dimension = hidden_dimension
        GateLayer = GLU if gate_type == 'glu' else MishGate
        self.gates = nn.ModuleDict({
            'mu_z': GateLayer(hidden_dimension, hidden_dimension),
            'a_z': GateLayer(hidden_dimension, hidden_dimension)
        })
        self.use_residual = use_residual
        self.batch_first = batch_first
        self._initialize_weights()

    def forward(self, input,h=None):
        if self.batch_first:
            input = input.transpose(0,1)
        if h == None:
            h = torch.zeros(1, input.size(1), self.hidden_dimension, device=input.device)

        outs = []
        for i, x in enumerate(input):
            h = self.gates['mu_z'](h,h) + self.gates['a_z'](x,h)
        return h
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class PlasVar(nn.Module):
    def __init__(self, hidden_dimension, gate_type='glu', num_layers=1, num_heads=1,
                 activation=None, normalization=None, use_residual=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.head_dimension = hidden_dimension // num_heads

        self.query = nn.Linear(hidden_dimension, hidden_dimension)
        self.key = nn.Linear(hidden_dimension, hidden_dimension)
        self.value = nn.Linear(hidden_dimension, hidden_dimension)

        kwargs = {}
        if normalization is not None:
            if normalization == 'batchnorm':
                kwargs['num_features'] = hidden_dimension
            elif normalization == 'layernorm':
                kwargs['normalized_shape'] = hidden_dimension

        self.activation = activation_function(activation)
        self.normalization = normalization_function(normalization,**kwargs)
        self.use_residual = use_residual
        self._initialize_weights()
        
    def attention(self, pv):
        batch_size = pv.size(0)
        Q = self.query(pv)
        K = self.key(pv)
        V = self.value(pv)
        dk = V.size(-1)
        Q = Q.view(batch_size, self.num_heads, self.head_dimension)
        K = K.view(batch_size, self.num_heads, self.head_dimension)
        V = V.view(batch_size, self.num_heads, self.head_dimension)
        QK = Q@K.transpose(-2,-1)
        attn = QK@V/math.sqrt(dk)
        attn = attn.view(-1,self.hidden_dimension)
        if self.activation is not None:
            attn = self.activation(attn)
        if self.normalization is not None:
            attn = self.normalization(attn)
        return attn

    def forward(self, pv):
        if len(pv.size()) > 2:
            N,T,D = pv.size()
            pv = pv.reshape(-1,D).contiguous()
            Transpose = True

        for _ in range(self.num_layers):
            attn = self.attention(pv)
            if self.use_residual:
                pv = pv + attn
        if Transpose:
            pv = pv.reshape(N,T,-1).contiguous()
        return pv
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
class PlasVarDyn(nn.Module):
    def __init__(self, hidden_dimension, gate_type='glu', dyn_num_layers=1, var_num_layers=1, var_num_heads=1,
                 activation=None, normalization=None, use_residual=False,batch_first=True):
        super().__init__()
        self.plas_dyn = PlasDyn(hidden_dimension, gate_type=gate_type,use_residual=False,batch_first=batch_first)
        self.plas_var = PlasVar(hidden_dimension, gate_type=gate_type, num_layers=var_num_layers, num_heads=var_num_heads,
                 activation=activation, normalization=normalization, use_residual=use_residual)
        self._initialize_weights()

    def forward(self, V, h=None):
        V = self.plas_var(V)
        h = self.plas_dyn(V,h=h)
        return h
    
    def _initialize_weights(self):
        self.plas_dyn._initialize_weights()
        self.plas_var._initialize_weights()

class PlasEquip(nn.Module):
    def __init__(self, hidden_dimension, sequence_length, gate_type='glu', batch_first=True):
        super().__init__()

        self.hidden_dimension = hidden_dimension
        self.batch_first = batch_first

        # 게이팅 메커니즘 선택
        GateLayer = GLU if gate_type == 'glu' else MishGate
        
        self.gates = nn.ModuleDict({
            'a_h': GateLayer(hidden_dimension, hidden_dimension),
            'mu_z': GateLayer(hidden_dimension, hidden_dimension),
            'a_z': GateLayer(hidden_dimension, hidden_dimension),
        })
    
    def pred_h(self,z,h):
        # GLU를 사용한 H 업데이트
        return self.gates['a_h'](z, h) + h

    def pred_z(self, z,v_next,h_next):
        # 기존 GLU 구조 유지
        return self.gates['mu_z'](z, z) + \
               self.gates['a_z'](v_next, h_next)

    def forward(self, z, V, h=None):

        if h is None:
            h = torch.zeros(1, z.size(1), self.hidden_dimension, device=z.device)

        for v in V:
            h = self.pred_h(z,h)
            z = self.pred_z(z,v,h)
        return z
    
class PlasVarEquip(nn.Module):
    def __init__(self, hidden_dimension, sequence_length, gate_type='glu', batch_first=True,
                 activation=None, normalization=None, use_residual=False,num_layers=1,num_heads=1):
        super().__init__()
        self.plas_equip = PlasEquip(hidden_dimension, sequence_length, gate_type, batch_first)
        self.plas_var = PlasVar(hidden_dimension, gate_type, num_layers, num_heads, activation, normalization, use_residual)
        self._initialize_weights()
    def forward(self, z, V, h=None):
        V = self.plas_var(V)
        z = self.plas_equip(z,V,h=h)
        return z
    def _initialize_weights(self):
        self.plas_equip._initialize_weights()
        self.plas_var._initialize_weights()


class PlasVarEquipDyn(nn.Module):
    def __init__(self, hidden_dimension, sequence_length, gate_type='glu', batch_first=True,direct_model='PlasVarDyn'):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.sequence_length = sequence_length
        self.batch_first = batch_first
        self.direct_model = direct_model
        self._load_direct_model()

        self.plas_var_equip = PlasVarEquip(hidden_dimension, sequence_length, gate_type, batch_first, direct_model)
    
    def _load_direct_model(self):
        if self.direct_model is not None:
            if isinstance(self.direct_model, str):
                direct_file_dir = f"model/{self.direct_model}/*"  # 모든 인코더 폴더를 포함하는 패턴

                # 해당 디렉토리에서 모든 인코더 폴더 검색
                directs_models = glob.glob(direct_file_dir)

                # 최신 인코더 폴더 찾기
                if directs_models:
                    latest_direct_dir = max(directs_models)  # 최신 폴더 찾기
                    latest_directs = glob.glob(os.path.join(latest_direct_dir, '*'))  # 최신 폴더 내의 모든 파일 검색

                # 원하는 hidden 값과 일치하는 파일 찾기
                    matching_files = []
                    for file in latest_directs:
                        # 파일 이름에서 hidden 뒤의 숫자를 찾기 위한 정규 표현식
                        match = re.search(r'_(\d+)_\d+\.pth$', file)
                        if match:
                            sequence_length = int(match.group(1))  # 찾은 숫자를 정수로 변환
                            if sequence_length == self.sequence_length:
                                matching_files.append(file)  # 원하는 파일을 리스트에 추가

                # 결과 출력
                    if matching_files:
                        direct_model_name = matching_files[0]  # 첫 번째 일치하는 파일 선택
                    else:
                        direct_model_name = None  # 일치하는 파일이 없을 경우 None으로 설정
                    print("Warning: No matching files found for the desired hidden value.")
                else:
                    print("Warning: No direct model folders found in the specified directory.")

            torch.serialization.add_safe_globals([eval(self.direct_model)])            
            self.direct_model = torch.load(direct_model_name, weights_only=False)
            for param in self.direct_model.parameters():
                param.requires_grad = False
            self.direct_model.eval()

    def estimate_h0(self, z,zT,VT):
        with torch.no_grad():
            z = self.direct_model(z,VT)
        return zT/z
    
    def forward(self, ZT, V2T):
        if self.batch_first:
            V2T = V2T.transpose(0,1)
            T = self.sequence_length//2
            VT = V2T[:T]
        h0 = self.estimate_h0(ZT[0],ZT[-1],VT)
        Z = self.plas_var_equip(ZT[0],V2T,h=h0)
        return Z
    
    def _initialize_weights(self):
        self.plas_var_equip._initialize_weights()

