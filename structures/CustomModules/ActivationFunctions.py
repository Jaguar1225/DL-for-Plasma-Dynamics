import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class act(nn.Module):
    def __init__(self, **params):
        super(act, self).__init__()
        self.params = params  # 파라미터를 저장
        
        activation_map = {
            'relu': ReLU,
            'leakyrelu': LeakyReLU,
            'elu': ELU,
            'gelu': GELU,
            'prelu': PReLU,
            'softplus': Softplus,
            'softplusbeta': SoftplusBeta,
            'tanh': Tanh,
            'mish': Mish,
            'mishbeta': MishBeta,
            'sigmoid': Sigmoid,
            'swish': Swish,
            'swishbeta': SwishBeta,
            # GLU 계열은 별도 처리가 필요
        }
        
        act_name = self.params.get('activation_function', '').lower()
        
        if act_name in activation_map:
            self.activation_function = activation_map[act_name]()
        elif act_name in ['glu', 'bilinearglu', 'reglu', 'geluglu', 'swiglu']:
            # GLU 계열은 별도 처리 (두 개의 입력을 받기 때문)
            glu_map = {
                'glu': GLU,
                'bilinearglu': BilinearGLU,
                'reglu': ReGLU,
                'geluglu': GELUGLU,
                'swiglu': SwiGLU
            }
            self.activation_function = glu_map[act_name]()
            self.is_glu = True
        else:
            # 기본값은 ReLU로 설정
            self.activation_function = ReLU()
            
    def forward(self, x: Tensor, y: Tensor = None):
        if hasattr(self, 'is_glu') and self.is_glu:
            if y is None:
                # GLU는 두 개의 텐서를 입력받아야 함
                # 입력이 하나라면 반으로 나누어 사용
                split_size = x.size(1) // 2
                a, b = torch.split(x, split_size, dim=1)
                return self.activation_function(a, b)
            return self.activation_function(x, y)
        return self.activation_function(x)
    
'''
ReLU function and its variants

*ReLU function:
    ReLU(x) = max(0, x)

*LeakyReLU function:
    LeakyReLU(x) = max(0, x) + alpha * min(0, x)
    alpha is a learnable parameter

*ELU function:
    ELU(x) = max(0, x) + alpha * (exp(min(0, x)) - 1)
    alpha is a learnable parameter
    
*GELU function:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

*PReLU function:
    PReLU(x) = max(0, x) + alpha * min(0, x)
    alpha is a learnable parameter

'''


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: Tensor):
        return F.relu(x)
    
class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def forward(self, x: Tensor):
        return F.leaky_relu(x)
    
class ELU(nn.Module):
    def __init__(self):
        super(ELU, self).__init__()

    def forward(self, x: Tensor):
        return F.elu(x)
        
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x: Tensor):
        return F.gelu(x)

class PReLU(nn.Module):
    def __init__(self):
        super(PReLU, self).__init__()

    def forward(self, x: Tensor):
        return F.prelu(x)


'''
Softplus function and its variants

*Softplus function:
    Softplus(x) = log(1 + exp(x))

*Softplus-beta function:
    Softplus-beta(x) = log(1 + exp(beta * x))
    beta is a learnable parameter
'''
class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, x: Tensor):
        return F.softplus(x)
    
class SoftplusBeta(nn.Module):
    def __init__(self):
        super(SoftplusBeta, self).__init__()
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor):
        return F.softplus(self.beta * x)
    
'''
Tanh function and its variants

*Tanh function:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
'''
class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: Tensor):
        return F.tanh(x)

'''
Mish function and its variants

*Mish function:
    Mish(x) = x * tanh(softplus(x))

*Mish-beta function:
    Mish-beta(x) = x * tanh(beta * softplus(x))
    beta is a learnable parameter

'''
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x: Tensor):
        return x * torch.tanh(F.softplus(x))
    
class MishBeta(nn.Module):
    def __init__(self):
        super(MishBeta, self).__init__()
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor):
        return x * torch.tanh(self.beta * F.softplus(x))
    
'''
Sigmoid function and its variants

*Sigmoid function:
    Sigmoid(x) = 1 / (1 + exp(-x))
'''

    
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: Tensor):
        return F.sigmoid(x)
    
'''
Swish functions and their variants

*Swish function:
    Swish(x) = x * sigmoid(x)
    beta is a learnable parameter

*Swish-beta function:
    Swish-beta(x) = x * sigmoid(beta * x)
    beta is a learnable parameter
'''

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: Tensor):
        return x * torch.sigmoid(self.beta * x)
    
class SwishBeta(nn.Module):
    def __init__(self):
        super(SwishBeta, self).__init__()
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor):
        return 2 * x * torch.sigmoid(self.beta * x)

'''
Gated Linear Unit and their variants

*Gated Linear Unit (GLU) : Start of the Gated Linear Units family
    GLU(a, b) = (aW + b) * sigma(aV + c)
    W, V, b, c are learnable parameters

*Bilinear Gated Linear Unit (BilinearGLU): More simple than GLU by removing the sigmoid function
    BilinearGLU(a, b) = (aW + b) * (aV + c)
    W, V, b, c are learnable parameters

*ReLU-Gated Linear Unit (ReGLU): More simple than GLU by substituting the sigmoid function with a ReLU function
    ReGLU(a, b) = (aW + b) * ReLU(aV + c)
    W, V, b, c are learnable parameters

*GELU-Gated Linear Unit (GELUGLU): More smooth gradient for ReGLU
    GELUGLU(a, b) = (aW + b) * GELU(aV + c)
    W, V, b, c are learnable parameters

*Swish-Gated Linear Unit (SwiGLU): More effective using Swish function instead of sigmoid function
    SwiGLU(a, b) = (aW + b) * sigma(aV + c)
    W, V, b, c are learnable parameters
'''
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        self.W = nn.Parameter(torch.randn(2, 2))
        self.V = nn.Parameter(torch.randn(2, 2))
        self.b = nn.Parameter(torch.randn(2))
        self.c = nn.Parameter(torch.randn(2))

    def forward(self, a: Tensor, b: Tensor):
        return (a @ self.W + self.b) * torch.sigmoid(a @ self.V + self.c)
    
class BilinearGLU(GLU):
    def __init__(self):
        super(BilinearGLU, self).__init__()
    def forward(self, a: Tensor, b: Tensor):
        return (a @ self.W + self.b) * (a @ self.V + self.c)
    
class ReGLU(GLU):
    def __init__(self):
        super(ReGLU, self).__init__()
    def forward(self, a: Tensor, b: Tensor):
        return (a @ self.W + self.b) * F.relu(a @ self.V + self.c)

class GELUGLU(GLU):
    def __init__(self):
        super(GELUGLU, self).__init__()
    def forward(self, a: Tensor, b: Tensor):
        return (a @ self.W + self.b) * F.gelu(a @ self.V + self.c)

class SwiGLU(GLU):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.Swish = Swish()
    def forward(self, a: Tensor, b: Tensor):
        return (a @ self.W + self.b) * self.Swish(a @ self.V + self.c)





