'''

24-06-04 21:27

Jaguar

RNN update -> Plasma model
    1. Separate plasma feature and equipment feature from Z feature
    2. Used Gated Linear Unit to describe the change of plasma equipment

'''
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from utils.Writer import StepWriter

import time
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class TDM(nn.Module):
    def __init__(self,in_dimension):
        super().__init__()
        self.in_dimension = in_dimension
        self.rho = nn.Parameter(torch.ones((1,in_dimension)))
        self._initialize_weights()
        
    def forward(self,z,time):
        return torch.exp(-time/(self.rho+1e-10))*z
    
    def _initialize_weights(self):
        self.rho = nn.Parameter(torch.ones((1,self.in_dimension)))

def glu(a,b):
    return a * torch.sigmoid(b)

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
    
class UnitCoder(nn.Module):
    def __init__(self,in_dimension, out_dimension):
        super(UnitCoder,self).__init__()
        self.Linear = nn.Sequential(
            nn.Linear(in_dimension,out_dimension),
            nn.BatchNorm1d(out_dimension),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self,X):
        Transpose = False
        if len(X.size()) > 2:
            N,T,D = X.size()
            X = X.reshape(-1,D).contiguous()
            Transpose = True
        H = self.Linear(X)
        if Transpose:
            H =  H.reshape(N,T,-1).contiguous()
        return H
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class PlasmaModel(nn.Module):
    def __init__(self,hidden_dimension, num_variable):
        super(PlasmaModel,self).__init__()

        self.hidden_dimension = hidden_dimension

        self.GLU = nn.ModuleDict()
        self.GLU['a_h'] = GLU(hidden_dimension,hidden_dimension)
        self.GLU['mu_z'] = GLU(hidden_dimension,hidden_dimension)
        self.GLU['a_z'] = GLU(hidden_dimension,hidden_dimension)

        self.tdm = TDM(hidden_dimension)

        self.Encoder_V = UnitCoder(num_variable,hidden_dimension)

    def forward(self,Z,V,time, H0 = None):

        try:
            _,T,_ = Z.size()
            mode = True
        except IndexError:
            mode = False

        if H0 == None:
            if mode:
                H_next = torch.zeros_like(Z[:,0])
            else:
                H_next = torch.zeros_like(Z)
    
        else:
            H_next = torch.tensor(H0,device = next(self.parameters()).device)

        V = self.Encoder_V(V)

        if mode:
            for t in range(T-1):
                H_next = self.pred_h(H_next, Z[:,t])
                Z_next = self.pred_z(H_next, V[:,t+1], Z[:,t], time)
        else:
            H_next = self.pred_h(H_next, Z)
            Z_next = self.pred_z(H_next, V, Z, time)

        return Z_next, H_next

    def pred_h(self,H,Z):
        return self.GLU['a_h'](Z,Z) + H
    
    def pred_z(self,h_next,v_next,z,time):
        return self.tdm(self.GLU['mu_z'](z,h_next),time) + self.GLU['a_z'](v_next,h_next)


class SAE_PlasmaModel(nn.Module,StepWriter):
    def __init__(self,**params):
        super(SAE_PlasmaModel,self).__init__()

        params.setdefault("device", device)

        params.setdefault('base_dimension',3648)
        params.setdefault('hidden_dimension', 16)
        params.setdefault('layer_dimension',[1824,912,456,228,114,57,32,16])
        params.setdefault('sequence_length',8)
        params.setdefault('process_variables',7)
        params.setdefault('lr_opt',None)
        self.params = params

        self.__init_params__()

        #Train_params
        self.params.setdefault('batch_size', 2**14)
        self.params.setdefault('k_fold',0.9)


        self.params.setdefault("dtype", torch.float32)
        self.params.setdefault("device", device)
        
        self.date = time.strftime('%Y-%m-%d', time.localtime())
        self.params.setdefault("random_seed", 0)
        
        self.params.setdefault("Writer_dir", f'logs/{self.date}')
        self.params.setdefault("keyword","")


    def __init_params__(self):
        self.__init_Encoder()
        self.__init_PlasmaModel()
        self.to(self.params['device'])


    def forward(self,X,V,time):
        Z = self.encode(X)
        Z_hat = self.PlasmaModel(Z,V,time)
        return Z_hat        

    def encode(self,X):
        Tranpose = False
        if len(X.size()) == 1:
            X = X.unsqueeze(0)
        elif len(X.size()) == 2:
            pass
        else:
            N,T,D = X.shape
            X = X.reshape(-1,self.params['base_dimension'])
            Tranpose = True
        
        for n in range(len(self.params['layer_dimension'])):
            X = self.EncoderDict[f'L{n}'](X)

        if Tranpose:
            X = X.view(-1, T,self.params['hidden_dimension'])
        return X
    
    def __init_Encoder(self):
        self.EncoderDict = nn.ModuleDict()
        in_dimension = self.params['base_dimension']
        for n, d in enumerate(self.params['layer_dimension']):
            self.EncoderDict[f'L{n}'] = UnitCoder(in_dimension,d)
            in_dimension = d

    def __init_PlasmaModel(self):
        self.PlasmaModel = PlasmaModel(self.params['hidden_dimension'],self.params['process_variables'])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.xavier_uniform_(param)

    def update(self, XV, R =1, epoch=500):
        if self.params['k_fold'] >= 2:
            self.__update_kfold(XV,epoch)
        else:
            self.__update_none_repeat(XV, R, epoch)
            
    def __update_kfold(self, XV, epoch):
        
        kfold = KFold(n_splits=self.params["k_fold"],shuffle=True, random_state=0).split
        best_model = None, None
        best_loss = None
        for f, (idx_t, idx_v) in enumerate(kfold(XV)):
            self._step_writer_init(fold=f)
            m = 0

            train_idx = torch.utils.data.SubsetRandomSampler(idx_t)
            val_idx = torch.utils.data.SubsetRandomSampler(idx_v)

            trainloader = DataLoader(XV, batch_size=self.params['batch_size'],
                                         sampler=train_idx)
            
            valloader = DataLoader(XV, batch_size=self.params['batch_size'],
                                      sampler=val_idx)
            self.__init_params__()
            self._step_writer_init(fold = f)
            self._memory_writer_init()

            if self.params['lr_opt'] == None:
                self.params["optimizer"] = optim.Adam(
                    self.parameters(), lr=1e-3
                    )
            else:
                self.params["optimizer"] = optim.Adam(
                    self.params['lr_opt'])
                
            self.params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.params["optimizer"], 'min',
                factor = 0.5, patience = 2**8, min_lr = 1e-5
                )

            kn = 0
            for e in tqdm(range(int(epoch))):
                kn += 1
                for x, v, t in trainloader:
                    m += 1
                    
                    loss = self.__step_update(x, v, t)
                    self._step_summary(m, loss)
                    
                high_error_x_list = []
                high_error_v_list = []
                
                for n, (x, v, t) in enumerate(valloader):
                    z_hat, h_hat = self(x,v,t)
                    z_next = self.encode(x[:,-1])
                    batch_loss_val = self.loss(z_hat,z_next,reduction='none').mean(dim=1)
                    top_indicies = torch.argsort(batch_loss_val, descending=True)[:100]
                    
                    high_error_x_list.append(x[top_indicies])
                    high_error_v_list.append(v[top_indicies])
                    
                    if n == 0:
                        loss_val = batch_loss_val.sum()
                    else:
                        loss_val += batch_loss_val.sum()
                
                x = torch.cat(high_error_x_list,dim=0)
                v = torch.cat(high_error_v_list,dim=0)
                        
                self._step_image_summary('/OES image/original',x[:,-1],e)
                
                self._step_summary(m, loss_val/(n+1), "/val loss")
                self._memory_check(kn)
                self.__step_lr_update(loss_val/(n+1))
                self._step_summary(m, self.params['optimizer'].param_groups[0]['lr'], '/lr')

                if self._step_lr_early_stop(e):
                    break
                #self._step_image_kernel_summary(e)

            if f==0:
                best_loss = loss_val.item()
                best_model = f, self.state_dict()
                continue

            if best_loss > loss_val.item():
                best_loss = loss_val.item()
                best_model = f, self.state_dict()

            self._summary_writer_close()
        self.load_state_dict(best_model[1])
        self._step_writer_save(
            self.params["Writer_dir"]+f'/{self.params["keyword"]} fold {best_model[0]}'
            )
        print(f"{best_model[0]} fold is best model")
        self._memory_writer_close()
        del self.params['image_writer']

    def __update_none_repeat(self,XV,R,epoch):

        best_model = None, None
        best_loss = None

        for r in range(R):
            loss_val = self.__update_none(XV,epoch)
            if r==0 or best_loss > loss_val.item():
                best_loss = loss_val.item()
                best_model = r+1, self.state_dict()
                self._step_writer_save(
                    self.params["Writer_dir"]+f'/{self.params["keyword"]}'
                    )
            self.__init_params__()
        self.load_state_dict(best_model[1])
        print(f"{best_model[0]} fold is best model")
        
    def __update_none(self, XV, epoch):

        if self.params['lr_opt'] == None:
            self.params["optimizer"] = optim.Adam(
                self.parameters(), lr=1e-3
                )
        else:
            self.params["optimizer"] = optim.Adam(
                self.params['lr_opt'])
        
        self.params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
            self.params["optimizer"], 'min',
            factor = 0.5, patience = 2**8, min_lr = 1e-5
            )
        
        indices = torch.randperm(len(XV))
        num_train_samples = int(len(XV) * self.params['k_fold'])

        train_idx = torch.utils.data.SubsetRandomSampler(indices[:num_train_samples])
        val_idx = torch.utils.data.SubsetRandomSampler(indices[num_train_samples:])

        trainloader = DataLoader(
            XV, batch_size=self.params['batch_size'],sampler=train_idx)
            
        valloader = DataLoader(
                XV, batch_size=self.params['batch_size'],sampler=val_idx)        
        kn = 0
        m = 0
        self._step_writer_init()
        self._memory_writer_init()

        for e in tqdm(range(int(epoch))):

            kn += 1
            for x, v, t  in trainloader:
                m += 1
                    
                loss = self.__step_update(x, v, t)
                self._step_summary(m, loss)

            high_error_x_list = []
            high_error_v_list = []
                
            for n, (x, v, t) in enumerate(valloader):
                z_hat, h_hat = self(x,v,t)
                z_next = self.encode(x[:,-1])
                batch_loss_val = self.loss(z_hat,z_next,reduction='none').mean(dim=1)
                top_indicies = torch.argsort(batch_loss_val, descending=True)[:100]
                    
                high_error_x_list.append(x[top_indicies])
                high_error_v_list.append(v[top_indicies])
                    
                if n == 0:
                    loss_val = batch_loss_val.mean()
                else:
                    loss_val += batch_loss_val.mean()
                
            x = torch.cat(high_error_x_list,dim=0)
            v = torch.cat(high_error_v_list,dim=0)
                        
            self._step_image_summary('/OES image/original',x[:,-1],e)
                
            self._step_summary(m, loss_val/(n+1), "/val loss")
            self._memory_check(kn)
            self.__step_lr_update(loss_val/(n+1))
            self._step_summary(m, self.params['optimizer'].param_groups[0]['lr'], '/lr')

            if self._step_lr_early_stop(e):
                break

        self._summary_writer_close()        
        self._memory_writer_close()
        self._step_writer_save(
            self.params["Writer_dir"]+f'/{self.params["keyword"]}'
            )
        del self.params['image_writer']
        return loss_val/(n+1)
    
    def __step_update(self,x,v,t):
        z_hat,h_hat = self(x,v,t)
        z_next = self.encode(x[:,-1])
        loss = self.loss(z_hat,z_next)
        self.params['optimizer'].zero_grad()
        loss.backward()
        self.params['optimizer'].step()
        return loss.item()
    
    def __step_lr_update(self, loss_val):
        self.params['scheduler'].step(loss_val)

    def _step_lr_early_stop(self,e):
        if self.params['scheduler'].num_bad_epochs > self.params['scheduler'].patience + 1 :
            print(f"Early stopping at epoch {e}")
            return True
        else:
            return False 
    
    def loss(self, z_hat, z_next, reduction = 'mean'):
        criterion = nn.MSELoss(reduction=reduction)
        return criterion(z_hat,z_next)