from tqdm import tqdm
import pickle
import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from utils.Writer import StepWriter

class AutoencoderTrains(StepWriter):
    def update(self, X, R=1, k_fold=1, epoch=1e3, use_writer=False, save = False):
        if k_fold >= 2:
            train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss = \
                self._update_kfold(X, k_fold, epoch, use_writer, save)
        else:
            train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss = \
                self._update_none_repeat(X, R, epoch, use_writer, save)
        return train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss
    
    def _update_kfold(self, X, k_fold, epoch, use_writer, save):
        kf = KFold(n_splits=k_fold, shuffle=True)
        best_model = None, None
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        for n, (train_idx, val_idx) in enumerate(kf.split(X)):
            trainloader = DataLoader(X[train_idx], batch_size=self.params['batch_size'])
            valloader = DataLoader(X[val_idx], batch_size=self.params['batch_size'])
            
            self.init_params()
            self.optimizer_init()
            
            train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss = \
                self._update_none(trainloader, valloader, epoch, n, use_writer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss, best_train_recon_loss, best_train_cos_loss, best_train_l1_loss = \
                    train_loss, train_recon_loss, train_cos_loss, train_l1_loss
                best_val_loss, best_val_recon_loss, best_val_cos_loss, best_val_l1_loss = \
                    val_loss, val_recon_loss, val_cos_loss, val_l1_loss
                best_model = n,self.state_dict()

            self._cleanup()
            
        self.load_state_dict(best_model[1])
        return best_train_loss, best_train_recon_loss, best_train_cos_loss, best_train_l1_loss, \
                best_val_loss, best_val_recon_loss, best_val_cos_loss, best_val_l1_loss
    
    def _update_none_repeat(self, X, R, epoch, use_writer, save):
        best_model = None, None
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        for r in range(R):
            # Writer 초기화
            if use_writer:
                self._step_writer_init(fold=r)
            
            indices = torch.randperm(len(X))
            split_idx = int(0.8 * len(X))
            
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            
            trainloader = DataLoader(X, batch_size=self.params['batch_size'],
                                  sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            valloader = DataLoader(X, batch_size=self.params['batch_size'],
                                sampler=torch.utils.data.SubsetRandomSampler(val_idx))

            self.optimizer_init()
            
            loss, recon_loss, cos_loss, l1_loss, val_loss, \
                val_recon_loss, val_cos_loss, val_l1_loss = \
                self._update_none(trainloader, valloader, epoch, r, use_writer)
            
            self._cleanup()
            
            current_loss = val_loss
            
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_train_loss, best_train_recon_loss, best_train_cos_loss, best_train_l1_loss = \
                    loss, recon_loss, cos_loss, l1_loss
                best_val_loss, best_val_recon_loss, best_val_cos_loss, best_val_l1_loss = \
                    val_loss, val_recon_loss, val_cos_loss, val_l1_loss
                best_model = r, self.state_dict()
                
                if use_writer:
                    log_dir = self.params["Writer_dir"] + f'/{self.params["keyword"]}'
                    if save:
                        self._step_writer_save(log_dir, self.params['save_path'])
            
            self.init_params()
            
            if use_writer:
                self._summary_writer_close()

        self.load_state_dict(best_model[1])
        print(f"{best_model[0]} repeat is best model. Best validation loss: {best_val_loss}")
        
        return best_train_loss, best_train_recon_loss, best_train_cos_loss, best_train_l1_loss, \
                best_val_loss, best_val_recon_loss, best_val_cos_loss, best_val_l1_loss
    
    def _update_none(self, trainloader, valloader, epoch, r, use_writer):
        self.step = 0  # step 초기화
        
        # epoch 진행바
        pbar_epoch = tqdm(range(epoch), desc=f'Training (Repeat {r+1})', leave=True)
        
        for e in pbar_epoch:
            self.train()
            total_loss = 0
            total_recon_loss = 0
            total_cos_loss = 0
            total_l1_loss = 0
            
            # batch 진행바
            pbar_batch = tqdm(enumerate(trainloader), 
                             total=len(trainloader),
                             desc=f'Epoch {e+1}/{epoch}',
                             leave=False)
            
            for i, batch in pbar_batch:
                # batch 처리 및 loss 계산
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x, l = batch
                    else:
                        x = batch[0]
                        l = None
                else:
                    x = batch
                    l = None
                
                # forward pass
                z_hat, z = self._process_batch(x, l = l)
                loss, recon_loss, cos_loss, l1_loss = self.loss(z_hat, z)
                
                # backward pass
                self._step_update(loss)
                
                # 현재 loss 표시
                current_loss = loss.item()
                total_loss += current_loss
                total_recon_loss += recon_loss
                total_cos_loss += cos_loss
                total_l1_loss += l1_loss
                
                avg_loss = total_loss / (i + 1)
                avg_recon_loss = total_recon_loss / (i + 1)
                avg_cos_loss = total_cos_loss / (i + 1)
                avg_l1_loss = total_l1_loss / (i + 1)
                
                pbar_batch.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
                
                self.step += 1
                
                # 메모리 정리
                del z_hat, z, loss
                if i % 5 == 0:
                    torch.cuda.empty_cache()

            if use_writer and 'Writer' in self.params and self.step % 10 == 0:
                self._summary_writer_update(avg_loss, avg_recon_loss, avg_cos_loss, avg_l1_loss)
            # validation
            val_loss, val_recon_loss, val_cos_loss, val_l1_loss = self._process_validation(valloader, use_writer)
            self._step_lr_update(val_loss)
            
            # epoch 진행바 업데이트
            pbar_epoch.set_postfix({
                'train_loss': f'{avg_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            if self._step_lr_early_stop(e):
                print("\nEarly stopping triggered")
                break
        
        return avg_loss, avg_recon_loss, avg_cos_loss, avg_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss
    
    def _summary_writer_update(self, loss, recon_loss, cos_loss, l1_loss):
        self._step_summary(loss)
        self._step_summary(recon_loss, title='/train mse loss')
        self._step_summary(cos_loss, title='/train cos loss')
        self._step_summary(l1_loss, title='/train l1 loss')
    
    def _summary_writer_update_val(self, val_loss, recon_loss, cos_loss, l1_loss):
        self._step_summary(val_loss, title='/val loss')
        self._step_summary(recon_loss, title='/val mse loss')
        self._step_summary(cos_loss, title='/val cos loss')
        self._step_summary(l1_loss, title='/val l1 loss')
        
    def _process_batch(self, x, l=None):
        if l is not None:  # RNN/PlasDyn case
            z_hat = self(x[:,0], l[:,1:])
            z = self.encode(x[:,-1])
        else:  # Autoencoder case
            x_hat = self(x)
            return x_hat, x
        return z_hat, z

    def _process_validation(self, valloader, use_writer):
        self.eval()
        total_val_loss = 0
        total_recon_loss = 0
        total_cos_loss = 0
        total_l1_loss = 0
        n_val_batches = 0
        
        # validation 진행바
        pbar_val = tqdm(enumerate(valloader), 
                        total=len(valloader),
                        desc='Validation',
                        leave=False)
        
        with torch.no_grad():
            for i, batch in pbar_val:
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x, l = batch
                    else:
                        x = batch[0]
                        l = None
                else:
                    x = batch
                    l = None
                
                z_hat, z = self._process_batch(x, l=l)
                loss, recon_loss, cos_loss, l1_loss = self.loss(z_hat, z)
                    
                current_loss = loss.item()
                total_val_loss += current_loss
                total_recon_loss += recon_loss
                total_cos_loss += cos_loss
                total_l1_loss += l1_loss
                n_val_batches += 1

                avg_val_loss = total_val_loss / n_val_batches
                avg_val_recon_loss = total_recon_loss / n_val_batches
                avg_val_cos_loss = total_cos_loss / n_val_batches
                avg_val_l1_loss = total_l1_loss / n_val_batches
                    
                # 현재 validation loss 표시
                pbar_val.set_postfix({
                    'val_loss': f'{current_loss:.4f}',
                    'avg_val_loss': f'{avg_val_loss:.4f}'
                    })
                    
                del z_hat, z, loss
                torch.cuda.empty_cache()
        
        self.train()
        if use_writer and self.step % 10 == 0 and 'Writer' in self.params:
            self._summary_writer_update_val(avg_val_loss, avg_val_recon_loss, \
                                            avg_val_cos_loss, avg_val_l1_loss)
        return avg_val_loss, avg_val_recon_loss, avg_val_cos_loss, avg_val_l1_loss

    def _step_update(self, loss):
        """
        손실에 대한 역전파 수행
        Args:
            loss: 계산된 손실값 (requires_grad=True)
        """
        self.optimizer.zero_grad()
        if loss.requires_grad:  # grad 필요한지 확인
            loss.backward()
            self._gradient_check()  # gradient 체크 추가
            self.optimizer.step()
        else:
            print("Warning: Loss does not require gradients")

    def _step_lr_update(self, loss_val):
        self.scheduler.step(loss_val)

    def _step_lr_early_stop(self, e):
        if self.scheduler.num_bad_epochs > self.scheduler.patience + 1:
            print(f"Early stopping at epoch {e}")
            return True
        return False
    
    def optimizer_init(self):
        if self.params["optimizer_sae"] is None:
            self.optimizer = optim.Adam(
                self.parameters(), lr=1e-2
            )
        else:
            try:
                self.optimizer = optim.Adam(
                    self.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_sae"].x[0], self.params["optimizer_sae"].x[1]),
                    weight_decay=self.params["optimizer_sae"].x[2]
                )
            except:
                self.optimizer = optim.Adam(
                    self.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_sae"][0], self.params["optimizer_sae"][1]),
                    weight_decay=self.params["optimizer_sae"][2]
                )
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.5, patience=2**8, min_lr=1e-5
        )
    
    def _gradient_check(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: {name} has invalid gradients")
                    param.grad.clamp_(-1, 1)

    def _cleanup(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        self.T_destination = None
        torch.cuda.empty_cache()

class RNNTrains(AutoencoderTrains):
    def optimizer_init(self):
        if self.params["optimizer_rnn"] is None:
            self.optimizer = optim.Adam(
                self.RNNDict.parameters(), lr=1e-2
            )
        else:
            try:
                self.optimizer = optim.Adam(
                    self.RNNDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_rnn"].x[0], self.params["optimizer_rnn"].x[1]),
                    weight_decay=self.params["optimizer_rnn"].x[2]
                )
            except:
                self.optimizer = optim.Adam(
                    self.RNNDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_rnn"][0], self.params["optimizer_rnn"][1]),
                    weight_decay=self.params["optimizer_rnn"][2]
                )

        torch.nn.utils.clip_grad_norm_(self.RNNDict.parameters(), max_norm=1.0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.5, patience=2**8, min_lr=1e-5
        )

    def _gradient_check(self):
        for name, param in self.RNNDict.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: {name} has invalid gradients")
                    param.grad.clamp_(-1, 1)

class PlasDynTrains(RNNTrains):
    def optimizer_init(self):
        if self.params["optimizer_plasdyn"] is None:
            self.optimizer = optim.Adam(
                self.PlasDynDict.parameters(), lr=1e-2
            )
        else:
            try:
                self.optimizer = optim.Adam(
                    self.PlasDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasdyn"].x[0], self.params["optimizer_plasdyn"].x[1]),
                    weight_decay=self.params["optimizer_plasdyn"].x[2]
                )
            except:
                self.optimizer = optim.Adam(
                    self.PlasDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasdyn"][0], self.params["optimizer_plasdyn"][1]),
                    weight_decay=self.params["optimizer_plasdyn"][2]
                )

        torch.nn.utils.clip_grad_norm_(self.PlasDynDict.parameters(), max_norm=1.0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.5, patience=2**8, min_lr=1e-5
        )

    def _gradient_check(self):
        for name, param in self.PlasDynDict.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: {name} has invalid gradients")
                    param.grad.clamp_(-1, 1)

    def _cleanup(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        torch.cuda.empty_cache()

class PlasVarDynTrains(RNNTrains):
    def optimizer_init(self):
        if self.params["optimizer_plasvardyn"] is None:
            self.optimizer = optim.Adam(
                self.PlasVarDynDict.parameters(), lr=1e-2
            )
        else:
            try:
                self.optimizer = optim.Adam(
                    self.PlasVarDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasvardyn"].x[0], self.params["optimizer_plasvardyn"].x[1]),
                    weight_decay=self.params["optimizer_plasvardyn"].x[2]
                )
            except:
                self.optimizer = optim.Adam(
                    self.PlasVarDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasvardyn"][0], self.params["optimizer_plasvardyn"][1]),
                    weight_decay=self.params["optimizer_plasvardyn"][2]
                )

        torch.nn.utils.clip_grad_norm_(self.PlasVarDynDict.parameters(), max_norm=1.0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.5, patience=2**8, min_lr=1e-5
        )

    def _gradient_check(self):
        for name, param in self.PlasVarDynDict.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: {name} has invalid gradients")
                    param.grad.clamp_(-1, 1)

class PlasEquipVarDynTrains(RNNTrains):
    def optimizer_init(self):
        if self.params["optimizer_plasequipvardyn"] is None:
            self.optimizer = optim.Adam(
                self.PlasEquipVarDynDict.parameters(), lr=1e-2
            )
        else:
            try:
                self.optimizer = optim.Adam(
                    self.PlasEquipVarDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasequipvardyn"].x[0], self.params["optimizer_plasequipvardyn"].x[1]),
                    weight_decay=self.params["optimizer_plasequipvardyn"].x[2]
                )
            except:
                self.optimizer = optim.Adam(
                    self.PlasEquipVarDynDict.parameters(), lr=1e-2,
                    betas=(self.params["optimizer_plasequipvardyn"][0], self.params["optimizer_plasequipvardyn"][1]),
                    weight_decay=self.params["optimizer_plasequipvardyn"][2]
                )

        torch.nn.utils.clip_grad_norm_(self.PlasEquipVarDynDict.parameters(), max_norm=1.0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.5, patience=2**8, min_lr=1e-5
        )

    def _gradient_check(self):
        for name, param in self.PlasEquipVarDynDict.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: {name} has invalid gradients")
                    param.grad.clamp_(-1, 1)