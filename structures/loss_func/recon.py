import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

class ReconstructionLoss(nn.Module):
    def __init__(self, **params):
        super(ReconstructionLoss, self).__init__()
        self.params = params

        loss_fns_map = {
            'l2': self.l2_loss,
            'l2_loss': self.l2_loss,
            'mse': self.l2_loss,
            'mse_loss': self.l2_loss,
            'l1': self.l1_loss,
            'l1_loss': self.l1_loss,
            'mae': self.l1_loss,
            'mae_loss': self.l1_loss,
            'huber': self.huber_loss,
            'huber_loss': self.huber_loss,
            'smooth_l1': self.smooth_l1_loss,
            'smooth_l1_loss': self.smooth_l1_loss,
            'cosine': self.cosine_loss,
            'cosine_loss': self.cosine_loss,
            'binary_cross_entropy': self.binary_cross_entropy_loss,
            'binary_cross_entropy_with_logits': self.binary_cross_entropy_with_logits_loss,
            'kl_divergence': self.kl_divergence_loss,
        }

        self.loss_fns = [loss_fns_map[loss_fn.lower()] for loss_fn in self.params.get('loss_fns', ['l2'])]

    def forward(self, x: Tensor, x_hat: Tensor):
        loss = 0
        for loss_fn in self.loss_fns:
            loss += loss_fn(x, x_hat)
        return loss
    
    def l2_loss(self, x: Tensor, x_hat: Tensor):
        return F.mse_loss(x, x_hat)
    
    def l1_loss(self, x: Tensor, x_hat: Tensor):
        return F.l1_loss(x, x_hat)
    
    def huber_loss(self, x: Tensor, x_hat: Tensor):
        return F.huber_loss(x, x_hat)
    
    def smooth_l1_loss(self, x: Tensor, x_hat: Tensor):
        return F.smooth_l1_loss(x, x_hat)
    
    def cosine_loss(self, x: Tensor, x_hat: Tensor):
        return F.cosine_loss(x, x_hat)
    
    def binary_cross_entropy_loss(self, x: Tensor, x_hat: Tensor):
        return F.binary_cross_entropy(x, x_hat)
    
    def binary_cross_entropy_with_logits_loss(self, x: Tensor, x_hat: Tensor):
        return F.binary_cross_entropy_with_logits(x, x_hat)
    
    def kl_divergence_loss(self, x: Tensor, x_hat: Tensor):
        return F.kl_divergence(x, x_hat)
    

