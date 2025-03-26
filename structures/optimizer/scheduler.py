import torch.optim as optim
import torch

class Scheduler:
    def __init__(self, optimizer, **params):
        self.optimizer = optimizer

        scheduler_map = {
            'step': optim.lr_scheduler.StepLR,
            'reducelronplateau': optim.lr_scheduler.ReduceLROnPlateau,
            'cosineannealinglr': optim.lr_scheduler.CosineAnnealingLR,
            'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'multisteplr': optim.lr_scheduler.MultiStepLR,
            'exponentialLR': optim.lr_scheduler.ExponentialLR,
            'constantLR': optim.lr_scheduler.ConstantLR,
            'linearLR': optim.lr_scheduler.LinearLR,
        }

        params.setdefault('scheduler', 'reducelronplateau')
        params.setdefault('scheduler_params',
                          {
                              'mode': 'min',
                              'factor': 0.1,
                              'patience': 10,
                              'threshold': 1e-4,
                              'min_lr': 0.0001,
                              'eps': 1e-8,
                          })
        
        self.scheduler = scheduler_map[params['scheduler'].lower()](self.optimizer, **params['scheduler_params'])

    def scheduler_step(self, epoch:int=None, loss:torch.Tensor=None):
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if loss is None:
                raise ValueError("ReduceLROnPlateau scheduler requires loss/metric value")
            self.scheduler.step(loss)
        else:
            if epoch is None:
                raise ValueError("Other schedulers require epoch value")
            self.scheduler.step(epoch)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def early_stopping(self, metric, patience, min_delta):
        if metric < self.scheduler.get_last_lr()[0] - min_delta:
            self.scheduler.last_epoch = 0
        else:
            self.scheduler.last_epoch += 1
        if self.scheduler.last_epoch >= patience:
            return True
        return False
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def get_lr(self):
        return self.scheduler.get_lr()
    
    def get_last_epoch(self):
        return self.scheduler.last_epoch
    
    def get_base_lrs(self):
        return self.scheduler.base_lrs
    
    def get_num_batches(self):
        return self.scheduler.num_batches
    
    def get_interval(self):
        return self.scheduler.interval
    
    def get_frequency(self):
        return self.scheduler.frequency 
    
