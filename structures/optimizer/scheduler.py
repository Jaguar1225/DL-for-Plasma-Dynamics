import torch.optim as optim

class Scheduler:
    def __init__(self, optimizer, **params):
        self.optimizer = optimizer

        scheduler_map = {
            'step': optim.lr_scheduler.StepLR,
            'reduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
            'cosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
            'cosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'multiStepLR': optim.lr_scheduler.MultiStepLR,
            'exponentialLR': optim.lr_scheduler.ExponentialLR,
            'constantLR': optim.lr_scheduler.ConstantLR,
            'constantLR_with_warmup': optim.lr_scheduler.ConstantLR_with_warmup,
            'linearLR': optim.lr_scheduler.LinearLR,
            'linearLR_with_warmup': optim.lr_scheduler.LinearLR_with_warmup,
        }

        params.setdefault('scheduler', 'reduceLROnPlateau')
        params.setdefault('scheduler_params',
                          {
                              'mode': 'min',
                              'factor': 0.1,
                              'patience': 10,
                              'threshold': 1e-4,
                              'min_lr': 0.0001,
                              'eps': 1e-8,
                          })
        
        self.scheduler = scheduler_map[params['scheduler']](self.optimizer, **params['scheduler_params'])

    def scheduler_step(self):
        self.scheduler.step()

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
    
