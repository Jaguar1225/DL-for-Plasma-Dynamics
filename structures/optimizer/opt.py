import torch.optim as optim
from .opt_base import Opt_base
from .scheduler import Scheduler

class Opt(Opt_base,Scheduler):
    def __init__(self, **params):
        super(Opt, self).__init__()
        self.params = params
        self.opt_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
        }

        self.params.setdefault('optimizer', 'adamW')
        self.params.setdefault('optimizer_params',
                          {
                              'params': self.training_layers.parameters(),
                              'lr': 0.001,
                              'betas': (0.9, 0.999),
                              'eps': 1e-8,
                              'weight_decay': 0.0001,
                          })
        self.params.setdefault('scheduler', 'reduceLROnPlateau')
        self.params.setdefault('scheduler_params',
                          {
                              'mode': 'min',
                              'factor': 0.1,
                              'patience': 10,
                              'threshold': 1e-4,
                              'min_lr': 0.0001,
                              'eps': 1e-8,
                          })
        self.optimizer = self.opt_map[self.params['optimizer'].lower()](**self.params['optimizer_params'])
        Scheduler.__init__(self, self.optimizer, **self.params['scheduler_params'])

    '''
    layer update function
    '''

    def update_layer(self):
        self.params['optimizer_params']['params'] = self.training_layers.parameters()
        self.optimizer = self.opt_map[self.params['optimizer'].lower()](**self.params['optimizer_params'])
        Scheduler.__init__(self, self.optimizer, **self.params['scheduler_params'])

    '''
    functions for optimizer
    '''

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


    


