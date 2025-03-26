import torch.optim as optim
from .opt_base import Opt_base
from .scheduler import Scheduler
from tqdm import tqdm

from utils.writer import SummaryWriter
from utils.dataloader import Train_Data_Set

class Opt(Opt_base, Scheduler, SummaryWriter):
    def __init__(self):
        super(Opt, self).__init__()
        
        #optimizer
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
                              'lr': 0.001,
                              'betas': (0.9, 0.999),
                              'eps': 1e-8,
                              'weight_decay': 0.0001,
                          })

        #scheduler
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


    '''
    layer update function
    '''

    def update_optimizer(self):
        self.optimizer = self.opt_map[self.params['optimizer'].lower()](
            self.training_layers.parameters(),
            **self.params['optimizer_params']
            )
        Scheduler.__init__(self, self.optimizer, **self.params['scheduler_params'])

    '''
    functions for optimizer
    '''

    def optimizer_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        
    def train(self, num_epochs):

        self.optimizer = self.opt_map[self.params['optimizer'].lower()](
            self.training_layers.parameters(), 
            **self.params['optimizer_params']
            )
        
        Scheduler.__init__(self, self.optimizer, **self.params['scheduler_params'])

        #tensorboard
        self.params.setdefault('log_dir', f'./logs/{self.params["model"]}/hidden_dim_{
            ("_").join([str(layer.output_dim) for layer in self.training_layers])
            }')
        SummaryWriter.__init__(self, self.params['log_dir'])

        pbar_epoch = tqdm(total=num_epochs, desc="Training", leave=True)
        train_data = self.params.get('train_data', Train_Data_Set())

        for epoch in range(num_epochs):
            pbar_batch = tqdm(total=len(train_data), desc="Training", leave=False)
            loss_sum = 0

            for batch_idx, (x, y) in enumerate(train_data):
                loss = self.step(epoch, x, y)
                loss_sum += loss
                pbar_batch.update(1)
                pbar_batch.set_postfix({'loss': loss})

            self.add_scalar(f'loss', loss_sum/len(train_data), epoch)

            pbar_epoch.update(1)
            pbar_epoch.set_postfix({'loss': loss_sum/len(train_data)})

            if self.early_stopping(
                loss_sum/len(train_data), 
                self.params['patience'], self.params['min_delta']):
                break

        pbar_batch.close()
        pbar_epoch.close()

        return loss
    
    def step(self, epoch, *inputs):
        loss = self.loss_fn(*inputs)

        self.zero_grad()
        loss.backward()
        self.optimizer_step()
        self.scheduler_step()

        return loss

    def loss_fn(self, *inputs):
        pass


    


