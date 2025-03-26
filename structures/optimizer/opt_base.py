import torch.nn as nn

class Opt_base(nn.Module):
    def __init__(self, **params):
        super(Opt_base, self).__init__()
        self.training_layers = nn.ModuleList()
        
    def add_training_layer(self, layer : nn.Module):
        self.training_layers.append(layer)

    def delete_training_layer(self):
        self.training_layers.pop()

    def clear_training_layer(self):
        """ModuleList를 새로운 빈 ModuleList로 초기화합니다."""
        self.training_layers = nn.ModuleList()
    
    def set_training_layers(self, training_layers):
        self.training_layers = training_layers
