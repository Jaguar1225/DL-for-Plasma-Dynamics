import torch
import numpy as np

from structures import autoencoder
from structures import layers
from utils.report.plot import Plotter

from tqdm import tqdm

class AE_Trainer:
    def __init__(self, **params):
        self.params = params

        self.model_map = {
            'autoencoder': autoencoder.Autoencoder,
            'variational_autoencoder': autoencoder.VariationalAutoencoder,
        }

        self.model = self.model_map[self.params['model']](**params)
        self.model.to(self.params['device'])
        self.model.params['shuffle'] = True

        self.layer_map = {
            'unit_coder':   layers.UnitCoder,
            'log_unit_encoder': layers.UnitLogEncoder,
            'log_unit_decoder': layers.UnitLogDecoder,
            'unit_transformer': layers.UnitTransformer,
        }

    def train(self):
        input_dim = self.params['input_dim']
        hidden_dim = self.params['input_dim']

        hidden_dim_list = []
        pbar_layer = tqdm(total=self.params['num_layers'], desc="Layer", leave=False)

        loss_log = np.zeros((self.params['num_layers'], int(np.log2(self.params['input_dim']))))

        for n in range(self.params['num_layers']):
            for m in range(int(np.log2(input_dim))):

                hidden_dim_list.append(hidden_dim)

                self.model.add_encoder_layer(
                    self.partial_layer(input_dim, hidden_dim)
                )
                self.model.add_decoder_layer(
                    self.partial_layer(hidden_dim, input_dim)
                )
                self.model.update_layer()
                
                loss_log[n][int(np.log2(self.params['input_dim'])-m)] = self.model.train(self.params['num_epochs'])

                if self.saturation_detection(loss_log, n, int(np.log2(self.params['input_dim'])-m)):
                    sat_encoder_layer = removed_encoder_layer.clone()
                    sat_decoder_layer = removed_decoder_layer.clone()
                    sat_hidden_dim = removed_hidden_dim
                    
                removed_encoder_layer = self.model.delete_encoder_layer()
                removed_decoder_layer = self.model.delete_decoder_layer()
                removed_hidden_dim = hidden_dim

                if removed_encoder_layer is not None:
                    removed_encoder_layer.to('cpu')
                if removed_decoder_layer is not None:
                    removed_decoder_layer.to('cpu')
    
                hidden_dim = hidden_dim//2
                hidden_dim_list.pop(-1)

            input_dim = sat_hidden_dim
            hidden_dim = sat_hidden_dim

            hidden_dim_list.append(hidden_dim)

            self.model.add_encoder_layer(sat_encoder_layer.to(self.params['device']))
            self.model.add_decoder_layer(sat_decoder_layer.to(self.params['device']))

        Plotter.plot_heatmap(loss_log, 
                             title = 'Loss Log', 
                             xlabel = 'Hidden Dimension', 
                             ylabel = 'Number of Layers', 
                             save_name = 'loss_log.png',
                             dpi = 300)
            
    def partial_layer(self,input_dim, hidden_dim):
        return self.layer_map[self.params['layer_type']](
            input_dim=input_dim, output_dim=hidden_dim,
            activation_function=self.params['activation_function']
        ).to(self.params['device'])
    
    def saturation_detection(self, loss_log: np.ndarray, row: int, col: int, criterion: float = 1e-1):
        if loss_log.shape[-1] - col < 2:
            return False
        
        sat_score = loss_log[row][col] - loss_log[row][col+1]
        sat_score = sat_score / (loss_log[row][col+2] - loss_log[row][col+1])
        
        if sat_score > criterion:
            return True
        else:
            return False