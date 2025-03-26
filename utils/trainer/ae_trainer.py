from structures import autoencoder
from structures import layers
from tqdm import tqdm
import torch

import os
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

        for n in range(self.params['num_layers']):
            temp_loss = []
            while True:
                hidden_dim_list.append(hidden_dim)
                pbar_layer.set_postfix({'hidden_dim': hidden_dim_list})

                self.model.add_encoder_layer(
                    self.partial_layer(input_dim, hidden_dim)
                )
                self.model.add_decoder_layer(
                    self.partial_layer(hidden_dim, input_dim)
                )
                self.model.update_layer()

                loss = self.model.train(self.params['num_epochs'])

                if len(temp_loss) < 2:
                    temp_loss.append(loss)
                    hidden_dim = hidden_dim // 2
                else:
                    if (loss-temp_loss[-1])/(temp_loss[-1]-temp_loss[-2]) > 1e-1:
                        self.model.delete_encoder_layer()
                        self.model.delete_decoder_layer()
                        self.model.add_encoder_layer(removed_encoder_layer.to(self.params['device']))
                        self.model.add_decoder_layer(removed_decoder_layer.to(self.params['device']))
                        input_dim = self.model.encoder_layers[-1].output_dim
                        hidden_dim = input_dim
                        hidden_dim_list.pop(-1)
                        hidden_dim_list.append(hidden_dim)
                        break
                    else:
                        temp_loss.append(loss)
                        temp_loss.pop(0)
                        hidden_dim = hidden_dim // 2

                removed_encoder_layer = self.model.delete_encoder_layer()
                removed_decoder_layer = self.model.delete_decoder_layer()

                if removed_encoder_layer is not None:
                    removed_encoder_layer.to(torch.device('cpu'))
                if removed_decoder_layer is not None:
                    removed_decoder_layer.to(torch.device('cpu'))

                hidden_dim_list.pop(-1)
                if hidden_dim < 1:
                    break
                pbar_layer.update(1)
            if hidden_dim == 1:
                break
        pbar_layer.close()
        os.makedirs(f'models/{self.params["model"]}_model', exist_ok=True)
        torch.save(self.model, f'models/{self.params["model"]}_model/hidden_dim_{("_").join([str(layer) for layer in hidden_dim_list])}.pth')
        return hidden_dim_list
            
    def partial_layer(self,input_dim, hidden_dim):
        return self.layer_map[self.params['layer_type']](
            input_dim=input_dim, output_dim=hidden_dim,
            activation_function=self.params['activation_function']
        ).to(self.params['device'])