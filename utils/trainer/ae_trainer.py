from structures import autoencoder
from structures import layers

class AE_Trainer:
    def __init__(self, **params):
        self.params = params

        self.model_map = {
            'autoencoder': autoencoder.Autoencoder,
            'variational_autoencoder': autoencoder.VariationalAutoencoder,
        }

        self.model = self.model_map[self.params['model']](**params)
        self.model.to(self.params['device'])

        self.layer_map = {
            'unit_coder':   layers.UnitCoder,
            'log_unit_encoder': layers.UnitLogEncoder,
            'log_unit_decoder': layers.UnitLogDecoder,
            'unit_transformer': layers.UnitTransformer,
        }

    def train(self):
        temp_loss = None
        input_dim = self.params['input_dim']
        hidden_dim = self.params['input_dim']
        for n in range(self.params['num_layers']):
            while True:

                self.model.add_encoder_layer(
                    self.partial_layer(input_dim, hidden_dim)
                )
                self.model.add_decoder_layer(
                    self.partial_layer(hidden_dim, input_dim)
                )
                self.model.update_layer()

                loss = self.model.train(self.params['num_epochs'])

                if temp_loss == None:
                    temp_loss = loss
                else:
                    if (loss-temp_loss)/temp_loss > 0.1:
                        break
                    else:
                        temp_loss = loss

                self.model.delete_encoder_layer()
                self.model.delete_decoder_layer()

                if hidden_dim == 1:
                    break
                hidden_dim = hidden_dim//2
            if hidden_dim == 1:
                break
            
    def partial_layer(self,input_dim, hidden_dim):
        return self.layer_map[self.params['layer_type']](
            input_dim=input_dim, output_dim=hidden_dim,
            activation_function=self.params['activation_function']
        ).to(self.params['device'])