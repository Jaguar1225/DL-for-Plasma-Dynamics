from structures import autoencoder
from structures import Layers

class AE_Trainer:
    def __init__(self, **params):
        self.params = params

        self.model_map = {
            'autoencoder': autoencoder.Autoencoder,
            'variational_autoencoder': autoencoder.VariationalAutoencoder,
        }

        self.model = self.model_map[self.params['model']](**params)

        self.layer_map = {
            'unit_coder':   Layers.UnitCoder,
            'log_unit_encoder': Layers.UnitLogEncoder,
            'log_unit_decoder': Layers.UnitLogDecoder,
            'unit_transformer': Layers.UnitTransformer,
        }

    def train(self, num_layers, num_epochs):
        temp_loss = None
        input_dim = self.params['input_dim']
        hidden_dim = self.params['input_dim']
        for n in range(num_layers):
            while True:
                self.model.add_encoder_layer(self.partial_layer(input_dim, hidden_dim))
                self.model.add_decoder_layer(self.partial_layer(hidden_dim, input_dim))

                loss = self.model.train(num_epochs)

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
            input_dim=input_dim, hidden_dim=hidden_dim,
            activation_function=self.params['activation_function']
        )