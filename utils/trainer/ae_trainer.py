import torch
import numpy as np

from structures import autoencoder
from structures import layers
from utils.report.plot import Plotter
import os

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
        prev_sat_idx = 1
        input_dim = self.params['input_dim']
        hidden_dim = self.params['input_dim']

        hidden_dim_list = []
        pbar_layer = tqdm(total=self.params['num_layers'], desc="Layer", leave=False)

        loss_log = np.zeros((self.params['num_layers'], int(np.log2(self.params['input_dim']))))

        for n in range(self.params['num_layers']):
            sat = False
            sat_hidden_dim = None

            for m in range(int(np.log2(input_dim))):

                temp_loss = None

                hidden_dim_list.append(hidden_dim)
                pbar_layer.set_postfix({'Hidden dim ': str(hidden_dim_list), 'Sat. hidden dim ': str(sat_hidden_dim)})

                self.model.add_encoder_layer(
                        self.partial_layer(input_dim, hidden_dim)
                    )
                self.model.add_decoder_layer(
                        self.partial_layer(hidden_dim, input_dim)
                    )
                self.model.update_layer()
                
                loss_log[n,loss_log.shape[-1]-prev_sat_idx-m] = self.model.train(self.params['num_epochs'])

                if not sat:
                    if self.saturation_detection(loss_log, prev_sat_idx, n, m):
                        sat_encoder_layer = removed_encoder_layer.clone()
                        sat_decoder_layer = removed_decoder_layer.clone()
                        sat_hidden_dim = removed_hidden_dim
                        prev_sat_idx = m
                        sat = True
                else:
                    pass
                            
                removed_encoder_layer = self.model.delete_encoder_layer()
                removed_decoder_layer = self.model.delete_decoder_layer()
                removed_hidden_dim = hidden_dim

                if removed_encoder_layer is not None:
                    removed_encoder_layer.to('cpu')
                if removed_decoder_layer is not None:
                    removed_decoder_layer.to('cpu')
        
                hidden_dim = hidden_dim//2
                hidden_dim_list.pop(-1)
                if hidden_dim < 1:
                    break

            Plotter(f'{self.params["model"]}').plot_heatmap(loss_log[:n], 
                             title = 'Loss Log', 
                             xlabel = 'Hidden Dimension', 
                             ylabel = 'Number of Layers', 
                             save_name = 'loss_log.png',
                             dpi = 300)
            
            if sat_hidden_dim is None:
                break

            input_dim = sat_hidden_dim
            hidden_dim = sat_hidden_dim

            hidden_dim_list.append(hidden_dim)

            self.model.add_encoder_layer(sat_encoder_layer.to(self.params['device']))
            self.model.add_decoder_layer(sat_decoder_layer.to(self.params['device']))

            pbar_layer.update(1)

        os.makedirs(f'models/{self.params["model"]}', exist_ok=True)
        torch.save(self.model, f'models/{self.params["model"]}/{self.params["model"]}_{"_".join(map(str, hidden_dim_list))}.pth')

    def partial_layer(self,input_dim, hidden_dim):
        return self.layer_map[self.params['layer_type']](
            input_dim=input_dim, output_dim=hidden_dim,
            activation_function=self.params['activation_function']
        ).to(self.params['device'])
    
    def saturation_detection(self, loss_log: np.ndarray, prev_sat_idx: int, row: int, col: int, criterion: float = 10):
        """
        Loss 증가 패턴을 감지하는 함수
        
        Args:
            loss_log: Loss 기록 배열
            prev_sat_idx: 이전 포화 인덱스
            row: 현재 레이어 인덱스
            col: 현재 hidden dimension 인덱스
            criterion: 감지 기준값 (기본값: 0.1 = 10% 증가)
        """

        col_rev = loss_log.shape[-1] - col -1

        #if col - prev_sat_idx< 2:
        #    return False
        
        # 현재 loss와 이전 loss의 상대적 증가율 계산
        current_loss = loss_log[row][col_rev]
        #prev_loss = loss_log[row][col_rev+1]
        # loss가 10% 이상 증가하면 포화로 판단
        #if current_loss > (1+criterion)*prev_loss:
        if current_loss > criterion:
            return True
            
        return False