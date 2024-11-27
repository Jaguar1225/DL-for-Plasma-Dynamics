import torch
import os
import gc
from utils.Data_loader import DataLoader
from utils.report import report_to_mail
from utils.debug import debug_class, tolist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Plotter:        
    def final_plot(self):
        pbar = tqdm(total=3, desc="Plotting final plot",leave=True)
        if self.model_type in ["RNN", "PlasDyn"]:
            self.heatmap('Sequence length')
            pbar.update(1)
            self.error_bar('Sequence length')
            pbar.update(1)
            self.minplot('Sequence length')
            pbar.update(1)
        else:
            self.heatmap('Hidden dimension')
            pbar.update(1)
            self.error_bar('Hidden dimension')
            pbar.update(1)
            self.minplot('Hidden dimension')
            pbar.update(1)
        pbar.close()
        self.sub_pbar.close()

    def heatmap(self, x_name):
        def text_plot(pivot, ax):
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.iloc[i, j]
                    if np.isnan(value):
                        continue
                    base, exponent = f'{value:.2e}'.split('e')
                    base = float(base)
                    exponent = int(exponent)
                    color = "white" if value > pivot.values.mean() else "black"
                    if exponent == 0:
                        text_string = f'${base:.2f}$'
                    else:
                        text_string = f'${base:.2f}×10^{{{exponent}}}$'
                    text = ax.text(j, i, text_string,
                                  ha="center", va="center", color=color, fontsize=10, fontweight='bold')
        
        def pivot_plot(df, ax, values, index, columns, aggfunc):
            aggfunc = 'mean' if aggfunc == 'Mean' else 'std'
            pivot = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc)
            median = float(np.nanmedian(pivot.to_numpy().flatten()))
            std = float(np.nanstd(pivot.to_numpy().flatten()))
            
            vmin = median - 0.5*std
            vmax = median + 0.5*std

            ax.set_box_aspect(1)
            im = ax.imshow(pivot, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f'{aggfunc} {values}', fontsize=25, fontweight='bold')
            ax.set_xlabel(columns, fontsize=25, fontweight='bold')
            ax.set_ylabel(index, fontsize=25, fontweight='bold')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, fontsize=20, fontweight='bold')
            ax.set_yticklabels(pivot.index, fontsize=20, fontweight='bold')
            ax.tick_params(axis='both', width=3.0, length=10)
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
            cbar = plt.colorbar(im, ax=ax, label=values)
            cbar.ax.tick_params(labelsize=20)
            cbar.ax.set_ylabel(values, fontsize=20, fontweight='bold')
            text_plot(pivot, ax)
            
        def plotting(df, loss_type, loss_name):
            subplot_size = 8
            fig, axes = plt.subplots(1, 2, figsize=(subplot_size*2.5, subplot_size+2),
                                     gridspec_kw={'width_ratios': [1, 1]})
            ((ax1, ax2)) = axes

            fig.suptitle(f'{loss_type} {loss_name}', fontsize=25, fontweight='bold', y=1.05)
            pivot_plot(df, ax1, f'{loss_type} {loss_name}', x_name, 'Number of layers', 'Mean')
            pivot_plot(df, ax2, f'{loss_type} {loss_name}', x_name, 'Number of layers', 'Std')

            fig.tight_layout()
            plt.savefig(f'Result/{self.model_class.__name__}/{self.save_path}/heatmap_{loss_type}_{loss_name}.png',dpi=300)
            self.plot_writer.add_figure(f'heatmap_{loss_type}_{loss_name}', fig)
            plt.close()
        
        file_path = f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt'
        df = pd.read_csv(file_path, sep=' ', names=['trial', x_name, 'Number of layers', 'Train loss', 'Train recon. loss', 'Train cos loss', 
                                                    'Train l1 loss', 'Validation loss', 'Validation recon. loss', 'Validation cos loss', 'Validation l1 loss'])
        df.astype({'trial': int, x_name: int, 'Number of layers': int})
        df.astype({'Train loss': float, 'Train recon. loss': float, 'Train cos loss': float, 'Train l1 loss': float, 
                    'Validation loss': float, 'Validation recon. loss': float, 'Validation cos loss': float, 'Validation l1 loss': float})
        
        for loss_type in ['Train', 'Validation']:
            for loss_name in ['loss', 'recon. loss', 'cos loss', 'l1 loss']:
                df[f'{loss_type} {loss_name}'] = np.log10(df[f'{loss_type} {loss_name}'])
        
        self.sub_pbar = tqdm(total=8, desc="Plotting heatmap", leave=False)
        for loss_type in ['Train', 'Validation']:
            for loss_name in ['loss', 'recon. loss', 'cos loss', 'l1 loss']:
                plotting(df, loss_type, loss_name)
                self.sub_pbar.update(1)
        self.sub_pbar.close()
                
    def minplot(self, x_name):

        def pivot_plot(df, x_name, ax, loss_name):
            color = {'Train': 'blue', 'Validation': 'red'}
            ax.set_box_aspect(1)
            for loss_type in ['Train', 'Validation']:
                df_pivot = pd.pivot_table(df, values=f'{loss_type} {loss_name}', index=x_name, aggfunc='min')

                np_x = df_pivot.index.to_numpy().flatten()
                np_y = df_pivot.values.flatten()
                ax.plot(np_x, np_y, marker='o', label=f'{loss_type} {loss_name}', color=color[loss_type])
            
            ax.legend(
                prop={'size': 20, 'weight': 'bold'},
                frameon=False,
                loc='best'
            )
            ax.set_title(f'{loss_name}', fontsize=25, fontweight='bold')
            ax.set_xlabel(x_name, fontsize=25, fontweight='bold')
            ax.set_ylabel('Log10 Loss', fontsize=25, fontweight='bold')
            style_axes(ax)

        def plotting(df, x_name):
            subplot_size = 8
            fig, ax = plt.subplots(2, 2, figsize=(subplot_size*2.5, subplot_size*2.5+2),
                                   gridspec_kw={'width_ratios': [1, 1],
                                                'height_ratios': [1, 1]})
            fig.suptitle(f'{x_name}', fontsize=25, fontweight='bold', y=1.05)
            ax = ax.flatten()
            for i, loss_name in enumerate(['loss', 'recon. loss', 'cos loss', 'l1 loss']):
                pivot_plot(df, x_name, ax[i], loss_name)

            fig.tight_layout()
            plt.savefig(f'Result/{self.model_class.__name__}/{self.save_path}/minplot_{x_name}.png',dpi=300)
            self.plot_writer.add_figure(f'minplot_{x_name}', fig)
            plt.close()

        def style_axes(ax):
            ax.grid(True)
            ax.set_facecolor('white')
            ax.spines['left'].set_linewidth(3.0)
            ax.spines['bottom'].set_linewidth(3.0)
            ax.spines['right'].set_linewidth(3.0)
            ax.spines['top'].set_linewidth(3.0)
        
            ax.tick_params(axis='both', width=3.0, length=10,which='major',labelsize=20)
            plt.setp(ax.get_xticklabels(), weight='bold')
            plt.setp(ax.get_yticklabels(), weight='bold')

            ax.set_xlabel(ax.get_xlabel(), fontsize=25, fontweight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontsize=25, fontweight='bold')
            ax.set_title(ax.get_title(), fontsize=25, fontweight='bold', pad=22)
        
        file_path = f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt'
        df = pd.read_csv(file_path, sep=' ', names=['trial', x_name, 'Number of layers', 'Train loss', 'Train recon. loss', 'Train cos loss', 
                                                    'Train l1 loss', 'Validation loss', 'Validation recon. loss', 'Validation cos loss', 'Validation l1 loss'])
        df.astype({'trial': int, x_name: int, 'Number of layers': int})
        df.astype({'Train loss': float, 'Train recon. loss': float, 'Train cos loss': float, 'Train l1 loss': float, 
                    'Validation loss': float, 'Validation recon. loss': float, 'Validation cos loss': float, 'Validation l1 loss': float})
        
        for loss_type in ['Train', 'Validation']:
            for loss_name in ['loss', 'recon. loss', 'cos loss', 'l1 loss']:
                df[f'{loss_type} {loss_name}'] = np.log10(df[f'{loss_type} {loss_name}'])
        
        self.sub_pbar = tqdm(total=4, desc="Plotting minplot", leave=False)
        for x in [x_name, 'Number of layers']:
            plotting(df, x)
            self.sub_pbar.update(1)
        self.sub_pbar.close()
 
    def error_bar(self, x_name):

        def pivot_plot(df, x_name, ax, loss_name):
            color = {'Train': 'blue', 'Validation': 'red'}
            ax.set_box_aspect(1)
            for loss_type in ['Train', 'Validation']:
                df_pivot = pd.pivot_table(df, values=f'{loss_type} {loss_name}', index=x_name, aggfunc='mean')
                df_std = pd.pivot_table(df, values=f'{loss_type} {loss_name}', index=x_name, aggfunc='std')

                np_x = df_pivot.index.to_numpy().flatten()
                np_y = df_pivot.values.flatten()
                np_yerr = df_std.values.flatten()

                ax.errorbar(np_x, np_y, yerr=np_yerr, fmt='o-', capsize=5, label=f'{loss_type} {loss_name}', color=color[loss_type])
            
            ax.legend(
                prop={'size': 20, 'weight': 'bold'},
                frameon=False,
                loc='best'
            )
            ax.set_title(f'{loss_name}', fontsize=25, fontweight='bold')
            ax.set_xlabel(x_name, fontsize=25, fontweight='bold')
            ax.set_ylabel('Log10 Loss', fontsize=25, fontweight='bold')
            style_axes(ax)

        def plotting(df, x_name):
            subplot_size = 8
            fig, ax = plt.subplots(2, 2, figsize=(subplot_size*2.5, subplot_size*2.5+2),
                                   gridspec_kw={'width_ratios': [1, 1],
                                                'height_ratios': [1, 1]})
            fig.suptitle(f'{x_name}', fontsize=25, fontweight='bold', y=1.05)
            ax = ax.flatten()
            for i, loss_name in enumerate(['loss', 'recon. loss', 'cos loss', 'l1 loss']):
                pivot_plot(df, x_name, ax[i], loss_name)

            fig.tight_layout()
            plt.savefig(f'Result/{self.model_class.__name__}/{self.save_path}/error_bar_{x_name}.png',dpi=300)
            self.plot_writer.add_figure(f'error_bar_{x_name}', fig)
            plt.close()

        def style_axes(ax):
            ax.grid(True)
            ax.set_facecolor('white')
            ax.spines['left'].set_linewidth(3.0)
            ax.spines['bottom'].set_linewidth(3.0)
            ax.spines['right'].set_linewidth(3.0)
            ax.spines['top'].set_linewidth(3.0)
        
            ax.tick_params(axis='both', width=3.0, length=10,which='major',labelsize=20)
            plt.setp(ax.get_xticklabels(), weight='bold')
            plt.setp(ax.get_yticklabels(), weight='bold')

            ax.set_xlabel(ax.get_xlabel(), fontsize=25, fontweight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontsize=25, fontweight='bold')
            ax.set_title(ax.get_title(), fontsize=25, fontweight='bold', pad=22)
        
        file_path = f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt'
        df = pd.read_csv(file_path, sep=' ', names=['trial', x_name, 'Number of layers', 'Train loss', 'Train recon. loss', 'Train cos loss', 
                                                    'Train l1 loss', 'Validation loss', 'Validation recon. loss', 'Validation cos loss', 'Validation l1 loss'])
        df.astype({'trial': int, x_name: int, 'Number of layers': int})
        df.astype({'Train loss': float, 'Train recon. loss': float, 'Train cos loss': float, 'Train l1 loss': float, 
                    'Validation loss': float, 'Validation recon. loss': float, 'Validation cos loss': float, 'Validation l1 loss': float})
        
        for loss_type in ['Train', 'Validation']:
            for loss_name in ['loss', 'recon. loss', 'cos loss', 'l1 loss']:
                df[f'{loss_type} {loss_name}'] = np.log10(df[f'{loss_type} {loss_name}'])
        
        self.sub_pbar = tqdm(total=4, desc="Plotting error bar", leave=False)
        for x in [x_name, 'Number of layers']:
            plotting(df, x)
            self.sub_pbar.update(1)
        self.sub_pbar.close()

    def report(self):
        file_list = [
            f'Result/{self.model_class.__name__}/{self.save_path}/heatmap.png'
        ]
        report_to_mail(
            f"Final Results - {self.date}",
            'kth102938@g.skku.edu',
            'ronaldo1225!',
            'code.jaguar1225@gmail.com',
            file_list = file_list
        )

class ModelTrainer(Plotter):
    def __init__(self, model_class, hidden_dimension, 
                 sequence_length, layer_dimension, num_trial, epoch, use_writer, date, device='cuda'):
        self.model_class = model_class
        self.hidden_dimension = hidden_dimension
        self.sequence_length = sequence_length
        self.layer_dimension = layer_dimension+1
        self.date = date
        self.device = device
        self.model_type = self._get_model_type()
        self.epoch = epoch
        self.num_trial = num_trial
        self.use_writer = use_writer
        
    def _get_model_type(self):
        name = self.model_class.__name__
        if "RNN" in name:
            return "RNN"
        elif "Plas" in name:
            return "PlasDyn"
        else:
            return "AE"
            
    def train(self):
        if os.path.exists(f'temp'):
            import shutil
            shutil.rmtree(f'temp')
        os.makedirs(f'temp', exist_ok=True)
        self.plot_writer = SummaryWriter('temp')

        model_params = {
            'batch_size': 2**14,
            'date': self.date,
            'device': self.device,
            'process_variables': 16,
            'hidden_dimension': self.hidden_dimension
        }
        if self.model_type in ["RNN", "PlasDyn"]:
            combinations = [(self.sequence_length,l) for l in range(self.layer_dimension)]
        else:
            combinations = [(self.hidden_dimension,l) for l in range(self.layer_dimension)]
       
        self.save_path = self.date
        check_list = []

        pbar = tqdm(total=self.num_trial*len(combinations), desc="Training", leave=True)
        train_data = None

        for trial in range(self.num_trial):
            for (h_or_t, layer_dim) in combinations:
                try:
                    with open(f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt', 'r') as f:
                        if f'{trial} {h_or_t} {layer_dim}' in f.read():
                            check_list.append(f'{trial} {h_or_t} {layer_dim}')
                            pbar.update(1)
                            pbar.set_postfix({'already_trained': f'{trial} {h_or_t} {layer_dim}'})
                            continue
                except FileNotFoundError as e:
                    print("Start training")

                if self.model_type in ["RNN", "PlasDyn"]:
                    train_data = DataLoader(T=h_or_t)
                    train_data.nor()
                    train_data.to(self.device)
                    layer_dimension = [16 for _ in range(layer_dim)] + [self.hidden_dimension]
                else:
                    if train_data is None:
                        train_data = DataLoader(T=0)
                        train_data.nor()
                        train_data.to(self.device)
                    layer_dimension = [3648 for _ in range(layer_dim)] + [self.hidden_dimension]
                # 모델 초기화
                model_params['keyword'] = f'{self.model_class.__name__} {trial} {h_or_t} {layer_dim}'
                model_params['layer_dimension'] = layer_dimension
                model = self.model_class(**model_params)
                self.save_path = model.params['save_path']
                model.to(self.device)
                train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss = \
                model.update(train_data,epoch=self.epoch,use_writer=self.use_writer)

                os.makedirs(f'Result/{self.model_class.__name__}/{self.save_path}/temp', exist_ok=True)
                with open(f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt', 'a') as f:
                    f.write(f'{trial} {h_or_t} {layer_dim} {train_loss} {train_recon_loss} {train_cos_loss} {train_l1_loss} {val_loss} {val_recon_loss} {val_cos_loss} {val_l1_loss}\n')
                del model
                gc.collect()

                try:
                    if self.model_type in ["RNN", "PlasDyn"]:
                        self.heatmap('Sequence length')
                        self.error_bar('Sequence length')
                        self.minplot('Sequence length')
                    else:
                        self.heatmap('Hidden dimension')
                        self.error_bar('Hidden dimension')
                        self.minplot('Hidden dimension')
                except ValueError:
                    pass
                pbar.update(1)
                pbar.set_postfix({'trained': f'{trial} {h_or_t} {layer_dim}'})
        pbar.close()
        self.plot_writer.close()
        return self.save_path
    
    def convergence_check(self):
        data_df = pd.read_csv(f'Result/{self.model_class.__name__}/{self.save_path}/temp/val_loss.txt', sep=' ', 
                              names=['trial', 'h_or_t', 'layer_dim', 'Train loss', 'Train recon. loss', 'Train cos loss', 
                                     'Train l1 loss', 'Validation loss', 'Validation recon. loss', 'Validation cos loss', 'Validation l1 loss'])
        
        for loss_type in ['Train', 'Validation']:
            for loss_name in ['loss', 'recon. loss', 'cos loss', 'l1 loss']:
                data_df[f'{loss_type} {loss_name}'] = np.log10(data_df[f'{loss_type} {loss_name}'])

        val_losses_vs_layer_dim_means_df = data_df.pivot_table(index='layer_dim', values='Validation loss', aggfunc='mean')
        val_losses_vs_layer_dim_stds_df = data_df.pivot_table(index='layer_dim', values='Validation loss', aggfunc='std')

        losses_vs_layer_dim_means_df = data_df.pivot_table(index='layer_dim', values='Train loss', aggfunc='mean')
        losses_vs_layer_dim_stds_df = data_df.pivot_table(index='layer_dim', values='Train loss', aggfunc='std')
        
        val_losses_vs_h_or_t_means_df = data_df.pivot_table(index='h_or_t', values='Validation loss', aggfunc='mean')
        val_losses_vs_h_or_t_stds_df = data_df.pivot_table(index='h_or_t', values='Validation loss', aggfunc='std')

        losses_vs_h_or_t_means_df = data_df.pivot_table(index='h_or_t', values='Train loss', aggfunc='mean')
        losses_vs_h_or_t_stds_df = data_df.pivot_table(index='h_or_t', values='Train loss', aggfunc='std')

        layer_dim_convergence_threshold = 0.05
        h_or_t_convergence_threshold = 0.1
        convergence_window = 3

        val_h_or_t_convergence = []
        val_layer_dim_convergence = []

        for i in range(convergence_window, len(val_losses_vs_layer_dim_means_df)):
            layer_dim_std = val_losses_vs_layer_dim_stds_df.iloc[i-convergence_window:i].std()
            if layer_dim_std.min() < layer_dim_convergence_threshold:
                val_layer_dim_convergence.append((2*i-convergence_window)//2)
                
        for i in range(convergence_window, len(val_losses_vs_h_or_t_means_df)):
            h_or_t_std = val_losses_vs_h_or_t_stds_df.iloc[i-convergence_window:i].std()
            if h_or_t_std.min() < h_or_t_convergence_threshold:
                if self.model_type in ["RNN", "PlasDyn"]:
                    val_h_or_t_convergence.append((2*i-convergence_window)//2)
                else:
                    val_h_or_t_convergence.append(2**((2*i-convergence_window)//2+1))

        optimal_layer_dim = max(val_layer_dim_convergence)
        optimal_h_or_t = max(val_h_or_t_convergence)


        
        fig, ax = plt.subplots(2, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [1, 1]})

        data_np = data_df[data_df['h_or_t'] == optimal_h_or_t]

        val_loss_layer_dim_means_df = data_np.pivot_table(index='layer_dim', values='Validation loss', aggfunc='mean')
        val_loss_layer_dim_stds_df = data_np.pivot_table(index='layer_dim', values='Validation loss', aggfunc='std')
        train_loss_layer_dim_means_df = data_np.pivot_table(index='layer_dim', values='Train loss', aggfunc='mean')
        train_loss_layer_dim_stds_df = data_np.pivot_table(index='layer_dim', values='Train loss', aggfunc='std')
        
        val_np_x = val_loss_layer_dim_means_df.index.to_numpy().flatten()
        val_np_y = val_loss_layer_dim_means_df.values.flatten()
        val_np_yerr = val_loss_layer_dim_stds_df.values.flatten()
        train_np_x = train_loss_layer_dim_means_df.index.to_numpy().flatten()
        train_np_y = train_loss_layer_dim_means_df.values.flatten()
        train_np_yerr = train_loss_layer_dim_stds_df.values.flatten()

        ax = ax.flatten()
        ax[0].set_box_aspect(1)
        ax[0].errorbar(val_np_x, val_np_y, 
                     yerr=val_np_yerr, 
                    fmt='-o', capsize=5, label='Validation loss', color='red', markersize=10)
        ax[0].errorbar(train_np_x, train_np_y, 
                     yerr=train_np_yerr, 
                    fmt='-o', capsize=5, label='Train loss', color='blue', markersize=10)
        if val_layer_dim_convergence:
            ax[0].axvline(x=max(val_layer_dim_convergence), color='red', linestyle='dashed', label='Validation loss convergence')
        ax[0].tick_params(axis='both', width=3.0, length=10,which='major',labelsize=20)
        ax[0].legend(fontsize=20, prop={'size': 20, 'weight': 'bold'})
        ax[0].grid(True, which="both", ls="-", alpha=0.2)
        ax[0].set_xlabel("Layer Depth", fontsize=20, fontweight='bold')
        ax[0].set_ylabel("Loss", fontsize=20, fontweight='bold')
        for spine in ax[0].spines.values():
            spine.set_linewidth(3.0)

        data_np = data_df[data_df['layer_dim'] == optimal_layer_dim]

        val_loss_h_or_t_means_df = data_np.pivot_table(index='h_or_t', values='Validation loss', aggfunc='mean')
        val_loss_h_or_t_stds_df = data_np.pivot_table(index='h_or_t', values='Validation loss', aggfunc='std')
        train_loss_h_or_t_means_df = data_np.pivot_table(index='h_or_t', values='Train loss', aggfunc='mean')
        train_loss_h_or_t_stds_df = data_np.pivot_table(index='h_or_t', values='Train loss', aggfunc='std')
        
        val_np_x = val_loss_h_or_t_means_df.index.to_numpy().flatten()
        val_np_y = val_loss_h_or_t_means_df.values.flatten()
        val_np_yerr = val_loss_h_or_t_stds_df.values.flatten()
        train_np_x = train_loss_h_or_t_means_df.index.to_numpy().flatten()
        train_np_y = train_loss_h_or_t_means_df.values.flatten()
        train_np_yerr = train_loss_h_or_t_stds_df.values.flatten()

        ax[1].set_box_aspect(1)
        ax[1].errorbar(val_np_x, val_np_y, 
                     yerr=val_np_yerr, 
                    fmt='-o', capsize=5, label='Validation loss', color='red', markersize=10)
        ax[1].errorbar(train_np_x, train_np_y, 
                     yerr=train_np_yerr, 
                    fmt='-o', capsize=5, label='Train loss', color='blue', markersize=10)
        if val_h_or_t_convergence:
            ax[1].axvline(x=max(val_h_or_t_convergence), color='red', linestyle='dashed', label='Validation loss convergence')
        ax[1].tick_params(axis='both', width=3.0, length=10,which='major',labelsize=20)
        ax[1].legend(fontsize=20, prop={'size': 20, 'weight': 'bold'})
        ax[1].grid(True, which="both", ls="-", alpha=0.2)
        ax[1].set_xlabel("Hidden Dimension", fontsize=20, fontweight='bold')
        ax[1].set_ylabel("Loss", fontsize=20, fontweight='bold')
        for spine in ax[1].spines.values():
            spine.set_linewidth(3.0)

        fig.tight_layout()
        fig.savefig(f'Result/{self.model_class.__name__}/{self.save_path}/convergence_check.png',dpi=300)
        plt.close()
        
        return optimal_layer_dim, optimal_h_or_t


    def _save_model(self, model, save_path, keyword):
        # 공백과 특수문자를 언더스코어로 변경
        safe_keyword = keyword.replace(' ', '_').replace('/', '_')
        model_path = f'model/{model.__class__.__name__}/{save_path}/{safe_keyword}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)