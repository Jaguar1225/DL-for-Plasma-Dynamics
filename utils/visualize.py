# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:24:50 2024

@author: user
"""
from tqdm import tqdm
import time
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import re
from torch.utils.data import DataLoader

from Data_loader import TestDataLoader

import matplotlib.pyplot as plt

def visualize_encoding(model_path,nomality):
    # 모델 구조에 맞게 수정해주세요
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.cpu()

    # 데이터 로드
    aetraindata = TestDataLoader(test_size=0,T=0,path=f'Data/{nomality}')
    aetraindata.nor()  # RNNDataLoader를 사용한다고 가정합니다

    # 인코딩
    with torch.no_grad():
        encoded = model.encode(aetraindata())  # 모델의 encode 메서드를 사용한다고 가정합니다

    # 결과 출력
    print("인코딩된 데이터 값:")
    print(encoded.squeeze().numpy())


def visualize_ae_rnn_results(model_path, nomality,device='cuda'):
    file_names = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    total = len(file_names)
    pbar = tqdm(file_names, desc='Processing files', leave=False, total=total)
    for f in pbar:
        l, t = re.search(r'_(\d+)_(\d+).pth',f).groups()
        l = int(l)
        t = int(t)

        testdata = TestDataLoader(path=f'Data/{nomality}', T=t, mode='time')

        sample_model = torch.load(os.path.join(model_path, f), weights_only=False)

        sample_model.to(device)
        for idx, test in testdata.items():
            test.to(device)
            test.nor()

        #plot_with_decay(sample_model)
        #try:
        #    analyze_model_influence(sample_model, testdata['Power10'])
        #except:
        #    analyze_model_influence(sample_model, testdata['1'])

        fig_size = 6
        fig_cos, axes_cos = plt.subplots(len(testdata)//4+1, 4, figsize=((fig_size+1)*4, (fig_size+2)*len(testdata)//4+3))
        axes_cos = axes_cos.flatten()
        fig_cos.suptitle(f"FD(cos) score plotting")

        fig_fd, axes_fd = plt.subplots(len(testdata)//4+1, 4, figsize=((fig_size+1)*4, (fig_size+2)*len(testdata)//4+3))
        axes_fd = axes_fd.flatten()
        fig_fd.suptitle(f"FD(mse) score plotting")

        l = 0

        for n, (idx,test) in enumerate(testdata.items()):
            print(f'{idx} file proceeding')
            testdata[idx].to(device)
            pbar.set_postfix(file=f'{idx}')
            # 인코딩
            be_time = time.time()
            Z_real = sample_model.encode(testdata[idx][:,-1][0]).cpu().detach()
            N = len(Z_real)
            ae_time = time.time()
            print(f'Time elapsed for encoding {N} length: {round(ae_time - be_time,2)}s ({round((ae_time - be_time)/N*1e3,2)}ms for each)')
            
            # 예측
            be_time = time.time()
            X = testdata[idx]()[0]
            V = testdata[idx]()[1].requires_grad_()
            Z_pred = sample_model(X[:,0:1], V[:,1:]).cpu()
            ae_time = time.time()
            print(f'Time elapsed for prediction of {N} length: {round(ae_time - be_time,2)}s ({round((ae_time - be_time)/N*1e3,2)}ms for each)')

            # FD 점수 계산
            criterion = nn.MSELoss(reduction='none')
            FD_score = torch.mean(criterion(Z_pred, Z_real),dim=1)
            FD_score.backward(torch.ones_like(FD_score),retain_graph=True)
            V_mse_grad = V.grad.cpu().detach().clone()

            sample_model.zero_grad()
            V.grad.zero_()

            criterion = nn.CosineEmbeddingLoss(reduction='none')
            cos_score = criterion(Z_pred, Z_real,torch.tensor([1]))
            cos_score.backward(torch.ones_like(cos_score))
            V_cos_grad = V.grad.cpu().detach().clone()

            FD_score_np = np.c_[np.arange(t*0.5, (N+t)*0.5, step=0.5),
                                Z_pred.detach().numpy(),
                                Z_real.detach().numpy(),
                                FD_score.detach().numpy(),
                                cos_score.detach().numpy()]
            
            V_np = np.c_[np.arange(t*0.5, (N+t)*0.5, step=0.5).reshape(-1,1),
                                V_mse_grad[:,-1,:].detach().numpy(),
                                V_cos_grad[:,-1,:].detach().numpy()]
            
            os.makedirs(f'Result/{sample_model.__class__.__name__}/{sample_model.date}/FD score/{nomality}',exist_ok =True)
            np.savetxt(f'Result/{sample_model.__class__.__name__}/{sample_model.date}/FD score/{nomality}/Pred FD score {idx} Sequence {t} hidden {32}.csv',FD_score_np,delimiter=',')
            os.makedirs(f'Result/{sample_model.__class__.__name__}/{sample_model.date}/V grad/{nomality}',exist_ok =True)
            np.savetxt(f'Result/{sample_model.__class__.__name__}/{sample_model.date}/V grad/{nomality}/V grad {idx} Sequence {t} hidden {32}.csv',V_np,delimiter=',')

            # 특징 플롯
            fig, axes = plt.subplots(32//4, 4, figsize=((fig_size+1)*4, (fig_size+2)*32//4+3))
            fig.suptitle(f"Sequence {t} hidden {32} file {idx}")

            for m, ax in tqdm(enumerate(axes.flatten())):
                ax.set_title(f'{m+1}/{32}')
                ax.plot(np.arange(t*0.5, (N+t)*0.5, step=0.5), Z_real[:,m], color='black', label='Real')
                ax.plot(np.arange(t*0.5, (N+t)*0.5, step=0.5), Z_pred[:,m].detach(), color='red', label='Predicted')
                ax.grid()
                ax.legend()

            plt.tight_layout()
            
            os.makedirs(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/Pred hidden/{nomality}", exist_ok=True)
            fig.savefig(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/Pred hidden/{nomality}/Pred feature Sequence {t} hidden {32} file {idx}.png", dpi=150)
            plt.close(fig)

            # V grad 플롯
            fig, axes = plt.subplots(4, 4, figsize=((fig_size+1)*4, (fig_size+2)*32//4+3))
            fig.suptitle(f"Sequence {t} V mse grad {32} file {idx}")

            for m, ax in tqdm(enumerate(axes.flatten())):
                ax.set_title(f'{m+1}/{32}')
                ax.plot(np.arange(t, (N+t), step=1), V_mse_grad[:,-1,m].detach(), color='red')
                ax.grid()

            plt.tight_layout()
            
            os.makedirs(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/V mse grad/{nomality}", exist_ok=True)
            fig.savefig(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/V mse grad/{nomality}/V mse grad Sequence {t} hidden {32} file {idx}.png", dpi=150)
            plt.close(fig)

            # V grad 플롯
            fig, axes = plt.subplots(4, 4, figsize=((fig_size+1)*4, (fig_size+2)*4+3))
            fig.suptitle(f"Sequence {t} V cos grad {32} file {idx}")

            for m, ax in tqdm(enumerate(axes.flatten())):
                ax.set_title(f'{m+1}/{32}')
                ax.plot(np.arange(t, (N+t), step=1), V_cos_grad[:,-1,m].detach(), color='red')
                ax.grid()

            plt.tight_layout()
            
            os.makedirs(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/V cos grad/{nomality}", exist_ok=True)
            fig.savefig(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/V cos grad/{nomality}/V cos grad Sequence {t} hidden {32} file {idx}.png", dpi=150)
            plt.close(fig)

            # FD 점수 플롯
            axes_fd[l].set_title(f'{l+1}/{len(testdata)}')
            axes_fd[l].plot(np.arange(t, (N+t), step=1), FD_score.detach(), color='red')
            axes_fd[l].grid()

                        # FD 점수 플롯
            axes_cos[l].set_title(f'{l+1}/{len(testdata)}')
            axes_cos[l].plot(np.arange(t, (N+t), step=1), cos_score.detach(), color='red')
            axes_cos[l].grid()
            
            l += 1

        os.makedirs(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/FD score/{nomality}", exist_ok=True)
        fig_fd.savefig(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/FD score/{nomality}/Pred FD score Sequence {t} hidden {32}.png", dpi=300)
        plt.close(fig_fd)

        os.makedirs(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/cos score/{nomality}", exist_ok=True)
        fig_cos.savefig(f"Fig/{sample_model.__class__.__name__}/{sample_model.date}/cos score/{nomality}/Pred cos score Sequence {t} hidden {32}.png", dpi=300)
        plt.close(fig_cos)
        pbar.update(1)

        del sample_model, testdata

def plot_with_decay(sample_model):
    # decay 값 계산
    rho = sample_model.RNNDict['TDM'].rho  # [hidden_dim]
    bounded_rho = torch.tanh(rho)  # rho 값을 -1~1 사이로 제한
    decay = torch.exp(-bounded_rho * 0.5)  # [hidden_dim]
    decay_value = decay.cpu().detach()
    
    # imshow로 각 차원의 decay 값 시각화
    plt.figure(figsize=(10, 2))
    im = plt.imshow(decay_value.unsqueeze(0), aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Decay Value')
    
    # 평균 decay 값 텍스트 추가
    mean_decay = decay_value.mean().item()
    if mean_decay < 0.01:
        text = f"Mean decay: {mean_decay:.2e}"
    else:
        text = f"Mean decay: {mean_decay:.2f}"
    
    plt.text(0.02, 0.98, text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('TDM Decay Values across Hidden Dimensions')
    plt.xlabel('Hidden Dimension')
    
    # 저장 경로 생성
    save_dir = os.path.join('Fig', sample_model.__class__.__name__, sample_model.date)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'TDM_decay_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_model_influence(model, data_loader):
    """세 가지 분석 방법을 모두 수행하고 시각화"""
    
    def weight_analysis(model):
        # State transition weights
        if "RNN" in model.__name__:
            weight_hh = model.RNNDict['RNN'].rnn_layers[0].weight_hh.detach().cpu()

        elif "PlasDyn" in model.__name__:
            weight_hh = model.PlasDynDict['PlasDyn'].weight.detach().cpu()
        # Control variable weights
        linear_weight = model.RNNDict['Linear'][0].weight.detach().cpu()
        rnn_ih_weight = model.RNNDict['RNN'].rnn_layers[0].weight_hh.detach().cpu()
        control_influence = torch.mm(rnn_ih_weight, linear_weight)
        
        return weight_hh, control_influence
    
    def sensitivity_analysis(model, batch_data):
        X, V, T = batch_data
        X.requires_grad_(True)
        V.requires_grad_(True)
        
        with torch.enable_grad():
            output = model(X[:,:-1], V[:,1:])
            # Z에 대한 sensitivity
            X_sensitivity = []
            for i in range(output.size(-1)):
                model.zero_grad()
                output[..., i].sum().backward(retain_graph=True)
                X_sensitivity.append(X.grad.clone().cpu())
                X.grad.zero_()
            
            # L에 대한 sensitivity
            V_sensitivity = []
            for i in range(output.size(-1)):
                model.zero_grad()
                output[..., i].sum().backward(retain_graph=True)
                V_sensitivity.append(V.grad.clone().cpu())
                V.grad.zero_()
                
        return torch.stack(X_sensitivity).mean(0), torch.stack(V_sensitivity).mean(0)
    
    def lrp_analysis(model, batch_data):
        X, V, T = batch_data
        eps = 1e-9
        
        # Forward pass
        with torch.no_grad():
            Z = model.encode(X[:,0])
            V_encoded = {}
            temp_V = V[:,1:].reshape(-1,V.size(-1))

            for e in range(len(model.params['layer_dimension'])):
                N, T, D = temp_V.size()
                V_encoded[f'E{e}'] = model.RNNDict[f'E{e}'](temp_V).reshape(N,T,-1)
                temp_V = V_encoded[f'E{e}']

            h_rnn = model(X[:,:-1], V[:,1:])[:,-1]
            W_hh = model.RNNDict['RNN'].rnn_layers[0].weight_hh
            W_ih = model.RNNDict['RNN'].rnn_layers[0].weight_ih
            W_ii = {}

            for e in range(len(model.params['layer_dimension'])):
                W_ii[f'E{e}'] = model.RNNDict[f'E{e}'][0].weight


        # Backward relevance propagation
        R = h_rnn  # 시작 relevance
        
        # RNN layer relevance
        Z_relevance = torch.matmul(R, (Z.reshape(-1,1,Z.size(-1)) * W_hh))
        Z_relevance = Z_relevance / (torch.matmul(Z.reshape(-1,1,Z.size(-1)), W_hh) + eps)  
        Z_relevance = Z_relevance.reshape(-1,model.params['hidden_dimension'])

        # Linear layer relevance
        temp_V = V_encoded[f'E{len(model.params["layer_dimension"])-1}']
        temp_V = temp_V.reshape(-1,1,temp_V.size(-1))
        over = torch.matmul(R, temp_V*W_ih)
        under = (torch.matmul(temp_V, W_ih) + eps)
        V_relevance_ii = over / under

        for e in range(len(model.params['layer_dimension'])-2,0,-1):
            temp_V = V_encoded[f'E{e}']
            temp_V = temp_V.reshape(-1,1,temp_V.size(-1))
            over = torch.matmul(V_relevance_ii, (temp_V*W_ii[f'E{e}'])) 
            under = (torch.matmul(temp_V, W_ii[f'E{e}']) + eps)
            V_relevance_ii = over / under

        V_relevance = V_relevance_ii.reshape(-1,model.params['process_variable'])

        return Z_relevance.cpu(), V_relevance.cpu()
    
    # 분석 실행
    model.eval()
    batch_data = data_loader()
    
    # 1. Weight Analysis
    w_state, w_control = weight_analysis(model)
    
    # 2. Sensitivity Analysis
    s_state, s_control = sensitivity_analysis(model, batch_data)
    s_state, s_control = s_state[:,0].detach(), s_control[:,1].detach()
    # 3. LRP Analysis
    l_state, l_control = lrp_analysis(model, batch_data)
    l_state, l_control = l_state.detach(), l_control.detach()

    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    methods = ['Weight Analysis', 'Sensitivity Analysis', 'LRP']
    state_maps = [w_state, s_state, l_state]
    control_maps = [w_control, s_control, l_control]
    
    for i, (method, s_map, c_map) in enumerate(zip(methods, state_maps, control_maps)):
        # State influence
        im = axes[i,0].imshow(s_map, aspect='auto', cmap='RdBu_r')
        axes[i,0].set_title(f'{method}: State Transition')
        axes[i,0].set_xlabel('Z_t')
        axes[i,0].set_ylabel('Z_{t+1}')
        plt.colorbar(im, ax=axes[i,0])
        
        # Control influence
        im = axes[i,1].imshow(c_map, aspect='auto', cmap='RdBu_r')
        axes[i,1].set_title(f'{method}: Control Influence')
        axes[i,1].set_xlabel('Control Variables')
        axes[i,1].set_ylabel('Z_{t+1}')
        plt.colorbar(im, ax=axes[i,1])
    
    plt.tight_layout()
    save_dir = os.path.join('Fig', model.__class__.__name__, model.date)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'influence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

date = '2024-11-27'
model_name = 'AE_RNN'
model_path = f'model/{model_name}/{date}'  # 모델이 저장된 경로

for nomality in ['Test']:
    visualize_ae_rnn_results(model_path,nomality)
