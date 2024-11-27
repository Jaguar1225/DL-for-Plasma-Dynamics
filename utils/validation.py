# utils/validation.py

from utils.Data_loader import DataLoader
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Autoencoder 모델
def load_autoencoder_model(model_path):
    model = torch.load(model_path,weights_only=False)
    model.cpu()
    model.eval()
    return model

def scientific_notation(x, pos=None):
    if x == 0:
        return "0"
    exp = int(np.log10(abs(x)))
    coeff = x / 10**exp
    return r"${:.1f}\times10^{{{}}}$".format(coeff, exp)

formatter = plt.FuncFormatter(scientific_notation)

X, label = DataLoader(T=1,test_size=0,label=True)
label = label[:,-1]
# 가정: AE_RNN 클래스의 인스턴스 ae_model이 이미 학습되었고, encode 메서드가 정의되어 있음
# 가정: 입력 데이터 X가 준비되어 있음 (예: X.shape = [num_samples, num_features])
model_name = 'Autoencoder'
date = '2024-11-23'
hidden_dimension = 32
num_layers = 2
model_path = f'model/{model_name}/{date}/{model_name}_{hidden_dimension}_{num_layers}.pth'
ae_model = load_autoencoder_model(model_path)

# 2. 3D Scatter Plot 생성
fig = plt.figure(figsize=(16, 18),dpi=300)
axes = []
for i in range(6):
    axes.append(fig.add_subplot(2,3,i+1,projection='3d'))

from tqdm import tqdm
pbar = tqdm(total=6,desc='Encoding',leave=False)
for i, method in enumerate(['Raw signal','Normalized','PCA','ICA','tSNE','Autoencoder']):
    pbar.set_postfix({"method":method})
    ax = axes[i]
    # Z의 차원이 3 이상일 경우, PCA 또는 t-SNE를 사용하여 3D로 축소할 수 있음
    if method == 'Raw signal':
        Z_reduced = X()[:,-1].numpy()
        Z_reduced = Z_reduced[:,[1000,2000,3000]]

    elif method == 'Normalized':
        X.nor()

        Z_reduced = X()[:,-1].numpy()
        Z_reduced = Z_reduced[:,[1000,2000,3000]]
        X.org()
    
    elif method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        Z_reduced = pca.fit_transform(X()[:,-1].numpy())  # PCA로 차원 축소

    elif method == 'ICA':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=3,
                    max_iter=1000)
        Z_reduced = ica.fit_transform(X()[:,-1].numpy())  # t-SNE로 차원 축소
    
    elif method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=3,
                    perplexity=30,
                    max_iter=1000)
        Z_reduced = tsne.fit_transform(X()[:,-1].numpy())  # t-SNE로 차원 축소

    elif method == model_name:
        X.nor()
        with torch.no_grad():
            Z_reduced = ae_model.encode(X()[:,-1])
        Z_reduced = Z_reduced[:,[1,2,3]].numpy()
        X.org()

    ax.scatter(Z_reduced[:, 0], Z_reduced[:, 1], Z_reduced[:, 2], 
              c=label[:,0],cmap='viridis',
              alpha=0.7)  # 점 크기 추가\
    
    
    
    ax.set_box_aspect([1,1,1])
    ax.set_title(method, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.set_xlabel('Feature 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=16, fontweight='bold')
    ax.set_zlabel('Feature 3', fontsize=16, fontweight='bold')
    ax.grid(True)
    pbar.update(1)

pbar.close()
plt.savefig(f'{model_name} {date} {hidden_dimension} 3D encoded features.png')
plt.close()
X.org()

pbar = tqdm(total=6,desc='Encoding',leave=False)
fig, axes = plt.subplots(2,3,figsize=(16,24),dpi=300)
for i, method in enumerate(['Raw signal','Normalized','PCA','ICA','tSNE','model_name']):
    pbar.set_postfix({"method":method})
    if method == 'Raw signal':
        Z_reduced = X()[:,-1].numpy()
        Z_reduced = Z_reduced[:,[2000,3000]]
    elif method == 'Normalized':
        X.nor()
        Z_reduced = X()[:,-1].numpy()
        Z_reduced = Z_reduced[:,[2000,3000]]
        X.org()
    elif method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Z_reduced = pca.fit_transform(X()[:,-1].numpy())
    elif method == 'ICA':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=2,
                    max_iter=1000)
        Z_reduced = ica.fit_transform(X()[:,-1].numpy())
    elif method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2,
                    perplexity=30,
                    max_iter=1000)
        Z_reduced = tsne.fit_transform(X()[:,-1].numpy())
    elif method == model_name:
        X.nor()
        with torch.no_grad():
            Z_reduced = ae_model.encode(X()[:,-1])
        Z_reduced = Z_reduced[:,[1,2]].numpy()
        X.org()
    ax = axes[i//3,i%3]
    ax.scatter(Z_reduced[:, 0], Z_reduced[:, 1], c=label[:,0],cmap='viridis', alpha=0.7)
    ax.set_title(method, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True)
    ax.set_xlabel('Feature 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=16, fontweight='bold')
    pbar.update(1)
pbar.close()
plt.savefig(f'{model_name} {date} {hidden_dimension} 2D encoded features.png')
plt.close()


