import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union, Optional
import os
from datetime import datetime

class PlotStyle:
    """OriginLab 스타일의 플로팅을 위한 스타일 설정"""
    
    @staticmethod
    def set_style():
        """기본 스타일 설정"""
        # 기본 스타일 설정
        plt.style.use('default')
        
        # 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 25
        plt.rcParams['axes.titlesize'] = 30
        plt.rcParams['axes.labelsize'] = 25
        
        # 그리드 설정
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5
        
        # 축 설정
        plt.rcParams['axes.linewidth'] = 3.0
        plt.rcParams['axes.edgecolor'] = 'black'
        
        # 틱 설정
        plt.rcParams['xtick.major.width'] = 3.0
        plt.rcParams['ytick.major.width'] = 3.0
        plt.rcParams['xtick.minor.width'] = 3.0
        plt.rcParams['ytick.minor.width'] = 3.0
        
        # 여백 설정
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['figure.constrained_layout.use'] = True

class Plotter:
    """플로팅 유틸리티 클래스"""
    
    def __init__(self, save_dir: str = datetime.now().strftime('%Y%m%d')):
        """
        Args:
            save_dir (str): 그래프 저장 디렉토리
        """
        self.save_dir = f'plots/{save_dir}'
        os.makedirs(self.save_dir, exist_ok=True)
            
    def plot_heatmap(self,
                    data: Union[List, np.ndarray],
                    title: str = '',
                    xlabel: str = '',
                    ylabel: str = '',
                    save_name: Optional[str] = None,
                    dpi: int = 300) -> None:
        """
        히트맵 플로팅
        
        Args:
            data (Union[List, np.ndarray]): 플로팅할 데이터
            title (str): 그래프 제목
            xlabel (str): x축 레이블
            ylabel (str): y축 레이블
            save_name (Optional[str]): 저장 파일명
            dpi (int): 저장 해상도
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(data, aspect='auto', cmap='RdBu_r')
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_line(self, 
                 x: Union[List, np.ndarray],
                 y: Union[List, np.ndarray],
                 title: str = '',
                 xlabel: str = '',
                 ylabel: str = '',
                 color: str = None,
                 marker: str = None,
                 linestyle: str = None,
                 label: str = None,
                 grid: bool = True,
                 save_name: Optional[str] = None,
                 dpi: int = 300) -> None:
        """
        선 그래프 생성
        
        Args:
            x: x축 데이터
            y: y축 데이터
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 선 색상
            marker: 마커 스타일
            linestyle: 선 스타일
            label: 범례 레이블
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """
        plt.figure(figsize=(8, 6))

        if x is list:
            x = np.array(x).T
        if y is list:
            y = np.array(y).T
        
        if color is None:
            color = self.colors[0:x.shape[-1]]  # 단일 색상으로 수정
        if marker is None:
            marker = self.markers[0:x.shape[-1]]
        if linestyle is None:
            linestyle = self.linestyles[0:x.shape[-1]]
        if label is None:
            label = [f'{i}' for i in range(x.shape[-1])]
        
        for i in range(x.shape[-1]):
            plt.plot(x[i], y[i], color=color[i], marker=marker[i], linestyle=linestyle[i], 
                label=label[i], linewidth=2.5, markersize=6)
                
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        if label:
            plt.legend()
            
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def plot_scatter(self,
                    x: Union[List, np.ndarray],
                    y: Union[List, np.ndarray],
                    title: str = '',
                    xlabel: str = '',
                    ylabel: str = '',
                    color: str = None,
                    marker: str = None,
                    label: str = None,
                    grid: bool = True,
                    save_name: Optional[str] = None,
                    dpi: int = 300) -> None:
        """
        산점도 생성
        
        Args:
            x: x축 데이터
            y: y축 데이터
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 점 색상
            marker: 마커 스타일
            label: 범례 레이블
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """

        if x is list:
            x = np.array(x).T
        if y is list:
            y = np.array(y).T
        plt.figure(figsize=(8, 6))
        
        if color is None:
            color = self.colors[0:x.shape[-1]]
        if marker is None:
            marker = self.markers[0:x.shape[-1]]
        if label is None:
            label = [f'{i}' for i in range(x.shape[-1])]
            
        for i in range(x.shape[-1]):
            plt.scatter(x[i], y[i], color=color[i], marker=marker[i], label=label[i], s=50)
        
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        if label:
            plt.legend()
            
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def plot_comparison(self,
                       true_values: Union[List, np.ndarray],
                       predicted_values: Union[List, np.ndarray],
                       title: str = 'True vs Predicted Values',
                       xlabel: str = 'True Values',
                       ylabel: str = 'Predicted Values',
                       color: str = None,
                       marker: str = None,
                       grid: bool = True,
                       save_name: Optional[str] = None,
                       dpi: int = 300) -> None:
        """
        예측값과 실제값 비교 그래프 생성
        
        Args:
            true_values: 실제값
            predicted_values: 예측값
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 점 색상
            marker: 마커 스타일
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """
        plt.figure(figsize=(8, 6))
        
        if color is None:
            color = self.colors[0]
        if marker is None:
            marker = self.markers[0]
            
        plt.scatter(true_values, predicted_values, color=color, marker=marker, 
                   alpha=0.5, s=50, label='Data points')
        
        # 대각선 추가
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect prediction')
        
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        plt.legend()
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def plot_learning_curve(self,
                          train_loss: List[float],
                          val_loss: Optional[List[float]] = None,
                          title: str = 'Learning Curve',
                          xlabel: str = 'Epoch',
                          ylabel: str = 'Loss',
                          grid: bool = True,
                          save_name: Optional[str] = None,
                          dpi: int = 300) -> None:
        """
        학습 곡선 생성
        
        Args:
            train_loss: 학습 손실
            val_loss: 검증 손실
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """
        plt.figure(figsize=(8, 6))
        
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, color=self.colors[0], marker=self.markers[0],
                linestyle=self.linestyles[0], label='Training Loss', linewidth=2)
        
        if val_loss is not None:
            plt.plot(epochs, val_loss, color=self.colors[1], marker=self.markers[1],
                    linestyle=self.linestyles[1], label='Validation Loss', linewidth=2)
        
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        plt.legend()
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
