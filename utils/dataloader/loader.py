import torch.utils.data as data
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold

class Data_Set(data.Dataset):
    def __init__(self, intensity, condition):
        self.intensity = intensity
        self.condition = condition

    def __call__(self):
        return self.intensity, self.condition
    
    def __len__(self):
        return len(self.intensity)
    
    def __getitem__(self, index):
        return self.intensity[index], self.condition[index]

class Data_Loader(data.DataLoader):
    def __init__(self, data_path, **kwargs):
        intensity, condition = self.load_data(data_path)
        self.intensity = intensity
        self.condition = condition
        dataset = Data_Set(intensity, condition)
        super(Data_Loader, self).__init__(dataset=dataset, **kwargs)

    def __call__(self):
        return self.dataset.intensity, self.dataset.condition

    def load_data(self, data_path):
        files = glob(data_path + '/*.csv')
        intensity = None
        condition = None
        
        pbar = tqdm(total=len(files), desc="Loading data files", leave=False)
        # tqdm으로 파일 로딩 진행 상황 표시
        for i, file in enumerate(files):
            data = np.loadtxt(file, delimiter=',', dtype=np.float32, skiprows=1)
            
            if i == 0:
                intensity = data[:, 25:]
                condition = data[:, 1:25]
            else:
                intensity = np.concatenate((intensity, data[:, 25:]), axis=0)
                condition = np.concatenate((condition, data[:, 1:25]), axis=0)
            pbar.update(1)
            pbar.set_postfix({'progress': f'{i+1}/{len(files)}'})
        
        pbar.close()
        return intensity, condition
    
    def create_kfold_loaders(self, n_splits=5, shuffle=True, random_state=42, **kwargs):
        """
        K-fold 교차 검증을 위한 데이터 로더들을 생성합니다.
        
        매개변수:
            n_splits (int): 폴드 수 (기본값: 5)
            shuffle (bool): 폴드 분할 전 데이터 셔플 여부 (기본값: True)
            random_state (int): 재현성을 위한 랜덤 시드 (기본값: 42)
            **kwargs: DataLoader에 전달할 추가 매개변수 (batch_size, num_workers 등)
            
        반환값:
            list: (train_loader, val_loader) 튜플의 리스트
        """
        # KFold 객체 생성
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # 인덱스 배열 생성
        indices = np.arange(len(self.intensity))
        
        # 각 폴드에 대한 데이터 로더 생성
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            # 훈련 데이터셋과 검증 데이터셋 생성
            train_dataset = Data_Set(
                self.intensity[train_idx],
                self.condition[train_idx]
            )
            
            val_dataset = Data_Set(
                self.intensity[val_idx],
                self.condition[val_idx]
            )
            
            # DataLoader 생성 시 kwargs 적용
            train_loader = data.DataLoader(train_dataset, **kwargs)
            val_loader = data.DataLoader(val_dataset, **kwargs)
            
            fold_loaders.append((train_loader, val_loader))
            
        return fold_loaders
    
    def get_single_fold(self, fold_idx=0, train_ratio=0.8, shuffle=True, random_state=42, **kwargs):
        """
        단일 훈련/검증 분할을 위한 데이터 로더를 생성합니다.
        
        매개변수:
            fold_idx (int): 폴드 인덱스 (기본값: 0)
            train_ratio (float): 훈련 데이터 비율 (기본값: 0.8)
            shuffle (bool): 분할 전 데이터 셔플 여부 (기본값: True)
            random_state (int): 재현성을 위한 랜덤 시드 (기본값: 42)
            **kwargs: DataLoader에 전달할 추가 매개변수
            
        반환값:
            tuple: (train_loader, val_loader) 튜플
        """
        # 인덱스 배열 생성
        indices = np.arange(len(self.intensity))
        
        # 셔플 적용
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        
        # 훈련/검증 분할 지점 계산
        split_idx = int(len(indices) * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        # 훈련 데이터셋과 검증 데이터셋 생성
        train_dataset = Data_Set(
            self.intensity[train_idx],
            self.condition[train_idx]
        )
        
        val_dataset = Data_Set(
            self.intensity[val_idx],
            self.condition[val_idx]
        )
        
        # DataLoader 생성
        train_loader = data.DataLoader(train_dataset, **kwargs)
        val_loader = data.DataLoader(val_dataset, **kwargs)
        
        return train_loader, val_loader

