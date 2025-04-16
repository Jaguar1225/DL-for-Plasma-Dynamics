import torch.utils.data as data
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from .preprocess import Normalize

class Data_Set(data.Dataset):
        self.intensity = torch.from_numpy(intensity)
        self.condition = torch.from_numpy(condition)

    def __call__(self):
        return self.intensity, self.condition
    
    def __len__(self):
        return len(self.intensity)
    
        return self.intensity[index], self.condition[index]
    
        self.intensity = self.intensity.to(device)
        self.condition = self.condition.to(device)

class Data_Loader(data.DataLoader):
        intensity, condition = self.load_data(data_path)
        intensity = Normalize().partial(intensity)
        dataset = Data_Set(intensity, condition)
        super(Data_Loader, self).__init__(dataset=dataset, **kwargs)

    def __call__(self):
        return self.dataset.intensity, self.dataset.condition
    
    def to(self, device: torch.device)->None:
        self.dataset.intensity = self.dataset.intensity.to(device)
        self.dataset.condition = self.dataset.condition.to(device)

    def load_data(self, data_path: str)->tuple[torch.Tensor, torch.Tensor]:
        files = glob(data_path + '/*.csv')
        intensity = None
        condition = None
        
        pbar = tqdm(total=len(files), desc="Loading data files", leave=False)
        # tqdm으로 파일 로딩 진행 상황 표시
        for i, file in enumerate(files):
            data = np.loadtxt(file, delimiter=',', dtype=object, skiprows=1)
            data = data[:, 1:].astype(np.float32)
            
            if i == 0:
                intensity = data[:, 25:]
                condition = data[:, :25]
            else:
                intensity = np.concatenate((intensity, data[:, 25:]), axis=0)
                condition = np.concatenate((condition, data[:, :25]), axis=0)
            pbar.update(1)
            pbar.set_postfix({'progress': f'{i+1}/{len(files)}'})
        
        pbar.close()
        return intensity, condition

