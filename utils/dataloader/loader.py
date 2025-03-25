import torch.utils.data as data
import numpy as np
from glob import glob
from tqdm import tqdm

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

