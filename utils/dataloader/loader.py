import torch.utils.data as data
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from .preprocess import Normalize

import sys
from os import path

class Data_Set(data.Dataset):
    def __init__(self, intensity: np.ndarray, condition: np.ndarray):
        self.intensity = torch.from_numpy(intensity)
        self.condition = torch.from_numpy(condition)

    def __call__(self):
        return self.intensity, self.condition
    
    def __len__(self):
        return len(self.intensity)
    
    def __getitem__(self, index: int)->tuple[torch.Tensor, torch.Tensor]:
        return self.intensity[index], self.condition[index]
    
    def to(self, device: torch.device)->None:
        self.intensity = self.intensity.to(device)
        self.condition = self.condition.to(device)

class Data_Loader(data.DataLoader):
    def __init__(self, data_path: str, preprocess: str, **kwargs):
        intensity, condition = self.load_data(data_path)
        intensity = Normalize(preprocess=preprocess)(intensity)
        self.window_size = kwargs.get('window_size', None)
        self.noise_map = self.check_noise_map()

        dataset = Data_Set(intensity, condition)
        
        super(Data_Loader, self).__init__(dataset=dataset, **kwargs)

    def __call__(self):
        return self.dataset.intensity, self.dataset.condition
    
    def to(self, device: torch.device)->None:
        self.dataset.intensity = self.dataset.intensity.to(device)
        self.dataset.condition = self.dataset.condition.to(device)

    def load_data(self, data_path: str)->tuple[np.ndarray, np.ndarray]:
        if sys.platform == 'win32':
            files = glob(path.join(data_path, '*.csv'))
        else: #linux or ubuntu
            files = glob(path.join(data_path, '*.csv'))
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
                temp_intensity = data[:, 25:]
                temp_condition = data[:, :25]
                if self.window_size is not None:
                    temp_intensity, temp_condition = self.reorganize_data(temp_intensity, temp_condition, window_size=self.window_size)
                intensity = temp_intensity
                condition = temp_condition
            else:
                temp_intensity = data[:, 25:]
                temp_condition = data[:, :25]
                if self.window_size is not None:
                    temp_intensity, temp_condition = self.reorganize_data(temp_intensity, temp_condition, window_size=self.window_size)
                intensity = np.concatenate((intensity, temp_intensity), axis=0)
                condition = np.concatenate((condition, temp_condition), axis=0)

            pbar.update(1)
            pbar.set_postfix({'progress': f'{i+1}/{len(files)}'})
        
        pbar.close()
        return intensity, condition
    
    def check_noise_map(self)->float:
        '''
        check the noise map of the data

        we can compare this with reconstruction loss of the model.
        if the noise map is high, the reconstruction loss is high.

        '''
        power_off_data = self.intensity[self.dataset.condition[:,0] == 0 & self.dataset.condition[:,4] == 0]

        power_off_data_mean = power_off_data.mean(dim=-1, keepdim=True)
        power_off_data_std = power_off_data.std(dim=-1, keepdim=True)

        return (power_off_data_std/power_off_data_mean).mean().item()*100

    def reorganize_data(self, intensity: np.ndarray, condition: np.ndarray, window_size: int = 2): 
        '''

        (F x N x D) -> (N x T x D)

        F : File number
        N : Number of data
        D : Number of features
        T : Number of sequence for prediction model

        More detailed step...

        (N x D) is the time series data of N data with D features.
        We want to apply the sliding window method to the data, for training the prediction model.
        So, we need to convert the data into (N x T x D) format.

        Then, make the random index for the data, with the size of,

        (N-T+1), by choosing the number between 0 and (N-T).

        we want to use these index to choose the data from the (N x D) format, like below,

        idx \in Idices { random number between 0 and (N-T) }
        one of sequence for dataset = F [ idx : idx + T ]

        we can get the dataset for file f,
        dataset = F [ idx : idx + T ] = (N-T+1) x T x D

        and, we can get the dataset for all files, by concatenating the dataset for each file,

        dataset = ((F x (N-T+1)) x T x D)

        Following is the code for the above process.

        '''

        N,D = intensity.shape

        indices = np.random.randint(0, N-window_size, size=(N-window_size+1,))

        intensity = intensity[indices]
        condition = condition[indices]

        intensity = intensity.reshape(N-window_size+1, window_size, D)
        condition = condition.reshape(N-window_size+1, window_size, 25)

        return intensity, condition
        



    
    