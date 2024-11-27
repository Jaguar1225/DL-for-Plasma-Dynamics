# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:53:29 2023

@author: user
"""

import sys
import glob
import warnings
import platform
import re
import numpy as np

from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dtype = torch.float32

class AEData(Dataset):
    def __init__(self,x_data,**kwargs):
        self.x_data = torch.FloatTensor(x_data)
        try:
            self.label = torch.FloatTensor(kwargs['label'])
        except:
            pass

        try:
            self.time = torch.FloatTensor(kwargs['time'])
            self.mode = 'time'
        except:
            self.mode = 'none'
            pass

        self.len = self.x_data.shape[0]
        self.cache = {}
        self.cache['device'] = 'cpu'
        self.cache['std'] = 'Org'
    
    def __call__(self):
        try:
            if self.mode == 'time':
                return self.x_data, self.label, self.time
            else:
                return self.x_data, self.label
        except:
            return self.x_data
    
    def __getitem__(self, index):
        try:
            if self.mode == 'time':
                return self.x_data[index], self.label[index], self.time[index]
            else:
                return self.x_data[index], self.label[index]
        except:
            return self.x_data[index]
    
    def __len__(self):
        return self.len
    def to(self, device):
        self.cuda(device)
    
    def cuda(self, device):
        self.x_data = self.x_data.to(device)
        try:
            self.label = self.label.to(device)
            if self.mode == 'time':
                self.time = self.time.to(device)
        except:
            pass
        self.cache['device'] = 'cuda'
        
    def cpu(self):
        self.x_data = self.x_data.to('cpu')
        try:
            self.label = self.label.to('cpu')
            if self.mode == 'time':
                self.time = self.time.to('cpu')
        except:
            pass
        self.cache['device'] = 'cpu'
    
    def std(self):
        if self.cache['std'] != 'std':
            try:
                if self.cache['std'] == 'Org':
                    self.origin = self.x_data.clone().to('cpu')

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.x_data = (self.x_data - self.x_data.mean(2,keepdims=True))/self.x_data.std(2,keepdims=True)
            
                self.x_data = torch.nan_to_num(self.x_data)
                self.cache['std'] = 'std'
            except:
                pass
        else:
            pass
    
    def nor(self):
        if self.cache['std'] != 'nor':
            try:
                if self.cache['std'] == 'Org':
                    self.origin = self.x_data.clone().to('cpu')
            
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.x_data = self.x_data/torch.norm(self.x_data,dim=len(self.x_data.size())-1,keepdim=True)
            
                self.x_data = torch.nan_to_num(self.x_data)
                self.cache['std'] = 'nor'
                
            except:
                pass
        
        else:
            pass
        
    def org(self):
        try:
            self.x_data = self.origin.to(self.cache['device'])
            del self.origin
            self.cache['std'] = 'Org'
        except:
            pass
    def clone(self):
        return deepcopy(self)
    
    def cache(self):
        print(self.cache)

def DataLoader(path='Data/Train', T=0, test_size=0, random_seed=0, mode=None, label=False):
    """
    통합된 데이터 로더
    Args:
        path: 데이터 경로
        T: sequence length (0이면 AE 모드, 1 이상이면 RNN 모드)
        test_size: 테스트 세트 비율
        random_seed: 랜덤 시드
        mode: 'time' 또는 None
        label: AE 모드에서 라벨 반환 여부
    """
    
    Start = True
    delimiter = '\\' if platform.system() == 'Windows' else '/'
    
    files = glob.glob(f'{path}/*.csv')
    for xl_path in tqdm(files, desc='Loading data files', leave=False):
        if Start:
            # 기본 데이터 로드
            xl = np.loadtxt(xl_path, delimiter=',', dtype=str)
            wavelengths = xl[0,26:].astype(float)
            x = xl[1:,26:].astype(float)
            l = xl[1:,[2,3,4,5,7,8,9,10,12,13,17,19,21,22,23,24]].astype(float)
            x[x<0] = 0
            
            if T > 0:  # RNN 모드
                N = len(x)
                x[(l[:,0]==0) & (l[:,4]==0)] = np.ones_like(x[(l[:,0]==0) & (l[:,4]==0)])
                temp_idx = np.random.choice(np.arange(0,N-T+1,step=1), size=N-T+1, replace=False)
                temp_idx = (temp_idx[:,None] + np.arange(T)).flatten()
                x_data = x[temp_idx].reshape(N-T+1,T,-1)[:,[0,-1]]
                l_data = l[temp_idx].reshape(N-T+1,T,-1)
                t_data = np.array([0.5]).repeat(N-T+1).reshape(-1,1)
            else:  # AE 모드
                condition = (l[:,0]==0) & (l[:,4]==0)
                x = x[~condition]
                x_data = x
                if label:
                    l = l[~condition]
                    l_data = l
            
            Start = False
            
        elif not Start:
            xl = np.loadtxt(xl_path, delimiter=',', dtype=str)
            x = xl[1:,26:].astype(float)
            l = xl[1:,[2,3,4,5,7,8,9,10,12,13,17,19,21,22,23,24]].astype(float)
            x[x<0] = 0
            
            if T > 0:  # RNN 모드
                N = len(x)
                x[(l[:,0]==0) & (l[:,4]==0)] = np.ones_like(x[(l[:,0]==0) & (l[:,4]==0)])
                temp_idx = np.random.choice(np.arange(0,N-T+1,step=1), size=N-T+1, replace=False)
                temp_idx = (temp_idx[:,None] + np.arange(T)).flatten()
                x = x[temp_idx].reshape(N-T+1,T,-1)[:,[0,-1]]
                l = l[temp_idx].reshape(N-T+1,T,-1)
                t = np.array([0.5]).repeat(N-T+1).reshape(-1,1)
                x_data = np.r_[x_data, x]
                l_data = np.r_[l_data, l]
                t_data = np.r_[t_data, t]
            else:  # AE 모드
                condition = (l[:,0]==0) & (l[:,4]==0)
                x = x[~condition]
                x_data = np.r_[x_data, x]
                if label:
                    l = l[~condition]
                    l_data = np.r_[l_data, l]
    
    # 데이터 크기 계산 및 출력
    size = sys.getsizeof(x_data)
    size_str = f"{size/1024**3:.1f}GB" if size/1024**3>1 else f"{size/1024**2:.1f}MB"
    print(f"Loaded {len(x_data):,} {'sequences' if T > 0 else 'samples'} ({size_str})")
    
    # 데이터 분할 및 반환
    if test_size > 0:
        if T > 1:  # RNN 모드
            train_X, test_X, train_L, test_L, train_t, test_t = train_test_split(
                x_data, l_data, t_data, test_size=test_size, random_state=random_seed)
            if mode == 'time':
                return AEData(train_X, label=train_L, time=train_t), AEData(test_X, label=test_L, time=test_t)
            return AEData(train_X, label=train_L), AEData(test_X, label=test_L)
        else:  # AE 모드
            train_X, test_X = train_test_split(x_data, test_size=test_size, random_state=random_seed)
            if label:
                return AEData(train_X), AEData(test_X), l_data
            return AEData(train_X), AEData(test_X)
    else:
        if T > 1:  # RNN 모드
            train_X, train_L, train_t = shuffle(x_data, l_data, t_data, random_state=random_seed)
            if mode == 'time':
                return AEData(train_X, label=train_L, time=train_t)
            return AEData(train_X, label=train_L)
        else:  # AE 모드
            train_X = shuffle(x_data, random_state=random_seed)
            if label:
                return AEData(train_X), l_data
            return AEData(train_X)

def TestDataLoader(path='Data/Train', T=0, test_size=0, random_seed=0, mode=None, label=False):
    """
    테스트 데이터 로더
    """
    file_label = {
        'Power':'RF source power',
        'Bias' : 'RF bias power',
        'Pr' : 'Pressure',
        'Ar' : 'Argon flow rate',
        'O' : 'Oxygen flow rate',
        'CF' : 'Fluorocarbon flow rate'
    }
    TestData ={}
    Start = True
    delimiter = '\\' if platform.system() == 'Windows' else '/'
    
    files = glob.glob(f'{path}/*.csv')
    for xl_path in tqdm(files, desc='Loading data files', leave=False):
        if 'Train' in path:
            xl_name = xl_path.split(delimiter)[-1]
            xl_label = re.search(r'(\d+)\.csv', xl_name).group(1)
        else:
            xl_name = xl_path.split(delimiter)[-1]
            xl_label, percentage = re.search(r'([A-Za-z]+)(\d+)\.csv', xl_name).groups()
            xl_label = file_label[xl_label]
            percentage = float(percentage)
        
        if Start:
            # 기본 데이터 로드
            xl = np.loadtxt(xl_path, delimiter=',', dtype=str)
            wavelengths = xl[0,26:].astype(float)
            x = xl[1:,26:].astype(float)
            l = xl[1:,[2,3,4,5,7,8,9,10,12,13,17,19,21,22,23,24]].astype(float)
            x[x<0] = 0

            if T > 0:  # RNN 모드
                N = len(x)
                x[(l[:,0]==0) & (l[:,4]==0)] = np.ones_like(x[(l[:,0]==0) & (l[:,4]==0)])
                temp_idx = np.arange(0,N-T+1,step=1)
                temp_idx = (temp_idx[:,None] + np.arange(T)).flatten()
                x_data = x[temp_idx].reshape(N-T+1,T,-1)[:,[0,-1]]
                l_data = l[temp_idx].reshape(N-T+1,T,-1)
                t_data = np.array([0.5]).repeat(N-T+1).reshape(-1,1)
            else:
                condition = (l[:,0]==0) & (l[:,4]==0)
                x_data = x[~condition]
                if label:
                    l_data = l[~condition]
            
            Start = False
        
        else:
            xl = np.loadtxt(xl_path, delimiter=',', dtype=str)
            x = xl[1:,26:].astype(float)
            l = xl[1:,[2,3,4,5,7,8,9,10,12,13,17,19,21,22,23,24]].astype(float)
            x[x<0] = 0

            if T > 0:
                N = len(x)
                x[(l[:,0]==0) & (l[:,4]==0)] = np.ones_like(x[(l[:,0]==0) & (l[:,4]==0)])  
                temp_idx = np.arange(0,N-T+1,step=1)
                temp_idx = (temp_idx[:,None] + np.arange(T)).flatten()
                x_data = x[temp_idx].reshape(N-T+1,T,-1)[:,[0,-1]]
                l_data = l[temp_idx].reshape(N-T+1,T,-1)
                t_data = np.array([0.5]).repeat(N-T+1).reshape(-1,1)
            else:
                condition = (l[:,0]==0) & (l[:,4]==0)
                x_data = x[~condition]
                if label:
                    l_data = l[~condition]
        if 'Train' in path:
            data_label = xl_label
        else:
            data_label = xl_label, percentage

        if T > 1:
            if mode == 'time':
                TestData[data_label] = AEData(x_data, label=l_data, time=t_data)
            else:
                TestData[data_label] = AEData(x_data, label=l_data)
        else:
            if label:
                TestData[data_label] = AEData(x_data, label=l_data)
            else:
                TestData[data_label] = AEData(x_data)
    return TestData
