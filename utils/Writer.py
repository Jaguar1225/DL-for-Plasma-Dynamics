import torch
import torch.nn.functional as F
import os
import shutil
import pickle
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import time

class StepWriter:
    def _step_writer_init(self, fold=None):
        if os.path.exists(f'{self.params["Writer_dir"]}/{self.params["keyword"]}'):
            shutil.rmtree(f'{self.params["Writer_dir"]}/{self.params["keyword"]}')
            try:
                shutil.rmtree(f'{self.params["Writer_dir"]}/{self.params["keyword"]} image')
            except:
                pass
                
        self.params["Writer"] = SummaryWriter(
            log_dir=f'{self.params["Writer_dir"]}/{self.params["keyword"]}'
        )

    def _step_summary(self, loss, title='/train loss'):
        self.params['Writer'].add_scalar(title, loss, self.step)

    def _step_writer_save(self, logdir, save_dir):
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        phases = ['/train','/val']
        loss_types = ['loss']

        if self.params['cos_weight']>0 or self.params['l1_weight']>0:
            loss_types.append('mse loss')
            if self.params['cos_weight']>0:
                loss_types.append('cos loss')
            if self.params['l1_weight']>0:
                loss_types.append('l1 loss')

        for phase in phases:
            for loss_type in loss_types:
                save_path = f'Result/{self.__class__.__name__}/{save_dir}/{phase} {loss_type}/'
                os.makedirs(save_path, exist_ok=True)

                tag = f'{phase} {loss_type}'
                loss_data = event_acc.Scalars(tag)
                loss_data = np.array([[event.step, event.value] for event in loss_data])
                
                file_path = f'{save_path}/{phase} {loss_type} {self.date}_{self.params["keyword"]}.csv'
                np.savetxt(file_path, loss_data, delimiter=',', 
                          header='Epoch,loss', comments='', fmt='%d,%.12f')

    def _memory_check(self):
        if not self.params["device"] == torch.device('cpu'):
            total_memory = torch.cuda.get_device_properties(self.params['device']).total_memory
            reserved_memory = torch.cuda.memory_reserved(self.params['device'])
            allocated_memory = torch.cuda.memory_allocated(self.params['device'])
            peak_memory = torch.cuda.max_memory_allocated(self.params['device'])

            reserved_memory_ratio = reserved_memory / total_memory * 100
            allocated_memory_ratio = allocated_memory / total_memory * 100
            peak_memory_ratio = peak_memory / total_memory * 100

            total_memory_gb = round(total_memory / (1024 ** 3), 2)
            reserved_memory_gb = round(reserved_memory / (1024 ** 3), 2)
            allocated_memory_gb = round(allocated_memory / (1024 ** 3), 2)
            peak_memory_gb = round(peak_memory / (1024 ** 3), 2)

            self._step_memory_summary(
                self.step, total_memory_gb, reserved_memory_gb,
                allocated_memory_gb, peak_memory_gb,
                reserved_memory_ratio, allocated_memory_ratio, peak_memory_ratio
            )
        else:
            raise RuntimeError("CUDA is not available. Please check your NVIDIA GPU and driver.")

    def _step_memory_summary(self, n, total_memory_gb, reserved_memory_gb, 
                           allocated_memory_gb, peak_memory_gb,
                           reserved_memory_ratio, allocated_memory_ratio, peak_memory_ratio):
        """메모리 데이터 기록 - writer가 None인 경우 건너뛰기"""
        writers_data = {
            'Memory_writer_Total': ('/GPU_memory', total_memory_gb),
            'Memory_writer_Reserved': ('/GPU_memory', reserved_memory_gb),
            'Memory_writer_Allocated': ('/GPU_memory', allocated_memory_gb),
            'Memory_writer_Peak': ('/GPU_memory', peak_memory_gb),
            'Memory_writer_Reserved_ratio': ('/GPU_memory_ratio', reserved_memory_ratio),
            'Memory_writer_Allocated_ratio': ('/GPU_memory_ratio', allocated_memory_ratio),
            'Memory_writer_Peak_ratio': ('/GPU_memory_ratio', peak_memory_ratio)
        }
        
        for writer_name, (tag, value) in writers_data.items():
            writer = self.params.get(writer_name)
            if writer is not None:
                try:
                    writer.add_scalar(tag, value, n)
                except Exception as e:
                    print(f"Warning: Failed to write {writer_name} data: {e}")

    def _step_image_summary(self, title, tensor, e):
        tensor = tensor.detach().cpu()
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)
        
        epsilon = 1e-8
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min > epsilon:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min + epsilon)
        else:
            tensor = torch.zeros_like(tensor)
        
        try:
            grid = utils.make_grid(tensor.data, normalize=False, scale_each=True)
            self.params['image_writer'].add_image(title, grid, e)
        except Exception as e:
            print(f"Error in make_grid: tensor shape = {tensor.shape}")
            raise e

    def _memory_writer_init(self):
        # 날짜/시간 기반 하위 디렉토리 구조
        current_date = time.strftime('%Y-%m-%d')
        current_hour = time.strftime('%H')
        
        # 기본 writer 디렉토리 구조화
        writer_base = os.path.normpath(f'{self.params["Writer_dir"]}/{current_date}/{current_hour}/{self.params["keyword"]}')
        
        # 메모리 모니터링용 writer들 초기화
        memory_writers = {
            'Total': None,
            'Reserved': None,
            'Allocated': None,
            'Peak': None,
            'Reserved_ratio': None,
            'Allocated_ratio': None,
            'Peak_ratio': None
        }
        
        # writer 초기화 및 이전 로그 정리
        for writer_name in memory_writers.keys():
            writer_dir = os.path.join(writer_base, writer_name)
            
            # 이전 로그 파일 정리
            if os.path.exists(writer_dir):
                try:
                    shutil.rmtree(writer_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove old logs in {writer_dir}: {e}")
            
            # 새 디렉토리 생성
            os.makedirs(writer_dir, exist_ok=True)
            
            # writer 초기화
            try:
                memory_writers[writer_name] = SummaryWriter(log_dir=writer_dir)
            except Exception as e:
                print(f"Warning: Failed to initialize {writer_name} writer: {e}")
                memory_writers[writer_name] = None
        
        # writer 참조 저장
        self.params.update({
            f'Memory_writer_{key}': writer 
            for key, writer in memory_writers.items()
        })

    def _summary_writer_close(self):
        self.params['Writer'].close()
        try:
            self.params['image_writer'].close()
        except:
            pass
    def _memory_writer_close(self):
        """활성화된 writer만 종료"""
        for key in self.params:
            if key.startswith('Memory_writer_'):
                writer = self.params[key]
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as e:
                        print(f"Warning: Failed to close {key}: {e}")

def plot_comparison(original, reconstructed, h, date, model_name):
    plt.figure(figsize=(20, 10))
    n = min(10, original.shape[0])
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.plot(original[i])
        plt.title(f"Original {i+1}")
        plt.grid(True)
        
        plt.subplot(2, n, i+n+1)
        plt.plot(reconstructed[i])
        plt.title(f"Reconstructed {i+1}")
        plt.grid(True)
    
    plt.tight_layout()
    save_dir = f'Fig/{model_name}/{date}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/reconstruction_comparison_hidden_{h}.png')
    plt.close()

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed)**2)
