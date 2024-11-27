import time
import traceback
import datetime
import sys
import argparse


import torch

from utils.report import report_to_mail


from utils.trainer import ModelTrainer

from structures import *
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
dtype = torch.float32
date = time.strftime('%Y-%m-%d', time.localtime())


import os
import sys
import time
import subprocess
import signal

def get_parent_pid():
    return os.getppid()

def terminate_parent_process():
    parent_pid = get_parent_pid()
    try:
        os.kill(parent_pid, signal.SIGTERM)
        print(f"부모 프로세스 (PID: {parent_pid})를 종료합니다.")
    except ProcessLookupError:
        print("부모 프로세스를 찾을 수 없습니다.")
    except PermissionError:
        print("부모 프로세스를 종료할 권한이 없습니다.")

def get_specific_processes(include_keywords, exclude_keywords):
    try:
        output = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid,command'], universal_newlines=True)
        lines = output.strip().split('\n')[1:]
        processes = []
        for line in lines:
            if all(keyword.lower() in line.lower() for keyword in include_keywords) and \
               not any(keyword.lower() in line.lower() for keyword in exclude_keywords):
                processes.append(line.strip())
        current_pid = os.getpid()
        processes = [proc for proc in processes if str(current_pid) not in proc.split()[0]]
        return processes
    except subprocess.CalledProcessError as e:
        print(f"프로세스 확인 중 오류가 발생했습니다: {e}")
        return None

def parse_list(arg):
    try:
        return [eval(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("리스트는 쉼표로 구분된 정수여야 합니다.")

def parse_bool(arg):
    try:
        return eval(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("주어진 인자는 bool이 아닙니다.")

def parse_models(arg):
    try:
        return eval(arg)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def parse_hidden(arg):
    try:
        return int(2**eval(arg))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--h', '--hidden', type=parse_hidden, help='Hidden dimensions')
    parser.add_argument('--m', '--model', type=parse_models, help='Model class')
    parser.add_argument('--s', '--sequence', type=int, help='Sequence length')
    args = parser.parse_args()
    
    # 시작 시간 기록
    start_time = time.time()
    try:
        params ={
            'model_class': Autoencoder,
            'hidden_dimension': 32,
            'sequence_length': 2,
            'layer_dimension': 5,
            'num_trial': 3,
            'epoch': 2**12,
            'use_writer': True,
            'date': date,
            'device': device
        }
        params.update(
            {
                'hidden_dimension': args.h or params['hidden_dimension'],
                'model_class': args.m or params['model_class'],
                'sequence_length': args.s or params['sequence_length']
            }
        )
        
        if "Autoencoder" in params['model_class'].__name__:
            params['sequence_length'] = 0
        # 4. 실제 학습 실행
        start_time = time.time()
        model_trainer = ModelTrainer(**params)
        model_trainer.train()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Complete! {datetime.timedelta(seconds= elapsed_time)} elapsed")

        # 포함할 키워드와 제외할 키워드 설정
        include_keywords = ['python', '__main__.py']  # 반드시 포함해야 할 키워드들
        exclude_keywords = ['cursor', 'idle', 'python3']  # 제외할 키워드들

        # 5. 모든 작업이 완료된 후에만 프로세스 체크
        specific_processes = get_specific_processes(include_keywords, exclude_keywords)

        if specific_processes:
            print(f"지정된 조건에 맞는 활성 프로세스 목록:")
            for proc in specific_processes:
                print(proc)
            print(f"\n총 {len(specific_processes)}개의 관련 프로세스가 실행 중입니다.")
            terminate_parent_process()
            print("현재 프로세스를 종료합니다.")
            sys.exit(0)

        else:
            print("지정된 조건에 맞는 실행 중인 프로세스가 없습니다.")
            model_trainer.final_plot()
            val_layer_dim_convergence, val_h_or_t_convergence = model_trainer.convergence_check()
            print(f'val_layer_dim_convergence: {val_layer_dim_convergence}')
            print(f'val_h_or_t_convergence: {val_h_or_t_convergence}')
            if "RNN" in model_trainer.model_class.__name__ or "PlasDyn" in model_trainer.model_class.__name__:
                layer_dim_convergence = [16 for _ in range(val_layer_dim_convergence)] + [model_trainer.hidden_dimension]
                sequence_length = val_h_or_t_convergence
                hidden_dimension = params['hidden_dimension']
            else:
                layer_dim_convergence = [3648 for _ in range(val_layer_dim_convergence)] + [val_h_or_t_convergence]
                sequence_length = params['sequence_length']
                hidden_dimension = val_h_or_t_convergence

            model_params = {
                'model_class': model_trainer.model_class,
                'date': model_trainer.date,
                'device': model_trainer.device,
                'process_variables': 16,
                'epoch': model_trainer.epoch,
                'num_trial': 1,
                'hidden_dimension': hidden_dimension,
                'layer_dimension': layer_dim_convergence,
                'sequence_length': sequence_length,
                'use_writer': True,
                'keyword': f'{model_trainer.model_class.__name__} {val_h_or_t_convergence} {val_layer_dim_convergence}',
            }
            start_time = time.time()

            model = params['model_class'](**model_params)
            model.to(params['device'])

            from utils.Data_loader import DataLoader as DL
            train_data = DL(T=sequence_length, random_seed=42)
            train_data.nor()
            train_data.to(params['device'])

            train_loss, train_recon_loss, train_cos_loss, train_l1_loss, \
                val_loss, val_recon_loss, val_cos_loss, val_l1_loss = \
                    model.update(train_data,epoch=params['epoch'],use_writer=params['use_writer'])
            model_trainer.save_path = model.params['save_path']
            model_trainer._save_model(model, model_trainer.save_path, model_params['keyword'])

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f'{model_trainer.model_class.__name__} {val_h_or_t_convergence} {val_layer_dim_convergence} completed')
            print(f'loss: {train_loss}, val_loss: {val_loss}')
            print(f'time: {datetime.timedelta(seconds=elapsed_time)}')
            
    except Exception as e:
        current_time = time.strftime('%Y-%m-%d', time.localtime())
        elapsed_time = time.time() - start_time
        print(f"Error occurred in main process: {str(e)}")
        report_to_mail(
            f"Error occurred in main process",
            'kth102938@g.skku.edu',
            'ronaldo1225!',
            'code.jaguar1225@gmail.com',
            contents = f'''Error {e} occured at {current_time}.
                    Program proceeded during {datetime.timedelta(seconds= elapsed_time)}.
                    {traceback.format_exc()}'''
        )
    
