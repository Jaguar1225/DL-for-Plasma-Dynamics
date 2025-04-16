import time
import traceback
import sys
import argparse
import json

import torch

from utils.report import mail
from utils.trainer import *

from structures import *

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
        os.kill(
            parent_pid, 
            signal.SIGTERM
            )
        print(f"부모 프로세스 (PID: {parent_pid})를 종료합니다.")
    except ProcessLookupError:
        print("부모 프로세스를 찾을 수 없습니다.")
    except PermissionError:
        print("부모 프로세스를 종료할 권한이 없습니다.")

def get_specific_processes(include_keywords: list[str], exclude_keywords: list[str])->list[str]:
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

def parse_list(arg: str)->list[int]:
    try:
        return [eval(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("리스트는 쉼표로 구분된 정수여야 합니다.")

def parse_bool(arg: str)->bool:
    try:
        return eval(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("주어진 인자는 bool이 아닙니다.")

def parse_models(arg: str)->str:
    try:
        return eval(arg)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def load_config(config_path: str)->dict:
    """설정 파일을 로드합니다."""
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            json_config = json.load(f)
    else:
        raise ValueError("지원하지 않는 설정 파일 형식입니다. .json 또는 .yaml/.yml 파일을 사용하세요.")
    return json_config

def get_model_params(params: dict, args: argparse.Namespace)->dict:
    """모델 파라미터를 설정합니다."""    
    # 명령줄 인자로 전달된 파라미터 업데이트
    if args.model:
        params['model'] = args.model
    if args.input_dim:
        params['input_dim'] = args.input_dim
    if args.hidden_dims:
        params['hidden_dims'] = args.hidden_dims
    if args.layer_type:
        params['layer_type'] = args.layer_type
    if args.activation:
        params['activation_function'] = args.activation
    if args.optimizer:
        params['optimizer'] = args.optimizer
    if args.learning_rate:
        params['optimizer_params'] = params.get('optimizer_params', {})
        params['optimizer_params']['lr'] = args.learning_rate
    if args.device:
        params['device'] = torch.device(args.device)
    if args.log_dir:
        params['log_dir'] = args.log_dir
    if args.num_layers:
        params['num_layers'] = args.num_layers
    if args.num_epochs:
        params['num_epochs'] = args.num_epochs

    return params

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train models')
    # 모델 관련 인자
    parser.add_argument('--model', '-m',  type=parse_models, help='Model class')
    parser.add_argument('--config', '-c', type=str, help='설정 파일 경로 (.json 또는 .yaml)')
    parser.add_argument('--data_type', '-dt', type=str, help='데이터 타입')

    # 모델 아키텍처 관련 인자
    parser.add_argument('--input_dim', '-i', type=int, help='입력 차원')
    parser.add_argument('--hidden_dims', '-hd', type=parse_list, help='은닉층 차원 리스트 (쉼표로 구분)')
    parser.add_argument('--layer_type', '-lt', type=str, choices=['unit_coder', 'log_unit_encoder', 'log_unit_decoder', 'unit_transformer'],
                      help='레이어 타입')
    parser.add_argument('--activation', '-a', type=str, help='활성화 함수')
    
    # 옵티마이저 관련 인자
    parser.add_argument('--optimizer', '-o', type=str, choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
                      help='옵티마이저 타입')
    parser.add_argument('--learning_rate', '-lr', type=float, help='학습률')
    
    # 기타 설정
    parser.add_argument('--device', '-d', type=str, choices=['cuda', 'cpu'], help='사용할 디바이스')
    parser.add_argument('--scheduler', '-s', type=str, choices=['reduceLROnPlateau', 'stepLR', 'cosineAnnealingLR'],
                      help='스케줄러 타입')
    
    parser.add_argument('--scheduler_params', '-sp', type=str, help='스케줄러 파라미터')
    parser.add_argument('--optimizer_params', '-op', type=str, help='옵티마이저 파라미터')
    parser.add_argument('--reconstruction_weight', '-rw', type=float, help='재구성 손실 가중치')
    parser.add_argument('--regularization_weight', '-regw', type=float, help='정규화 손실 가중치')

    parser.add_argument('--batch_size', '-bs', type=int, help='배치 크기')

    parser.add_argument('--num_layers', '-nl', type=int, help='레이어 수')
    parser.add_argument('--num_epochs', '-ne', type=int, help='학습 반복 횟수')

    parser.add_argument('--log_dir', '-ld', type=str, help='텐서보드 로그 디렉토리')
    
    args = parser.parse_args()
    
    # 시작 시간 기록
    start_time = time.time()
    
    try:
        # 모델 파라미터 설정
        params = load_config('./configs/model_config.json')
        params = get_model_params(params, args)
        
        # 모델 생성 및 훈련
        if params['model'].lower() == 'autoencoder':
            trainer = AE_Trainer(**params)
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {args.model}")
        trainer.train()
        
        # 종료 시간 기록
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"훈련 완료! 실행 시간: {execution_time:.2f}초")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print(traceback.format_exc())
        terminate_parent_process()
        sys.exit(1)
