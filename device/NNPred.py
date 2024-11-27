import numpy as np
import torch
import torch.nn as nn

class Prediction():
    def __init__(self,encoder,er_model,sequence_length=8,hidden_dimension=16,process_variables = 7):
        self.window_l = TensorQueue(sequence_length,hidden_dimension)
        self.window_z = TensorQueue(sequence_length,hidden_dimension)
        self.er_vm = 0
        self.er_pred = 0
        
        self.structure = encoder
        self.ERmodel = er_model
    
    def encode(self,X):
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        Z = self.structure.encode(X)
        self.temp = Z.copy()
        self.WindowZInput(self.temp)
        return Z
    
    def rnn(self):
        self.Z_hat = self.structure(self.window_z,self.window_l)
        return self.Z_hat
    
    def ERVM(self):
        er_temp = self.ERmodel(self.temp).item()
        self.er_vm += er_temp
        return er_temp

    def ERPred(self):
        er_temp = self.ERmodel(self.Z_hat).item()
        self.er_pred += er_temp
        return er_temp

    def EtchAmount(self):
        return self.er_vm, self.er_pred

    def WindowZInput(self,z):
        self.window_z.enqueue(z)

    def WindowLInput(self,l):
        self.window_l.enqueue(l)
    
    def PredictionInit(self):
        self.er_vm, self.er_pred = 0, 0

class TensorQueue:
    def __init__(self,sequence_length, num_dimensions):
        self.tensors = torch.empty((sequence_length,num_dimensions))
    def pop(self):
        result = self.tensors[0]
        self.tensors = self.tensors[1:]
        return result
    def append(self,tensor):
        self.tensors = torch.cat([self.tensors,tensor],dim=0)
        return self.tensors
    def enqueue(self,tensor):
        self.append(tensor)
        self.pop()
