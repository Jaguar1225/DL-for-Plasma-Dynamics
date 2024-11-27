import numpy as np
import math
'''
Normalization 
'''

class Clustering():
    def __init__(self,time_window = 10, num_wl= 3648, Nor = None, DWT = None, Selected_wave = None, Noise = None,
                 cluster_method = 'KMC'):
        self.window = np.empty(shape=(time_window,num_wl))

    def KMC(self):
        self.Center_Cluster_1 = self.window[0:1]
        self.Center_Cluster_2 = self.window[-2:-1]
        i=0
        while True:
            self.dist_1 = np.linalg.norm(self.window-self.Center_Cluster_1,axis=1)
            self.dist_2 = np.linalg.norm(self.window-self.Center_Cluster_2,axis=1)
            temp_label = np.argmin(np.c_[self.dist_1,self.dist_2],axis=1)

            if i==0:
                i += 1
                self.label = temp_label.copy()
            else:
                if np.all(temp_label == self.label):
                    break
            self.label = temp_label
            self.Center_Cluster_1 = self.window[self.label==0].mean(axis=0,keepdims=True)
            self.Center_Cluster_2 = self.window[self.label==1].mean(axis=0,keepdims=True)


        return self.ValidityScore()

    def GMM(self):
        trunc_size = math.trunc(len(self.window)/3)
        Cluster_1 = self.window[:trunc_size]
        Cluster_2 = self.window[-trunc_size:]

        mu_1,std_1 = Cluster_1.mean(axis=0), Cluster_1.std(axis=0)
        mu_2,std_2 = Cluster_2.mean(axis=0), Cluster_2.std(axis=0)
        i = 0
        while True:
            temp_label = np.argmax(
                np.c_[
                    self.Gaussian1D(self.window,mu_1,std_1),
                    self.Gaussian1D(self.window,mu_2,std_2)],
                axis=1
            )
            if i==0:
                i += 1
                self.label = temp_label.copy()
                continue
            if np.all(temp_label==self.label):
                break

            mu_1, std_1 = self.window[self.label==0].mean(axis=0), self.window[self.label==0].std(axis=0)
            mu_2, std_2 = self.window[self.label==1].mean(axis=0), self.window[self.label==1].std(axis=0)

        return self.ValidityScore()

    def Gaussian1D(self,x,mu,std):
        return (1 / (std@std.T)**0.5)*np.exp(-(x-mu)@(x-mu).T/std@std.T)

    def SC(self):
        X = self.Normalize(self.window)
        distance_matrix = (X**2).sum(axis=1)-\
                          2*X@X.T + \
                          (X.T**2).sum(axis=0)

        A = np.exp(-distance_matrix)
        D = np.eye(len(distance_matrix))*A.sum(axis=1)
        L = D-A
        NorL = D**(-0.5)@L@D**(-0.5)

        eigV, eigVec = np.linalg.eig(NorL)
        label = eigVec[eigV.argsort()[1]]

        label[label>0]=1
        label[label<0]=0

        self.label = label

        return self.ValidityScore()

    def Normalize(self,X):
        return (X-X.mean(axis=0))/(X.std(axis=0))


    def ValidityScore(self):

        Center_Cluster_1 = np.mean(self.window[self.label == 0], axis=0)
        Center_Cluster_2 = np.mean(self.window[self.label == 1], axis=0)

        dist_1 = np.linalg.norm(self.window[self.label==0] - Center_Cluster_1,axis=1)
        dist_2 = np.linalg.norm(self.window[self.label==1] - Center_Cluster_2, axis=1)
        dist_Overall = np.linalg.norm(self.window-self.window.mean(axis=0),axis=1)

        return (dist_1.sum()+dist_2.sum())/dist_Overall.sum()

    def WindowInput(self,OES_Raw):
        self.window = np.append(self.window,OES_Raw,axis=0)
        self.window = self.window[1:]