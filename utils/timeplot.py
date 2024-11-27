import matplotlib.pyplot as plt
import numpy as np
import glob

anomaly = 'Test'
dir_path = f'Data/{anomaly}'
Variables = ['Power','Bias', 'Pr','O','Ar','CF']
percent = [10,5,1]

file_path = []
for v in Variables:
    for p in percent:
        file_path.append(dir_path+f'/{v}{p}.csv')
    
subplot_num = len(file_path)
subplot_size = 10
subplot_row = 6
subplot_col = subplot_num // subplot_row
subplot_title_size = 5
title_size = 5



for v in Variables:
    fig, ax = plt.subplots(1,3,figsize=(subplot_size*3,subplot_size))
    fig.suptitle(f'{anomaly} {v} raw data',fontsize=10,fontweight='bold',y=1)
    ax = ax.flatten()

    for idx, p in enumerate(percent):
        path = dir_path+f'/{v}{p}.csv'
        data = np.loadtxt(path, delimiter=',',skiprows=1,usecols=(1000,2000,3000))
        print(path)
        time = np.arange(data.shape[0]*0.5,step=0.5)
        ax[idx].plot(time,data[:,0],label='1000',color='red')
        ax[idx].plot(time,data[:,1],label='2000',color='green')
        ax[idx].plot(time,data[:,2],label='3000',color='blue')
        ax[idx].set_title(path.split('/')[-1].split('.')[0],fontsize=10,fontweight='bold')
        ax[idx].set_xlabel('Time [s]',fontsize=10,fontweight='bold')
        ax[idx].set_ylabel('Intensity [a.u.]',fontsize=10,fontweight='bold')
        ax[idx].legend(fontsize=10)
        ax[idx].grid()
        ax[idx].tick_params(axis='both', which='major', labelsize=10)
        for axis in ['top','bottom','left','right']:
            ax[idx].spines[axis].set_linewidth(1)

    plt.tight_layout()
    plt.savefig(f'{anomaly}_{v}_raw_data.png',dpi=300)

