# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from models.cnn_feedforward_div_norm import cnn_feedforward_div_norm
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

def main():

    # define root
    dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

    # adapt = 'div_norm'
    layers = ['conv1', 'sconv1', 'conv2', 'conv3', 'fc1']
    layer = ['conv3']
    params = ['tau1_init', 'tau2_init', 'sigma_init'] 

    # import grid search data
    df1 = pd.read_csv('accu/grid_search2/meta.txt', index_col=0, sep=' ', header=0)
    # print(df1)

    # import grid search data
    df2 = pd.read_csv('accu/grid_search3/meta.txt', index_col=0, sep=' ', header=0)
    # print(df2)

    # concatenate 
    df = pd.concat([df1, df2])
    df = df.sort_values('Init')
    df.reset_index(drop=True, inplace=True)
    # print(df)

    inits = df['Init'].unique()
    for n, init in enumerate(inits):
        print(n, ': ', init)

    noise_patterns = ['same', 'different']
    color = ['dodgerblue', 'darkorange']
    acc = ['acc1', 'acc2', 'acc3']

    # # determine minimal and maximal accuracy
    # acc_min = torch.min(torch.Tensor(df1['acc']))
    # print('Minimal accuracy: ', acc_min)
    # acc_max = torch.max(torch.Tensor(df1['acc']))
    # print('Maximal accuracy: ', acc_max)

    # # define accuracy bins
    # acc_bins = np.arange(0, 1.2, 0.2)
    # print('Accuracy bins: ', acc_bins)

    # # make selection
    # thrs = 0.6
    # df1_select = df1[df1.acc > thrs]
    # df1_select = df1_select.sort_values('acc')
    # df1_select.reset_index(drop=True, inplace=True)
    # print(df1_select)
    # print('Number of networks with accuracy above %i: %i' % (thrs*100, len(df1_select)))

    # # load testing data
    # batch_size = 64
    # t_steps = 10
    # dataset = 'test'
    # contrast = 'lcontrast'

    # noise = 'same'
    # noise_imgs_same = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_imgs_' + contrast)
    # noise_lbls_same = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_lbls_' + contrast)
    # dt_same = noiseMNIST_dataset(noise_imgs_same, noise_lbls_same)
    # print('Shape training set (same): ', noise_imgs_same.shape, ', ', noise_lbls_same.shape)

    # noise = 'different'
    # noise_imgs_different = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_imgs_' + contrast)
    # noise_lbls_different = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_lbls_' + contrast)
    # dt_different = noiseMNIST_dataset(noise_imgs_different, noise_lbls_different)
    # print('Shape training set (same): ', noise_imgs_different.shape, ', ', noise_lbls_different.shape)

    rows = 3
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    num = 0
    for i in range(rows):
        for j in range(cols):

            # set parameters
            df_select = df[df.Init == inits[num]]
            df_select.reset_index(drop=True, inplace=True)
            axs[i,j].set_title('Initialization ' + str(inits[num]))

            for n, noise in enumerate(noise_patterns):
                  
                # select noise condition
                idx = df_select[df_select.noise == noise].index
                acc_temp = np.array(df_select.loc[idx, acc])

                # plot accuracies
                axs[i,j].scatter(np.ones(len(acc))*n, acc_temp*100, color=color[n], alpha=0.5, s=20, zorder=1)
                axs[i,j].scatter(n, np.mean(acc_temp)*100, color=color[n], s=80, zorder=2)
            axs[i,j].set_xlim(-0.5, 1.5)
            axs[i,j].set_ylim(59, 75)
            axs[i,j].set_xticks([0, 1])
            axs[i,j].set_xticklabels(['Same', 'Diff.'])
            if j == 0:
                axs[i,j].set_ylabel('Accuracy (%)')
            if i == rows-1:
                axs[i,j].set_xlabel('Training data')

            # increment count
            num+=1

    # save figure
    plt.tight_layout()
    plt.savefig('visualizations/grid_search', dpi=300)


if __name__ == '__main__':
    main()