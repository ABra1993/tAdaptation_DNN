# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
import numpy as np
from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.functions import *

def create_stimuli(data):

    # define root
    linux = True # indicates whether script is run on mac or linux
    if linux:
        dir = '/home/amber/ownCloud/'
    else:
        dir = '/Users/a.m.brandsuva.nl/surfdrive/'

    # download data
    if data == 'train':
        dt, _, data_n, _ = download_data()
    elif data == 'test':
        _, dt, _, data_n = download_data()
    print(data, ': ', data_n)

    # define input shape
    input_shape = dt[0][0].shape
    w = input_shape[1]
    h = input_shape[1]
    c = input_shape[0]
    print('Input shape: ', input_shape)

    # grayscale image
    x = torch.ones(input_shape)
    isi = torch.multiply(x, 0.5)

    # dataset settings
    img_num = data_n
    # img_num = 2

    noise_imgs = torch.empty([img_num*3, 3, c, w, h])
    noise_lbls = torch.empty(img_num*3, dtype=torch.long)
    noise_pattern = []

    for i in range(img_num):

        # select img
        img = dt[i][0]
        noise_lbls[i*3:i*3+3] = torch.ones(3)*dt[i][1]

        # none
        test = torch.rand(input_shape)
        adapter = isi
        noise_imgs[i*3, 0, :, :, :] = adapter
        noise_imgs[i*3, 1, :, :, :] = isi
        noise_imgs[i*3, 2, :, :, :] = test + img
        noise_pattern.append('none')

        # same
        test = torch.rand(input_shape)
        adapter = test
        noise_imgs[i*3+1, 0, :, :, :] = adapter
        noise_imgs[i*3+1, 1, :, :, :] = isi
        noise_imgs[i*3+1, 2, :, :, :] = test + img
        noise_pattern.append('same')

        # different
        test = torch.rand(input_shape)
        adapter = torch.rand(input_shape)
        noise_imgs[i*3+2, 0, :, :, :] = adapter
        noise_imgs[i*3+2, 1, :, :, :] = isi
        noise_imgs[i*3+2, 2, :, :, :] = test + img
        noise_pattern.append('different')

    # save 
    torch.save(noise_imgs, dir+'Documents/code/DNN_adaptation/stimuli/noiseMNIST/' + data + '/imgs')
    torch.save(noise_lbls, dir+'Documents/code/DNN_adaptation/stimuli/noiseMNIST/' + data + '/lbls')

    with open(dir+'Documents/code/DNN_adaptation/stimuli/noiseMNIST/' + data + '/pattern.txt', 'w') as f:
        for line in noise_pattern:
            f.write(line)
            f.write('\n')

create_stimuli('train')
create_stimuli('test')

