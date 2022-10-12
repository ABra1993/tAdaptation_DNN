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
import torchvision.transforms.functional as F

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.functions import *

# define root
linux = True # indicates whether script is run on mac or linux
if linux:
    dir = '/home/amber/ownCloud/'
else:
    dir = '/Users/a.m.brandsuva.nl/surfdrive/'

# set contrast
# contrast = 'lcontrast'

# t_steps = 30
# dur = 8
# start = [5, 18]

t_steps = 3
dur = 1
start = [0, 2]

def create_stimuli(data, noise, contrast):

    # encode timtesteps
    t_steps_label = torch.zeros(t_steps)
    t_steps_label[start[0]:start[0]+dur] = 1
    if noise == 'same':
        t_steps_label[start[1]:start[1]+dur] = 2
    elif noise == 'different':
        t_steps_label[start[1]:start[1]+dur] = 3

    # download data
    if data == 'train':
        dt, _, data_n, _ = download_data()
    elif data == 'test':
        _, dt, _, data_n = download_data()
    print(data, ': ', data_n)

    # define input shape
    input_shape = dt[0][0].shape
    w = input_shape[1]
    h = input_shape[2]
    c = input_shape[0]
    print('Input shape: ', input_shape) 

    # grayscale image
    x = torch.ones(input_shape)
    isi = torch.multiply(x, 0.5)

    # dataset settings
    img_num = data_n
    # img_num = 1

    noise_imgs = torch.empty([img_num, t_steps, c, w, h])
    noise_lbls = torch.empty(img_num, dtype=torch.long)

    for i in range(img_num):

        # create noise pattern
        adapter = torch.rand(input_shape)

        for t in range(t_steps):

            # select img
            img = dt[i][0]
            if contrast == 'lcontrast':
                img = F.adjust_contrast(img, 0.5) # lower contrast
            noise_lbls[i] = dt[i][1]

            # assign stimuli to current timestep
            if t_steps_label[t] == 0:           
                noise_imgs[i, t, :, :, :] = isi
            elif t_steps_label[t] == 1:                 # first stimulus
                noise_imgs[i, t, :, :, :] = adapter
            elif t_steps_label[t] == 2:                 # same
                noise_imgs[i, t, :, :, :] = adapter + img
            elif t_steps_label[t] == 3:                 # different
                adapter2 = torch.rand(input_shape)
                noise_imgs[i, t, :, :, :]  = adapter2 + img

        # print progress
        if (i+1)%500 == 0:
            print('Created all timesteps for image: ', i+1)

    # save 
    torch.save(noise_imgs, dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + data + '_imgs_' + contrast)
    torch.save(noise_lbls, dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + data + '_lbls_' + contrast)

create_stimuli('train', 'same', 'lcontrast')
create_stimuli('test', 'same', 'lcontrast')

create_stimuli('train', 'different', 'lcontrast')
create_stimuli('test', 'different', 'lcontrast')

create_stimuli('train', 'same', 'hcontrast')
create_stimuli('test', 'same', 'hcontrast')

create_stimuli('train', 'different', 'hcontrast')
create_stimuli('test', 'different', 'hcontrast')


