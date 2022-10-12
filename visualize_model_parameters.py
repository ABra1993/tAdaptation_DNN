# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
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
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

# start time
startTime = time.time()

# define root
linux = True # indicates whether script is run on mac or linux
if linux:
    dir = '/home/amber/ownCloud/'
else:
    dir = '/Users/a.m.brandsuva.nl/surfdrive/'

# noise pattern
noise_patterns = ['same', 'different']
contrast = 'lcontrast'

# plot alpha and beta values
fig = plt.figure()
ax = plt.gca()

# layer names
param_alpha = ['sconv1.alpha', 'sconv2.alpha', 'sconv3.alpha', 'sfc1.alpha']
param_alpha_value = np.zeros(len(param_alpha))
param_beta = ['sconv1.beta','sconv2.beta', 'sconv3.beta', 'sfc1.beta']
param_beta_value = np.zeros(len(param_beta))

# layers
layers = ['conv1', 'conv2', 'conv3', 'fc']

linestyle = ['solid', 'dotted']
marker = ['o', '^']
color = ['navy', 'deeppink']
for i, noise in enumerate(noise_patterns):

    # initiate model
    model = cnn_feedforward_exp_decay()
    model.load_state_dict(torch.load(dir+'Documents/code/nAdaptation_DNN/weights/weights_feedforward_exp_decay_' + noise + '_' + contrast + '.pth'))

    print("Model's state_dict:")
    params = []
    for param_tensor in enumerate(model.state_dict()):
        params.append(param_tensor[1])
    print(params)

    for idx, p in enumerate(model.parameters()):
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        if (params[idx] in param_alpha):
            idx = param_alpha.index(params[idx])
            param_alpha_value[idx] = torch.mean(p.data)
        if (params[idx] in param_beta):
            idx = param_beta.index(params[idx])
            param_beta_value[idx] = torch.mean(p.data)

    # plot parameter values
    plt.plot(np.arange(len(param_alpha)), param_alpha_value, marker=marker[i], linestyle=linestyle[i], label=noise + r', $\alpha$', color='navy', markeredgecolor='navy', markerfacecolor='white', markersize=10)
    plt.plot(np.arange(len(param_beta)), param_beta_value, marker=marker[i], linestyle=linestyle[i], label= noise + r', $\beta$', markeredgecolor='deeppink', markerfacecolor='white', color='deeppink', markersize=10)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(np.arange(len(layers)))
ax.set_ylabel('Parameter value')
ax.set_xticklabels(layers, rotation=45)
plt.legend(frameon=False, ncol=2)

plt.savefig('visualizations/parameters_' + contrast)
plt.show()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('\nExecution time in seconds: ' + str(executionTime))
