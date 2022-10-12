# %%

import torch
import re
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import numpy as np
from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

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

# noise labels
noise_labels = ['No adaptation', 'Same', 'Different']
contrast = 'lcontrast'

# load accuracies with random initializations
accu_no = torch.load(dir+'Documents/code/nAdaptation_DNN/accu/feedforward_same_' + contrast)
accu_no_mean = torch.mean(accu_no)*100
accu_no_std = torch.std(accu_no)/math.sqrt(len(accu_no))*100

accu_same = torch.load(dir+'Documents/code/nAdaptation_DNN/accu/feedforward_exp_decay_same_' + contrast)
accu_same_mean = torch.mean(accu_same)*100
accu_same_std = torch.std(accu_same)/math.sqrt(len(accu_same))*100

accu_different = torch.load(dir+'Documents/code/nAdaptation_DNN/accu/feedforward_exp_decay_different_' + contrast)
accu_diff_mean = torch.mean(accu_different)*100
accu_diff_std = torch.std(accu_different)/math.sqrt(len(accu_different))*100

mean = [accu_no_mean, accu_same_mean, accu_diff_mean]
std = [accu_no_std, accu_same_std, accu_diff_std]

# initiate figure
fig = plt.figure()

for i in range(3):
    plt.scatter(i, mean[i], color='red', zorder=1, s=80)
    plt.plot([i, i], [mean[i] - std[i], mean[i] + std[i]], color='black', zorder=-2, lw=3)
ax = plt.gca()

# adjust axis
ax.set_xticks(np.arange(len(noise_labels)))
ax.set_xticklabels(noise_labels, rotation=45, fontsize=15)
ax.set_ylabel('Accuracy (in %)', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


plt.savefig(dir+'Documents/code/nAdaptation_DNN/visualizations/test_performance_' + contrast) # + '.svg', format='svg')
plt.show()
