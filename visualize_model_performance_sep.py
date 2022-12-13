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
import math

# import required script
# from models.cnn_feedforward import cnn_feedforward
# from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

# start time
startTime = time.time()

# define root
linux = True # indicates whether script is run on mac or linux
if linux:
    dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

# noise labels
noise_labels = ['No adaptation', 'Same_exp_decay', 'Same_div_norm', 'Different_exp_decay', 'Different_div_norm']
contrast = 'lcontrast'

# load accuracies with random initializations
accu_no = torch.load(dir+'accu/feedforward_same_' + contrast)
accu_no_mean = torch.mean(accu_no)*100
accu_no_std = torch.std(accu_no)/math.sqrt(len(accu_no))*100

accu_same_exp_decay = torch.load(dir+'accu/feedforward_exp_decay_same_' + contrast)
accu_same_mean_exp_decay = torch.mean(accu_same_exp_decay)*100
accu_same_std_exp_decay = torch.std(accu_same_exp_decay)/math.sqrt(len(accu_same_exp_decay))*100

accu_different_exp_decay = torch.load(dir+'accu/feedforward_exp_decay_different_' + contrast)
accu_diff_mean_exp_decay = torch.mean(accu_different_exp_decay)*100
accu_diff_std_exp_decay = torch.std(accu_different_exp_decay)/math.sqrt(len(accu_different_exp_decay))*100

accu_same_div_norm = torch.load(dir+'accu/feedforward_div_norm_same_' + contrast)
accu_same_mean_div_norm = torch.mean(accu_same_div_norm)*100
accu_same_std_div_norm = torch.std(accu_same_div_norm)/math.sqrt(len(accu_same_div_norm))*100

accu_different_div_norm = torch.load(dir+'accu/feedforward_div_norm_different_' + contrast)
accu_diff_mean_div_norm = torch.mean(accu_different_div_norm)*100
accu_diff_std_div_norm = torch.std(accu_different_div_norm)/math.sqrt(len(accu_different_div_norm))*100

mean = [accu_no_mean, accu_same_mean_exp_decay, accu_same_mean_div_norm, accu_diff_mean_exp_decay, accu_diff_mean_div_norm]
std = [accu_no_std, accu_same_std_exp_decay, accu_same_std_div_norm, accu_diff_std_exp_decay, accu_diff_std_div_norm]

# initiate figure
fig = plt.figure()

for i in range(5):
    plt.scatter(i, mean[i], color='red', zorder=1, s=80)
    plt.plot([i, i], [mean[i] - std[i], mean[i] + std[i]], color='black', zorder=-2, lw=3)
ax = plt.gca()

# adjust axis
ax.set_xticks(np.arange(len(noise_labels)))
ax.set_xticklabels(noise_labels, rotation=45, fontsize=15)
ax.set_ylabel('Accuracy (in %)', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(dir+'visualizations/test_performance_' + contrast) # + '.svg', format='svg')
plt.show()
