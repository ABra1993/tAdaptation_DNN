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
# noise_labels = ['No adaptation - same', 'No adaptation - different', 'Exp. decay - same',  'Exp. decay - different', 'Div. norm. - same', 'Div. norm. - different']
noise_labels = ['No adaptation', 'Exp. decay', 'Div. norm.']

contrast = 'lcontrast'

# load accuracies with random initializations
accu_same_no = torch.load(dir+'accu/feedforward_same_' + contrast)
accu_same_no_mean = torch.mean(accu_same_no)*100
accu_same_no_std = torch.std(accu_same_no)/math.sqrt(len(accu_same_no))*100

accu_different_no = torch.load(dir+'accu/feedforward_different_' + contrast)
accu_different_no_mean = torch.mean(accu_different_no)*100
accu_different_no_std = torch.std(accu_different_no)/math.sqrt(len(accu_different_no))*100

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

mean = [accu_same_no_mean, accu_different_no_mean, accu_same_mean_exp_decay,accu_diff_mean_exp_decay, accu_same_mean_div_norm, accu_diff_mean_div_norm]
std = [accu_same_no_std, accu_different_no_std, accu_same_std_exp_decay, accu_same_std_div_norm, accu_diff_std_exp_decay, accu_diff_std_div_norm]

# set alphas
alpha_low = 0.5
alpha = [1, alpha_low, 1, alpha_low, 1, alpha_low]
color = ['dodgerblue', 'darkorange', 'dodgerblue', 'darkorange', 'dodgerblue', 'darkorange']
x = [0, 1, 4, 5, 8, 9]

# initiate figure
fig = plt.figure()

plt.scatter(x, mean, color=color, s=120)
for i in range(len(mean)):
    plt.plot([x[i], x[i]], [mean[i] - std[i], mean[i] + std[i]], color='black', zorder=1, lw=3)
   
ax = plt.gca()

# adjust axis
ax.set_xticks([0.5, 4.5, 8.5])
ax.set_xticklabels(noise_labels, rotation=45, fontsize=15)
ax.set_ylabel('Accuracy (in %)', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(dir+'visualizations/test_performance_' + contrast +'.svg', format='svg') # + '.svg', format='svg')
plt.savefig(dir+'visualizations/test_performance_' + contrast)
# plt.show()
