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
param = 'fixed'

# load accuracies with random initializations
accu_same_no = torch.load(dir+'accu/feedforward_same_' + contrast)
accu_same_no_mean = torch.mean(accu_same_no)*100
accu_same_no_std = torch.std(accu_same_no)/math.sqrt(len(accu_same_no))*100

accu_different_no = torch.load(dir+'accu/feedforward_different_' + contrast)
accu_different_no_mean = torch.mean(accu_different_no)*100
accu_different_no_std = torch.std(accu_different_no)/math.sqrt(len(accu_different_no))*100

accu_same_exp_decay = torch.load(dir+'accu/feedforward_exp_decay_same_' + contrast + '_' + param)
accu_same_mean_exp_decay = torch.mean(accu_same_exp_decay)*100
accu_same_std_exp_decay = torch.std(accu_same_exp_decay)/math.sqrt(len(accu_same_exp_decay))*100

accu_different_exp_decay = torch.load(dir+'accu/feedforward_exp_decay_different_' + contrast + '_' + param)
accu_diff_mean_exp_decay = torch.mean(accu_different_exp_decay)*100
accu_diff_std_exp_decay = torch.std(accu_different_exp_decay)/math.sqrt(len(accu_different_exp_decay))*100

accu_same_div_norm = torch.load(dir+'accu/feedforward_div_norm_same_' + contrast + '_' + param)
accu_same_mean_div_norm = torch.mean(accu_same_div_norm)*100
accu_same_std_div_norm = torch.std(accu_same_div_norm)/math.sqrt(len(accu_same_div_norm))*100

accu_different_div_norm = torch.load(dir+'accu/feedforward_div_norm_different_' + contrast + '_' + param)
accu_diff_mean_div_norm = torch.mean(accu_different_div_norm)*100
accu_diff_std_div_norm = torch.std(accu_different_div_norm)/math.sqrt(len(accu_different_div_norm))*100

points = [accu_same_no, accu_different_no, accu_same_exp_decay, accu_different_exp_decay, accu_same_div_norm, accu_different_div_norm]
mean = [accu_same_no_mean, accu_different_no_mean, accu_same_mean_exp_decay,accu_diff_mean_exp_decay, accu_same_mean_div_norm, accu_diff_mean_div_norm]
# std = [accu_same_no_std, accu_different_no_std, accu_same_std_exp_decay, accu_diff_std_exp_decay, accu_same_std_div_norm, accu_diff_std_div_norm]

# set alphas
alpha_low = 0.5
alpha = [1, 1, alpha_low, 1, alpha_low]
color = ['dodgerblue', 'darkorange', 'dodgerblue', 'darkorange', 'dodgerblue', 'darkorange']
x = [0, 1, 3, 4, 6, 7]

# initiate figure
fig = plt.figure()
ax = plt.gca()

# plot data
ax.scatter(x, mean, edgecolor=color, color='white', s=100)
for i in range(len(points)):
    print(points[i])
    ax.scatter(np.ones(len(points[i]))*x[i], points[i]*100, color=color[i], s=50, alpha=0.2, zorder=-1)

# adjust axis
ax.set_xlim(-0.5, 7.5)
ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels(noise_labels, rotation=45, fontsize=15)
# ax.set_ylim(59, 75)
ax.set_ylabel('Accuracy (in %)', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(dir+'visualizations/test_performance_' + contrast +'.svg', format='svg') # + '.svg', format='svg')
plt.savefig(dir+'visualizations/test_performance_' + contrast, dpi=300)
plt.show()
