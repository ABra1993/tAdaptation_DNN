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
noise_labels = ['none', 'same', 'different']

# load test set
noise_imgs = torch.load(dir+'Documents/code/nAdaptation_DNN/stimuli/noiseMNIST/test/imgs')
noise_lbls = torch.load(dir+'Documents/code/nAdaptation_DNN/stimuli/noiseMNIST/test/lbls')

noise_patterns = []
with open(dir+'Documents/code/nAdaptation_DNN/stimuli/noiseMNIST/test/pattern.txt') as file:
    for line in file:
        noise_patterns.append(line.strip('\n'))
# print(noise_patterns)

# create test dataset
testdt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape test set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# dataloader
ldrs = {    
    'test'  : torch.utils.data.DataLoader(testdt, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=1),
}
print('\nNumber of training batches: ', len(ldrs['test']), '\n')

# example stimuli
for a, (img1, img2, img3, lbls) in enumerate(ldrs['test']):
    if a < 3:

        # initiate plot
        fig, axs = plt.subplots(1, 3, figsize=(8, 2))
        axs[1].set_title(noise_patterns[a] + ', ' + str(lbls[0]))

        # plot images
        tmp = img1[0, :, :, :]
        shape = tmp.shape
        axs[0].imshow(tmp.reshape(28, 28, 1))

        tmp = img2[0, :, :, :]
        shape = tmp.shape
        axs[1].imshow(tmp.reshape(28, 28, 1))

        tmp = img3[0, :, :, :]
        shape = tmp.shape
        axs[2].imshow(tmp.reshape(28, 28, 1))
# plt.show()

# initiate model
model = cnn_feedforward_exp_decay()
for name, param in model.named_parameters(): 
    if param.requires_grad: 
        print(name) 
        # print(param.data)
model.load_state_dict(torch.load(dir+'Documents/code/nAdaptation_DNN/weights/weights_feedforward_exp_decay.pth'))

# Test the model
accu = np.zeros(len(noise_patterns))
model.eval()
with torch.no_grad():
    for a, (img1, img2, img3, lbls) in enumerate(ldrs['test']):
        testoutp = model([img1, img2, img3])
        predicy = torch.argmax(testoutp[6], dim=1)
        if predicy == lbls:
            accu[a] = 1

# retrieve indices
idx_no = list()
idx_same = list()
idx_diff = list()
for i, noise in enumerate(noise_patterns):
    if noise == 'none':
        idx_no.append(i)
    elif noise == 'same':
        idx_same.append(i)
    else:
        idx_diff.append(i)

# print accuracies
print('No noise: ', sum(accu[idx_no])/len(idx_no))
print('Same noise: ', sum(accu[idx_same])/len(idx_same))
print('Different noise: ', sum(accu[idx_diff])/len(idx_diff))

fig = plt.figure()
plt.bar(np.arange(len(noise_labels)), [sum(accu[idx_no])/len(idx_no), sum(accu[idx_same])/len(idx_same), sum(accu[idx_diff])/len(idx_diff)], color='grey', width=0.5)
ax = plt.gca()
ax.set_xticks(np.arange(len(noise_labels)))
ax.set_xticklabels(noise_labels, rotation=45)

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

