import torch as torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn

from model_train_dataloader_novelty import PairsDataset

def sequence_test(imgs, current_analysis, contrast, n_bin=None, n_samples=None, depVar=None):

    # compute mean of test images
    mean_per_image = imgs.mean(dim=[1, 2, 3], keepdim=True) 
    mean_imgs = mean_per_image.expand(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
    # print(mean_imgs[0, :, :, :])

    # initiate sequence storage
    ax = []

    # patch
    patch_width     = 12
    patch_height    = 12

    # height and with
    height  = imgs.shape[2]
    width   = imgs.shape[3]

    # create a tensor with values [10, 11, 12] to set the height
    min          = 10
    max          = 13
    values = torch.arange(min, max, dtype=torch.float32)
    indices = values.multinomial(num_samples=imgs.shape[0], replacement=True)
    # print(values)
    # print(indices)

    # occlude images
    for iB in range(imgs.shape[0]): # iterate over batch
        for t in range(int((width - patch_width)/2+1)):
            
            # replicate img tensor to store 
            imgs_occluded = imgs.clone()

            # Apply the occlusion
            # imgs_occluded[iB, :, start_height:start_height + patch_height, start_width:start_width + patch_width] = mean_imgs[iB, :, :patch_height, :patch_width]
            if iB == 0:
                imgs_occluded[iB, :, int(values[indices[iB]].item()):int(values[indices[iB]].item())+patch_height, t*2:t*2+patch_width] = mean_imgs[iB, :, :patch_height, :patch_width]
                ax.append(imgs_occluded)
            else:
                ax[t][iB, :, int(values[indices[iB]].item()):int(values[indices[iB]].item())+patch_height, t*2:t*2+patch_width] = mean_imgs[iB, :, :patch_height, :patch_width]

    # normalize in the range [0, 1]
    for t in range(len(ax)):
        ax[t] = torch.clamp(ax[t], 0, 1)

    return ax

def create_samples(current_analysis, batch_size):

    if current_analysis == 'in_distribution':
        contrasts = [1]
        return contrasts, None, None, None
    elif current_analysis == 'activations':
        contrasts = [1]
        return contrasts, None, None, None
    elif current_analysis == 'both':
        contrasts = [1]
        return contrasts, None, None, None

    