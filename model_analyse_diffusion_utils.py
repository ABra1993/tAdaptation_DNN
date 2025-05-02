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

    # initiate input
    ax = []

    # number of timesteps
    n_timestep = 10

    # compute mean of test images
    mean_imgs = torch.mean(imgs, dim=[2, 3], keepdim=True).expand(imgs.shape)

    # adjust contrast
    imgs_adjusted = (imgs - mean_imgs) * contrast # + mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]

    # create weights
    weights = torch.linspace(0, 1, n_timestep)

    # create noise pattern
    noise = torch.randn_like(imgs)*.5

    for t in range(n_timestep):
        test = torch.multiply(1 - weights[t], mean_imgs + noise) + torch.multiply(weights[t], imgs_adjusted)
        test = mean_imgs + (test - mean_imgs)
        ax.append(test.clamp(0, 1))

    return ax


def create_samples(current_analysis, batch_size):

    if current_analysis == 'in_distribution':
        contrasts = [1]
        return contrasts, None, None, None
    elif current_analysis == 'out_distribution_contrast':
        contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        return contrasts, None, None, None
    elif current_analysis == 'decoding':
        contrasts = [1]
        return contrasts, None, None, None
    elif current_analysis == 'activations':
        contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        return contrasts, None, None, None


    