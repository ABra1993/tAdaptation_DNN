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

    # set standard deviations
    std_normal = 0.2
    std_uniform = 0.2

    # create noise
    if (current_analysis == 'out_distribution_noise'):
        noise = torch.rand_like(imgs) - std_uniform
    else:
        noise = torch.randn_like(imgs)

    # create OOD
    if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
        noise = noise * std_normal

    if (current_analysis == 'out_distribution_std'):
        for i in range(n_bin):
            noise[i*n_samples:(i+1)*n_samples, :, :, :] = noise[i*n_samples:(i+1)*n_samples, :, :, :] * depVar[i]

    elif (current_analysis == 'out_distribution_offset'):
        for i in range(n_bin):
           noise[i*n_samples:(i+1)*n_samples, :, :, :] = noise[i*n_samples:(i+1)*n_samples, :, :, :] * std_normal + depVar[i]

    elif (current_analysis == 'out_distribution_noise'):
        for i in range(n_bin):
            noise[i*n_samples:(i+1)*n_samples, :, :, :] = noise[i*n_samples:(i+1)*n_samples, :, :, :] * depVar[i]

    elif (current_analysis == 'out_distribution_shift'):
        noise = noise * std_normal
        noise_test = noise.clone()
        for i in range(n_bin):
            noise_test[i*n_samples:(i+1)*n_samples, :, :, :] = torch.roll(noise[i*n_samples:(i+1)*n_samples, :, :, :], shifts=(-depVar[i], 0), dims=(2, 3))
    
    elif (current_analysis == 'out_distribution_different'):
        noise = noise * std_normal
        noise_test =  torch.randn_like(imgs) * std_normal

    # number of timesteps
    n_timestep = 3

    # compute mean of test images
    mean_imgs = torch.mean(imgs, dim=[2, 3], keepdim=True).expand(imgs.shape)
    # print(mean_imgs.shape)

    # adjust contrast
    imgs_adjusted = (imgs - mean_imgs) * contrast # + mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]

    # add mean to noise
    noise_adjusted = (noise + mean_imgs) #* contrast + mean_imgs
    if (current_analysis == 'out_distribution_shift') | (current_analysis == 'out_distribution_different'):
        noise_test_adjusted = (noise_test + mean_imgs) #* contrast + mean_imgs

    # create sequence
    for t in range(n_timestep):
        if t == 0:
            ax.append(noise_adjusted)#.clamp(0, 1))

        elif t == 1:
            blank = torch.ones_like(imgs) * mean_imgs
            ax.append(blank)

        elif t == 2:

            # create test
            if (current_analysis == 'out_distribution_shift'):
                test = imgs_adjusted + noise_test_adjusted

            elif (current_analysis == 'out_distribution_different'):

                # test = imgs_adjusted + noise_test_adjusted
                test = imgs_adjusted + noise_adjusted
                test[int(imgs.shape[0]/2):, :, :, :] = imgs_adjusted[int(imgs.shape[0]/2):, :, :, :] + noise_test_adjusted[int(imgs.shape[0]/2):, :, : , :]
            
            else:
                test = imgs_adjusted + noise_adjusted

            # clamp between 0 and 1
            ax.append(test)#.clamp(0, 1))

    return ax


def create_samples(current_analysis, batch_size):

    if current_analysis == 'in_distribution':
        contrasts = [0.2, 1]
        return contrasts, None, None, None
    elif current_analysis == 'out_distribution_contrast':
        contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        return contrasts, None, None, None
    elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise'):
        contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        n_bin = 8
        depVar = torch.linspace(0.01, 1, n_bin)
        n_samples = int(batch_size/(n_bin))
        return contrasts, n_bin, depVar, n_samples
    elif (current_analysis == 'out_distribution_shift'):
        # contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        contrasts = [0.2, 1]
        n_bin = 8
        depVar = torch.arange(0, n_bin)
        n_samples = int(batch_size/(n_bin))
        return contrasts, n_bin, depVar, n_samples
    elif (current_analysis == 'out_distribution_different'):
        contrasts = [0.2, 0.4, 0.6, 0.8, 1]
        return contrasts, None, None, None

    