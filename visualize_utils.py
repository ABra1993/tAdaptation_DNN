import torch as torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns

def visualize_sequence(ax, imgs, lbls, classes, task, dataset, n_add=4):

    # fontsizes
    fontsize_title          = 28
    fontsize_legend         = 22
    fontsize_label          = 22
    fontsize_tick           = 20

    # number of images
    n_imgs = 3

    # settings
    fontsize_title      = 8
    fontsize_label      = 6

    if (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):

        # initiate figure
        _, axs = plt.subplots(n_imgs, ax.shape[1], figsize=(ax.shape[1], n_imgs))

        for n_img in range(n_imgs):
            for t in range(ax.shape[1]):

                # plot image
                if ax[t].shape[1] == 1:  
                    axs[n_img, t].imshow(ax[n_img, t, :, :, :].permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
                else:
                    axs[n_img, t].imshow(ax[n_img, t, :, :, :].permute(1, 2, 0), vmin=0, vmax=1)

                # plot label
                if t < n_add:
                    axs[n_img, t].set_title(classes[lbls[n_img, t]], fontsize=fontsize_title, weight='bold')
                else:
                    axs[n_img, t].set_title(classes[lbls[n_img, t]], fontsize=fontsize_title, weight='bold')

                # adjust axes
                axs[n_img, t].set_xticks([])
                axs[n_img, t].set_yticks([])
                
                # if n_img == n_imgs-1:
                #     axs[n_img, t].set_xlabel('t = ' + str(t+1), fontsize=fontsize_label)

    else:

        # initiate figure
        _, axs = plt.subplots(n_imgs, len(ax) + 1, figsize=(len(ax)+1, n_imgs))
        # print(len(ax))

        for n_img in range(n_imgs):

            # plot ground truth
            if imgs.shape[1] == 1:  
                axs[n_img, 0].imshow(imgs[-1-n_img, :, :, :].permute(1, 2, 0), cmap='gray')
            else:
                axs[n_img, 0].imshow(imgs[-1-n_img, :, :, :].permute(1, 2, 0))

            # adjust axes
            axs[n_img, 0].set_title('``' + classes[lbls[-1-n_img].item()] + '``', fontsize=fontsize_title, weight='bold')
            axs[n_img, 0].set_xticks([])
            axs[n_img, 0].set_yticks([])

            for t in range(len(ax)):

                # plot input
                if ax[t].shape[1] == 1:
                    axs[n_img, t+1].imshow(ax[t][-1-n_img, :, :, :].permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
                else:
                    axs[n_img, t+1].imshow(ax[t][-1-n_img, :, :, :].permute(1, 2, 0), vmin=0, vmax=1)

                # adjust axes
                axs[n_img, t+1].set_xticks([])
                axs[n_img, t+1].set_yticks([])
                if n_img == n_imgs-1:
                    axs[n_img, t+1].set_xlabel('t = ' + str(t+1), fontsize=fontsize_label)

    # save plot
    plt.tight_layout()
    plt.savefig('visualization/stim_examples/' + task + '_' + dataset, dpi=300)
    plt.savefig('visualization/stim_examples/' + task + '_' + dataset + '.svg')
    plt.close()

    return ax

def performance(tempDynamics, tempDynamics_lbls, Datasets, task, init, root):

    # fontsizes
    fontsize_title          = 10
    fontsize_legend         = 10
    fontsize_label          = 10
    fontsize_tick           = 10

    # initiate figure
    # fig, ax = plt.subplots(1, len(Datasets), figsize=(1, 2*len(Datasets)))
    fig, ax = plt.subplots(1, len(Datasets), figsize=(3*len(Datasets), 3))

    # settings for plotting
    sns.despine(offset=10)

    # plot settings
    barwidth = 0.5

    # mean of no temporal dynamics
    mean_accu_none = np.zeros(len(Datasets))

    # plot spread
    for iD, dataset in enumerate(Datasets):
        for iTd, tempDynamic in enumerate(tempDynamics):

            # load
            accu = np.load(root + 'accu/' + task + '/' + tempDynamic + '_' + dataset + '.npy')   
            # print(accu.shape)

            # compute values
            mean = np.mean(accu)
            if tempDynamic == 'none':
                mean_accu_none[iD] = mean
            sem = np.std(accu)/math.sqrt(init)

            # plot
            ax[iD].bar(iTd, mean, color='silver', width=barwidth)

            ax[iD].plot(iTd, mean - sem, mean + sem, color='black', zorder=1)
            # sns.stripplot(x=np.ones(init)*iTd, y=accu, jitter=True, ax=ax, color='dimgrey', size=15, alpha=1, native_scale=True)

        # adjust axes
        ax[iD].tick_params(axis='both', labelsize=fontsize_tick)
        ax[iD].spines['top'].set_visible(False)
        ax[iD].spines['right'].set_visible(False)
        ax[iD].set_title(dataset, fontsize=fontsize_title)
        ax[iD].set_xticks(np.arange(len(tempDynamics)))
        ax[iD].set_xticklabels(tempDynamics_lbls, fontsize=fontsize_label, rotation=45, ha='right')
        ax[iD].set_xlabel('Dataset', fontsize=fontsize_label)
        ax[iD].set_ylabel('Accuracy', fontsize=fontsize_label)
        ax[iD].axhline(0.1, linestyle='dashed', lw=1, color='black')
        ax[iD].set_ylim(0, 1.1)
        ax[iD].axhline(mean_accu_none[iD], lw=1, linestyle='dashed', color='red')

    # save plot
    plt.tight_layout()
    plt.savefig('visualization/performance_' + task, dpi=300)
    plt.close()