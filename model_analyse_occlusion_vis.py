import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns

from scipy.stats import f_oneway
from scipy.stats import tukey_hsd

global tdyn_color
tdyn_color = ['dimgrey', '#ACD39E', '#E49C39', '#225522', '#DF4828']


# set fontsizes
global fontsize_title
global fontsize_label
global fontsize_tick
global fontsize_legend

fontsize_title          = 15
fontsize_label          = 12
fontsize_tick           = 12
fontsize_legend         = 12

def visualize_in_distribution_withoutAvg(task, accu, tempDynamics, current_analysis, current_dataset, init, root, *args):

    # timesteps
    t_steps = args[0]

    # initiate figure
    fig = plt.figure(figsize=(3.25, 2))
    axs = plt.gca()

    # first timepoints above 50
    t_first = np.zeros((len(tempDynamics), init))

    for iT, current_tempDynamics in enumerate(tempDynamics):

        # select data
        data_current = accu[iT, :, 0, :]#.mean(1)

        # compute averages
        data_mean = np.mean(data_current, 0)
        data_sem = np.std(data_current, 0)/math.sqrt(init)

        # print('\n', current_tempDynamics, current_dataset)
        # print(np.round(data_mean, 5)*100)
        # print(np.round(np.std(data_current), 5)*100)

        # plot
        axs.plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT], zorder=-2) #, alpha=0.6)
        # sns.stripplot(x=np.ones(init)*(iT), y=data_current, jitter=True, ax=ax, color='dimgrey', size=3, native_scale=True, zorder=-1)
        # axs[0].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, facecolor=tdyn_color[iT], edgecolor='white', zorder=0, alpha=0.2)

        # compute when first time above 50%
        for iInit in range(init):
            t_first[iT, iInit] = accu[iT, iInit, 0, :].mean()
            # first = False
            # for t in range(t_steps):
            #     if (data_current[iInit, t] > 0.7) & (first == False):
            #         first=True
            #         t_first[iT, iInit] = t
            #         continue

    # adjust axes
    axs.tick_params(axis='both', which='major', labelsize=fontsize_tick)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.axhline(0.1, linestyle='dashed', lw=1.5, color='black')
    if current_dataset == 'mnist':
        axs.set_yticks([0.0, 0.5, 1])
    if current_dataset == 'fmnist':
        axs.set_yticks([0.0, 0.4, 0.8])
    if current_dataset == 'cifar':
        axs.set_yticks([0.0, 0.3, 0.6])
    axs.set_xticks([1, 4, 7, 10])

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_withoutAvg_' + current_dataset)
    plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_withougAvg_' + current_dataset + '.svg')

def visualize_in_distribution(task, accu, tempDynamics, current_analysis, current_dataset, init, root, *args):

    # timesteps
    t_steps = args[0]

    # initiate figure
    fig, axs = plt.subplots(1, 2, figsize=(3.25, 2), gridspec_kw={'width_ratios': [3, 2]})#, sharey=True)

    # first timepoints above 50
    t_first = np.zeros((len(tempDynamics), init))

    for iT, current_tempDynamics in enumerate(tempDynamics):

        # select data
        data_current = accu[iT, :, 0, :]#.mean(1)

        # compute averages
        data_mean = np.mean(data_current, 0)
        data_sem = np.std(data_current, 0)/math.sqrt(init)

        # print('\n', current_tempDynamics, current_dataset)
        # print(np.round(data_mean, 5)*100)
        # print(np.round(np.std(data_current), 5)*100)

        # plot
        axs[0].plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT], zorder=-2) #, alpha=0.6)
        # sns.stripplot(x=np.ones(init)*(iT), y=data_current, jitter=True, ax=ax, color='dimgrey', size=3, native_scale=True, zorder=-1)
        # axs[0].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, facecolor=tdyn_color[iT], edgecolor='white', zorder=0, alpha=0.2)

        # compute when first time above 50%
        for iInit in range(init):
            t_first[iT, iInit] = accu[iT, iInit, 0, :].mean()
            # first = False
            # for t in range(t_steps):
            #     if (data_current[iInit, t] > 0.7) & (first == False):
            #         first=True
            #         t_first[iT, iInit] = t
            #         continue

    # visualize
    for iT, current_tempDynamics in enumerate(tempDynamics):

        # select
        data_current = t_first[iT, :]

        # compute
        data_mean = np.mean(data_current)
        data_sem = np.std(data_current)/math.sqrt(init)

        print('\n', current_tempDynamics, current_dataset)
        print(np.round(data_mean, 5)*100)
        print(np.round(np.std(data_current), 5)*100)

        # plot
        axs[1].scatter(iT, data_mean, facecolor=tdyn_color[iT], edgecolor='white', s=60)
        axs[1].plot([iT, iT], [data_mean - data_sem, data_mean + data_sem], color=tdyn_color[iT], zorder=-2)
        # axs[1].scatter(np.ones(init)*iT, t_first[iT, :], color='dimgrey', s=2)

    # perform stats
    print('\nDataset: ', current_dataset)
    res = f_oneway(t_first[0, :], t_first[1, :], t_first[2, :], t_first[3, :], t_first[4, :])
    print(res)
    if res[1] < 0.05:
        res = tukey_hsd(t_first[0, :], t_first[1, :], t_first[2, :], t_first[3, :], t_first[4, :])
        print(res)

    # adjust axes
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[0].axhline(0.1, linestyle='dashed', lw=1.5, color='black')
    if current_dataset == 'mnist':
        axs[0].set_yticks([0.0, 0.5, 1])
    if current_dataset == 'fmnist':
        axs[0].set_yticks([0.0, 0.4, 0.8])
    if current_dataset == 'cifar':
        axs[0].set_yticks([0.0, 0.3, 0.6])
    axs[0].set_xticks([1, 4, 7, 10])

    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].set_xticklabels([' ', ' ', ' ', ' ', ' '])
    axs[1].set_xlim(-1, len(tempDynamics))

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
    plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_activations(task, accu, tempDynamics, current_analysis, current_dataset, init, root, *args):

    # timesteps
    t_steps = args[0]

    # fontsize
    fontsize_title          = 15
    fontsize_label          = 12
    fontsize_tick           = 12
    fontsize_legend         = 12

    try:
        layers = args[0]
        t_steps = args[1]
        contrasts = args[2]
    except:
        pass

    # layer
    layer_idx = 0

    # initiate figure
    fig = plt.figure(figsize=(2.5, 1.2))
    ax = plt.gca()

    for iT, current_tempDynamics in enumerate(tempDynamics):

        # select data
        data_current = accu[iT, :, 0, layer_idx, :]#/np.max(accu[iT, :, 0, layer_idx, :], 1)

        # compute averages
        data_mean = np.mean(data_current, 0)
        data_sem = np.std(data_current, 0)/math.sqrt(init)

        # visualize
        ax.plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT])#, alpha=0.4+iC*0.1) # lw=lw[-iC]
        # axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

        # adjust axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.arange(t_steps)+1)
        if current_dataset == 'mnist':
            ax.set_yticks([0.0, 0.1, 0.2])
        # if current_dataset == 'fmnist':
        #     ax.set_yticks([0.0, 0.4, 0.8])
        # if current_dataset == 'cifar':
        #     ax.set_yticks([0.0, 0.3, 0.6])
        ax.set_xticks([1, 4, 7, 10])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_both(task, tempDynamics, init, root):


    # define 
    analysis = ['in_distribution', 'activations']
    dataset = ['mnist', 'fmnist', 'cifar']

    # fontsize
    fontsize_title          = 15
    fontsize_label          = 12
    fontsize_tick           = 12
    fontsize_legend         = 12

    # layer
    layer_idx = 0

    # initiate figure
    fig, axs = plt.subplots(len(dataset), len(analysis), figsize=(4.5, 3.5))

    for iA, current_analysis in enumerate(analysis):
            for iD, current_dataset in enumerate(dataset):

                # define number of timesteps
                if current_dataset == 'cifar':
                    t_steps = 10
                else:
                    t_steps = 9

                # visualize
                for iT, current_tempDynamics in enumerate(tempDynamics):

                    # select data
                    data =  np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
                    
                    # select data
                    if current_analysis == 'in_distribution':
                        data_current = data.mean(1)
                    else:
                        data_current = data[:, 0, layer_idx, :]

                    # compute averages
                    data_mean = np.mean(data_current, 0)
                    data_sem = np.std(data_current, 0)/math.sqrt(init)

                    # visualize
                    axs[iD, iA].plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT])#, alpha=0.4+iC*0.1) # lw=lw[-iC]
                    axs[iD, iA].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

                    # adjust axes
                    axs[iD, iA].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                    axs[iD, iA].spines['top'].set_visible(False)
                    axs[iD, iA].spines['right'].set_visible(False)
                    axs[iD, iA].set_xticks(np.arange(t_steps)+1)
                    axs[iD, iA].set_xticks([1, 4, 7, 10])
                    if iD < len(dataset) - 1:
                        axs[iD, iA].set_xticklabels([' ', ' ', ' ', ' '])


    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/' + task + '/both')
    plt.savefig(root + 'visualization/' + task + '/both.svg')