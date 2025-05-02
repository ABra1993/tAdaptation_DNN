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

fontsize_title          = 20
fontsize_label          = 15
fontsize_tick           = 15
fontsize_legend         = 15

def visualize_in_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root):

    # set color
    contrast_lbl = ['20%', '40%']
    contrast_color = ['dodgerblue', 'crimson']

    # visualize
    offset_iC = 6
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig = plt.figure(figsize=(3, 2.5))
        ax = plt.gca()

        for iT, current_tempDynamics in enumerate(tempDynamics):
            for iC in range(len(contrast_color)):

                # select data
                data_current = accu[iD, iT, :, iC]

                # compute averages
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(init)
                print('\n', current_tempDynamics, current_dataset, iC+1)
                print(np.round(data_mean, 5)*100)
                print(np.round(np.std(data_current), 5)*100)

                # plot
                ax.bar(iC*offset_iC+iT, data_mean, color=tdyn_color[iT], zorder=-2) #, alpha=0.6)
                sns.stripplot(x=np.ones(init)*(iC*offset_iC+iT), y=data_current, jitter=True, ax=ax, color='dimgrey', size=3, native_scale=True, zorder=-1)
                ax.plot([iC*offset_iC+iT, iC*offset_iC+iT], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)

        # adjust axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        ax.set_xticks([])
        # ax.set_ylabel('Accuracy (%)', fontsize=fontsize_label)
        # ax.set_xticks([2, 8])
        # ax.set_xticklabels([20, 100])
        # ax.set_xlabel('Contrast (%)', fontsize=fontsize_label)

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

        # # perform stats
        # print('\nDataset (low contrast): ', current_dataset)
        # res = f_oneway(accu[iD, 0, :, 0], accu[iD, 1, :, 0], accu[iD, 2, :, 0], accu[iD, 3, :, 0], accu[iD, 4, :, 0])
        # print(res)
        # res = tukey_hsd(accu[iD, 0, :, 0], accu[iD, 1, :, 0], accu[iD, 2, :, 0], accu[iD, 3, :, 0], accu[iD, 4, :, 0])
        # print(res)

        # print('\nDataset (high contrast): ', current_dataset)
        # res = f_oneway(accu[iD, 0, :, 1], accu[iD, 1, :, 1], accu[iD, 2, :, 1], accu[iD, 3, :, 1], accu[iD, 4, :, 1])
        # print(res)
        # res = tukey_hsd(accu[iD, 0, :, 1], accu[iD, 1, :, 1], accu[iD, 2, :, 1], accu[iD, 3, :, 1], accu[iD, 4, :, 1])
        # print(res)

    
def visualize_out_distribution_contrast(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # extract contrasts
    contrasts = args[0]

    # set color
    contrast_color = ['lightgrey', 'khaki', 'gold', 'orange',  'dimgrey']

    # visualize
    offset = 6
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig = plt.figure(figsize=(4, 2.5))
        ax = plt.gca()

        for iT, current_tempDynamics in enumerate(tempDynamics):
            for iC in range(len(contrasts)):

                # select data
                data_current = accu[iD, iT, :, iC]

                # compute averages
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(init)

                # plot
                ax.bar(iT*offset+iC, data_mean, color=contrast_color[iC], zorder=-2)
                sns.stripplot(x=np.ones(init)*(iT*offset+iC), y=data_current, jitter=True, ax=ax, color='dimgrey', size=3, native_scale=True, zorder=-1)
                ax.plot([iT*offset+iC, iT*offset+iC], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)

        # adjust axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        # ax.set_ylabel('Accuracy (%)', fontsize=fontsize_label)
        ax.set_xticks([])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_out_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 10
    fontsize_legend         = 15

    try:
        contrasts = args[0]
        depVar = args[1]
    except:
        pass

    # set color
    # lw = np.ones(len(contrasts))

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics), figsize=(5, 1.25), sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):
                
            for iC in range(len(contrasts)):

                # select data
                data_current = accu[iD, iT, :, iC, :]

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(depVar, data_mean, color=tdyn_color[iT], alpha=1-0.15*iC) # lw=lw[-iC]

            # adjust axes
            axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[iT].spines['top'].set_visible(False)
            axs[iT].spines['right'].set_visible(False)
            # if iT == 0:
            #     axs[iT].set_ylabel('Accuracy (%)', fontsize=fontsize_label)
            # if current_analysis == 'out_distribution_std':
            #     # axs[iT].set_title('Gaussian', fontsize=fontsize_title)
            #     axs[iT].set_xlabel('Noise SD', fontsize=fontsize_label)
            # elif current_analysis == 'out_distribution_offset':
            #     # axs[iT].set_title('Gaussian + offset', fontsize=fontsize_title)
            #     axs[iT].set_xlabel('Offset', fontsize=fontsize_label)
            # elif current_analysis == 'out_distribution_noise':
            #     # axs[iT].set_title('Uniform', fontsize=fontsize_title)
            #     axs[iT].set_xlabel('Noise SD', fontsize=fontsize_label)

            # axs[iT].axhline(0.1, linestyle='dashed', lw=2, color='dimgrey')
            if (current_analysis == 'out_distribution_std'):
                axs[iT].axvline(0.2, linestyle='dashed', lw=2, color='lightgrey')
            axs[iT].set_xticks([0, 0.5, 1])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_out_distribution_shift(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # extract contrasts
    contrasts = args[0]
    adapters = args[1]

    # set color
    # contrast_color = ['lightgrey', 'khaki', 'gold', 'orange',  'dimgrey']
    # contrast_color = ['lightgrey', 'dimgrey']

    cmap = plt.get_cmap('GnBu')
    colors_low = [cmap(i) for i in np.linspace(0.2, 0.65, accu.shape[-1])]

    cmap = plt.get_cmap('GnBu')
    colors_high = [cmap(i) for i in np.linspace(0.65, 1, accu.shape[-1])]

    contrast_color = [colors_low, colors_high]

    # clone accu to normalize
    accu_norm = np.copy(accu)

    # visualize
    offset = 10
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig = plt.figure(figsize=(4.5, 2))
        ax = plt.gca()

        for iT, current_tempDynamics in enumerate(tempDynamics):
            for iC in range(len(contrasts)):
                for idepVar in range(accu.shape[-1]): # adapters

                    # select data
                    data_current = accu[iD, iT, :, -1-iC, idepVar]/accu[iD, iT, :, -1, 0]
                    accu_norm[iD, iT, :, -1-iC, idepVar] = data_current

                    # compute averages
                    data_mean = np.mean(data_current)
                    data_sem = np.std(data_current)/math.sqrt(init)

                    # plot
                    ax.bar(iT*offset+idepVar, data_mean, facecolor=contrast_color[-1-iC][idepVar], edgecolor=contrast_color[-1-iC][idepVar], zorder=-2)
                    sns.stripplot(x=np.ones(init)*(iT*offset+idepVar), y=data_current, jitter=True, ax=ax, color='grey', size=2, native_scale=True, zorder=-1)
                    ax.plot([iT*offset+idepVar, iT*offset+idepVar], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)

        # adjust axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        ax.set_xticks([])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

        # # perform stats
        # print('\nDataset (low contrast): ', current_dataset)
        # res = f_oneway(accu_norm[iD, 0, :, 0, :].mean(1), accu_norm[iD, 1, :, 0, :].mean(1), accu_norm[iD, 2, :, 0, :].mean(1), accu_norm[iD, 3, :, 0, :].mean(1), accu_norm[iD, 4, :, 0, :].mean(1))
        # print(res)
        # res = tukey_hsd(accu_norm[iD, 0, :, 0, :].mean(1), accu_norm[iD, 1, :, 0, :].mean(1), accu_norm[iD, 2, :, 0, :].mean(1), accu_norm[iD, 3, :, 0, :].mean(1), accu_norm[iD, 4, :, 0, :].mean(1))
        # print(res)

        # print('\nDataset (high contrast): ', current_dataset)
        # res = f_oneway(accu_norm[iD, 0, :, 1, :].mean(1), accu_norm[iD, 1, :, 1, :].mean(1), accu_norm[iD, 2, :, 1, :].mean(1), accu_norm[iD, 3, :, 1, :].mean(1), accu_norm[iD, 4, :, 1, :].mean(1))
        # print(res)
        # res = tukey_hsd(accu_norm[iD, 0, :, 1, :].mean(1), accu_norm[iD, 1, :, 1, :].mean(1), accu_norm[iD, 2, :, 1, :].mean(1), accu_norm[iD, 3, :, 1, :].mean(1), accu_norm[iD, 4, :, 1, :].mean(1))
        # print(res)

    # print drop for CIFAR-10
    for itDyn, current_tempDynamics in enumerate(tempDynamics):

        if itDyn == 0:
            continue

        for iC in range(2):

            # select data
            data_current1 = accu_norm[2, itDyn, :, iC, 0]
            data_current2 = accu_norm[2, itDyn, :, iC, 1:].mean(1)
            data_current = 1 - data_current2/data_current1

            # compute mean
            data_mean = np.mean(data_current)

            # print
            print('\n', current_tempDynamics, current_dataset, iC+1)
            print(np.round(data_mean, 2)*100)
            # print(np.round(np.std(data_current), 5)*100)

def visualize_out_distribution_different(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # extract contrasts
    contrasts = args[0]
    adapters = args[1]

    # set color
    # adapters_color = ['dodgerblue', np.array([212, 170, 0])/255]

    cmap = plt.get_cmap('Greys')
    colors_same = [cmap(i) for i in np.linspace(0.3, 0.7, len(contrasts))]

    cmap = plt.get_cmap('YlOrBr')
    colors_different = [cmap(i) for i in np.linspace(0.3, 1, len(contrasts))]

    adapters_color = [colors_same, colors_different]

    # clone accu to normalize
    accu_norm = np.copy(accu)

    # visualize
    offset = 6
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig = plt.figure(figsize=(4.5, 2))
        ax = plt.gca()

        for iT, current_tempDynamics in enumerate(tempDynamics):
            for iC in range(len(contrasts)):
                for iA in range(len(adapters)):

                    # select data
                    data_current = accu[iD, iT, :, iC, iA]/accu[iD, iT, :, -1, 0]
                    accu_norm[iD, iT, :, iC, iA] = data_current

                    # compute averages
                    data_mean = np.mean(data_current)
                    data_sem = np.std(data_current)/math.sqrt(init)

                    # plot
                    ax.bar(iT*offset+iC, data_mean, color=adapters_color[iA][iC], zorder=-2)#, alpha=0.5)
                    sns.stripplot(x=np.ones(init)*(iT*offset+iC), y=data_current, jitter=True, ax=ax, color='dimgrey', size=1.5, native_scale=True, zorder=-1)
                    ax.plot([iT*offset+iC, iT*offset+iC], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)

        # adjust axes
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        ax.set_xticks([])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

        # perform stats
        print('\nDataset (low contrast): ', current_dataset)
        res = f_oneway(accu_norm[iD, 0, :, :, 1].mean(1), accu_norm[iD, 1, :, :, 1].mean(1), accu_norm[iD, 2, :, :, 1].mean(1), accu_norm[iD, 3, :, :, 1].mean(1), accu_norm[iD, 4, :, :, 1].mean(1))
        print(res)
        res = tukey_hsd(accu_norm[iD, 0, :, :, 1].mean(1), accu_norm[iD, 1, :, :, 1].mean(1), accu_norm[iD, 2, :, :, 1].mean(1), accu_norm[iD, 3, :, :, 1].mean(1), accu_norm[iD, 4, :, :, 1].mean(1))
        print(res)