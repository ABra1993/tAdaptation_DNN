import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns
from scipy import stats

from scipy.stats import f_oneway
from scipy.stats import tukey_hsd

global tdyn_color
tdyn_color = ['dimgrey', '#ACD39E', '#E49C39', '#225522', '#DF4828']

def visualize_in_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 10
    fontsize_legend         = 15

    try:
        t_steps = args[0]
    except:
        pass

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, 2, figsize=(5, 2), gridspec_kw={'width_ratios': [4, 3]})#, sharey=True)

        # first timepoints above 50
        t_first = np.zeros((len(tempDynamics), init))

        for iT, current_tempDynamics in enumerate(tempDynamics):

            # if iT == 0: continue

            # select data
            data_current = accu[iD, iT, :, :]

            # compute averages
            data_mean = np.mean(data_current, 0)
            data_sem = np.std(data_current, 0)/math.sqrt(init)

            # visualize
            axs[0].plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT])
            # axs[0].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

            # compute when first time above 50%
            for iInit in range(init):
                t_first[iT, iInit] = accu[iD, iT, iInit, :].mean()

        # visualize average accuracy
        for iT, current_tempDynamics in enumerate(tempDynamics):

            if iT == 0: continue

            # compute values
            data_mean = np.mean(t_first[iT, :])
            data_sem = np.std(t_first[iT, :])/math.sqrt(init)

            # print('\n', current_tempDynamics, current_dataset)
            # print(np.round(data_mean, 5)*100)
            # print(np.round(np.std(t_first[iT, :]), 5)*100)

            # plot
            axs[1].scatter(iT, data_mean, color=tdyn_color[iT], edgecolor='white', s=60)
            # sns.stripplot(x=np.ones(init)*iT, y=t_first[iT, :], jitter=True, ax=axs[1], color='dimgrey', size=2, native_scale=True, zorder=-1)
            axs[1].plot([iT, iT], [data_mean - data_sem, data_mean + data_sem], color=tdyn_color[iT], zorder=-2)

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
        # axs[0].set_xticks(np.arange(t_steps)+1)
        axs[0].axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        axs[0].set_ylim(0, 1)

        axs[1].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].set_xlim(0, 5)
        axs[1].set_xticks([1, 2, 3, 4])
        axs[1].set_xticklabels([ ' ', ' ', ' ', ' '])

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def onset_accu(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # color onsets
    color = ['royalblue', 'deepskyblue', 'mediumturquoise']
    linestyle = ['solid', 'solid', 'solid']

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 8
    fontsize_legend         = 15

    try:
        t_steps = args[0]
        onsets = args[1]
    except:
        pass

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10.5, 1.5), sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):

            for iO, onset in enumerate(onsets):

                # select data
                data_current = accu[iD, iT, iO, :, :]

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(np.arange(t_steps)+1, data_mean, color=color[iO], lw=2, linestyle=linestyle[iO])
                axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=color[iO], alpha=0.2) # lw=lw[-iC]

                # plot onset
                axs[iT].axvline(onset, color=color[iO], linestyle='dashed', lw=1.5, alpha=0.7)

            # adjust axes
            axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[iT].spines['top'].set_visible(False)
            axs[iT].spines['right'].set_visible(False)

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def onset_activations(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # color onsets
    color = ['mediumturquoise', 'deepskyblue', 'dodgerblue']
    linestyle = ['dashed', 'dotted', 'dashdot']

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 8
    fontsize_legend         = 15

    try:
        t_steps = args[0]
        onsets = args[1]
    except:
        pass

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10.5, 1.5))#, sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):

            # plot second image
            for iO, onset in enumerate(onsets):

                # plot first image
                data_current = accu[iD, iT, iO, :, 0, :]

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(np.arange(t_steps)+1, data_mean, color='grey', lw=2, linestyle=linestyle[iO])
                axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color='grey', alpha=0.05) # lw=lw[-iC]

                # select data
                data_current = accu[iD, iT, iO, :, 1, :]

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(np.arange(t_steps)+1, data_mean, color=color[iO], lw=2, linestyle=linestyle[iO])
                axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=color[iO], alpha=0.1) # lw=lw[-iC]

                # plot onset
                axs[iT].axvline(onset+1, color=color[iO], linestyle='dashed', lw=2, alpha=0.7)

            # adjust axes
            axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[iT].spines['top'].set_visible(False)
            axs[iT].spines['right'].set_visible(False)


        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')


def intervention(task, accu, tempDynamics, dataset, init, root, *args):

    # color onsets
    color = ['mediumturquoise', 'deepskyblue', 'dodgerblue']
    linestyle = ['dashed', 'dotted', 'dashdot']

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 8
    fontsize_legend         = 15

    try:
        t_steps = args[0]
        intervent = args[1]
        stimulus = args[2]
        onset = args[3]
    except:
        pass

    # color intervention
    color = ['lightgrey', 'crimson']

    # visualize
    offset_iStim = [0, 3]
    offset_iInt = [-0.5, 0.5]
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10, 1.5))#, sharey=True)
        # fig, axs = plt.subplots(1, 2, figsize=(3, 1.5))#, sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):

            # plot second image
            for iStim, stim in enumerate(stimulus):
                for iInt, int in enumerate(intervent):

                    # plot first image
                    data_current = accu[iD, iT, :, iInt, iStim, onset:].mean(1)

                    # compute averages
                    data_mean = np.mean(data_current, 0)
                    data_sem = np.std(data_current, 0)/math.sqrt(init)

                    # visualize
                    axs[iT].bar(offset_iStim[iStim]+offset_iInt[iInt], data_mean, color=color[iInt], zorder=-2)
                    sns.stripplot(x=np.ones(init)*(offset_iStim[iStim]+offset_iInt[iInt]), y=data_current, jitter=True, ax=axs[iT], color='dimgrey', size=3, native_scale=True, zorder=-1)
                    axs[iT].plot([offset_iStim[iStim]+offset_iInt[iInt], offset_iStim[iStim]+offset_iInt[iInt]], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)

            # adjust axes
            axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[iT].spines['top'].set_visible(False)
            axs[iT].spines['right'].set_visible(False)
            axs[iT].set_xticks([offset_iStim[0], offset_iStim[1]])
            axs[iT].set_xticklabels(['First', 'Second'])
            axs[iT].set_xlim(-1.5, 4.5)

            # perform stats
            print(current_tempDynamics)
            res = stats.ttest_ind(accu[iD, iT, :, 1, 0, :].mean(1), accu[iD, iT, :, 1, 1, :].mean(1))
            if res[1] < 0.05:
                print(res)



        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/intervention_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/intervention_' + current_dataset + '.svg')