import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import seaborn as sns

from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

global tdyn_color
tdyn_color = ['dimgrey', '#ACD39E', '#E49C39', '#225522', '#DF4828']

def visualize_in_distribution(task, accu, accu_repeated_noise, tempDynamics, current_analysis, dataset, init, root, *args):

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
        fig, axs = plt.subplots(1, 2, figsize=(3.25, 2), gridspec_kw={'width_ratios': [3, 2]})#, sharey=True)

        # first timepoints above 50
        t_first = np.zeros((len(tempDynamics), init))

        for iT, current_tempDynamics in enumerate(tempDynamics):

            # plot accuracy for repeated_noise task
            data_current = accu_repeated_noise[iD, iT, :, 1]#.mean(1)

            # compute averages
            data_mean = np.mean(data_current)
            data_sem = np.std(data_current)/math.sqrt(init)

            # visualize
            # axs[1].plot([iT - 0.35, iT + 0.35], [data_mean, data_mean], color=tdyn_color[iT])
            # axs[0].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

            # select data
            data_current = accu[iD, iT, :, :, :].mean(1)

            # compute averages
            data_mean = np.mean(data_current, 0)
            data_sem = np.std(data_current, 0)/math.sqrt(init)

            # visualize
            axs[0].plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT])
            # axs[0].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

            # compute when first time above 50%
            for iInit in range(init):
                t_first[iT, iInit] = accu[iD, iT, iInit, 0, :].mean()
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
        axs[0].set_xticks(np.arange(t_steps)+1)
        axs[0].axhline(0.1, linestyle='dashed', lw=1.5, color='black')
        axs[0].set_ylim(0, 1)

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

def visualize_out_distribution_contrast(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 10
    fontsize_legend         = 15

    try:
        contrasts = args[0]
        t_steps = args[1]
    except:
        pass

    # color
    colors = plt.cm.viridis(np.linspace(0, 1, len(contrasts)))

    # visualize
    offset_iC = 7
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics)+1, figsize=(10, 2), gridspec_kw={'width_ratios': [2, 2, 2, 2, 2, 6]})

        # first timepoints above 50
        t_first = np.zeros((len(tempDynamics), len(contrasts), init))

        for iT, current_tempDynamics in enumerate(tempDynamics):
            for iC, contrast in enumerate(contrasts):

                # if iC == 4:
                #     continue

                # select data
                data_current = accu[iD, iT, :, iC, :]

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(np.arange(t_steps)+1, data_mean, color=colors[-1-iC]) #, alpha=0.3+0.15*iC, lw=2)
                # axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

                # compute when first time above 50%
                for iInit in range(init):
                    t_first[iT, iC, iInit] = accu[iD, iT, iInit, iC, :].mean()
                    # first = False
                    # for t in range(t_steps):
                    #     if (data_current[iInit, t] > 0.25) & (first == False):
                    #         first=True
                    #         t_first[iT, iC, iInit] = t
                    #         continue

                # adjust axes
                axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[iT].spines['top'].set_visible(False)
                axs[iT].spines['right'].set_visible(False)
                axs[iT].set_ylim(0, 1)
        
        # plot avaerage
        for iC in range(len(contrasts)):

            # if iC == 4:
            #     continue
            
            for iT, current_tempDynamics in enumerate(tempDynamics):

                # select data
                # data_current = accu[iD, iT, :, iC, -2]
                data_current = accu[iD, iT, :, iC, :].mean(-1)
                # data_current = t_first[iT, iC, :]

                # compute averages
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(init)

                # plot
                axs[5].bar(iC*offset_iC+iT, data_mean, color=tdyn_color[iT])
                axs[5].plot([iC*offset_iC+iT, iC*offset_iC+iT], [data_mean - data_sem, data_mean + data_sem], color='dimgrey')
                axs[5].scatter(np.ones(init)*iC*offset_iC+iT, data_current, color='silver', s=3)
                axs[5].set_xticks([])
            

            # adjust axes
            axs[5].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[5].spines['top'].set_visible(False)
            axs[5].spines['right'].set_visible(False)
            if current_dataset == 'cifar':
                axs[5].set_yticks([0, 0.1, 0.2])

            # # perform stats - ONEWAY
            # print('\nDataset: ', current_dataset, ', contrast:, ', iC)
            # res = f_oneway(t_first[0, iC, :], t_first[1, iC, :], t_first[2, iC, :], t_first[3, iC, :], t_first[4, iC, :])
            # print(res)
            # if res[1] < 0.05:
            #     res = tukey_hsd(t_first[0, iC, :], t_first[1, iC, :], t_first[2, iC, :], t_first[3, iC, :], t_first[4, iC, :])
            #     print(res)

            # Reshape the accu array to a 1D array (flatten it)
            accu_flat = t_first.flatten()

            # Total number of observations (length of the accu_flat array)
            n_observations = len(accu_flat)

            # Create the 'tempDynamics' factor: repeat each tempDynamics value for each contrast and each init replicate
            tempDynamics_repeated = np.repeat(tempDynamics, len(contrasts) * init)

            # Create the 'contrasts' factor: repeat each contrast value for each init replicate, then tile for tempDynamics
            contrasts_repeated = np.tile(np.repeat(contrasts, init), len(tempDynamics))

            # Check that all arrays have the same length
            assert len(tempDynamics_repeated) == len(contrasts_repeated) == n_observations, \
                "The lengths of the factors and the response variable must match!"

        # Create a DataFrame from the reshaped data
        df = pd.DataFrame({
            'tempDynamics': tempDynamics_repeated,
            'contrasts': contrasts_repeated,
            'accu': accu_flat
        })

        # Perform two-way ANOVA
        model = ols('accu ~ C(tempDynamics) + C(contrasts) + C(tempDynamics):C(contrasts)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA
        print(anova_table)

        # Convert tempDynamics and contrasts to strings before concatenating
        df['interaction'] = df['tempDynamics'].astype(str) + ':' + df['contrasts'].astype(str)

        # Post-hoc test: Tukey's HSD for the interaction
        tukey_interaction = pairwise_tukeyhsd(endog=df['accu'], groups=df['interaction'], alpha=0.05)
        print(tukey_interaction)


        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_decoding(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 10
    fontsize_legend         = 15

    try:
        layers = args[0]
        t_steps = args[1]
    except:
        pass

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(layers), figsize=(9, 2), sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):
                
            for iL in range(len(layers)):

                # select data
                data_current = accu[iD, iT, :, :, iL, :].mean(1)

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iL].plot(np.arange(t_steps)+1, data_mean, color=tdyn_color[iT]) # lw=lw[-iC]
                axs[iL].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

                # adjust axes
                axs[iL].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[iL].spines['top'].set_visible(False)
                axs[iL].spines['right'].set_visible(False)
                axs[iL].set_xticks(np.arange(t_steps)+1)

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')

def visualize_activations(task, accu, tempDynamics, current_analysis, dataset, init, root, *args):

    # fontsize
    fontsize_title          = 20
    fontsize_label          = 15
    fontsize_tick           = 10
    fontsize_legend         = 15

    try:
        layers = args[0]
        t_steps = args[1]
        contrasts = args[2]
    except:
        pass

    # layer
    layer_idx = 0

    # color
    colors = plt.cm.viridis(np.linspace(0, 1, len(contrasts)))

    # visualize
    for iD, current_dataset in enumerate(dataset):

        # initiate figure
        fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10.5, 1.5), sharey=True)

        for iT, current_tempDynamics in enumerate(tempDynamics):
                
            for iC, contrast in enumerate(contrasts):

                # select data
                # print(np.max(accu[iD, iT, :, iC, layer_idx, :], 0).shape)
                data_current = accu[iD, iT, :, iC, layer_idx, :]/np.max(accu[iD, iT, :, iC, layer_idx, :], 1)

                # compute averages
                data_mean = np.mean(data_current, 0)
                data_sem = np.std(data_current, 0)/math.sqrt(init)

                # visualize
                axs[iT].plot(np.arange(t_steps)+1, data_mean, color=colors[-1-iC])#, alpha=0.4+iC*0.1) # lw=lw[-iC]
                # axs[iT].fill_between(np.arange(t_steps)+1, data_mean - data_sem, data_mean + data_sem, color=tdyn_color[iT], alpha=0.3) # lw=lw[-iC]

                # adjust axes
                axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[iT].spines['top'].set_visible(False)
                axs[iT].spines['right'].set_visible(False)
                axs[iT].set_xticks(np.arange(t_steps)+1)
                # axs[iT].set_ylim(0, 0.1)

        # save figure
        plt.tight_layout()
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset)
        plt.savefig(root + 'visualization/' + task + '/' + current_analysis + '_' + current_dataset + '.svg')