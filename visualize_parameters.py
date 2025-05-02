import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim
from torchsummary import summary
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd

# models
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_novelty import cnn_feedforward_novelty
from model_train_utils import *
from visualize_utils import *

# set the seed
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# top or bottom
top = False

# define root
root                = '' ## ADD home directory

# dataset
dataset = ['mnist', 'fmnist', 'cifar']

# tasks = ['core', 'repeated_noise', 'diffusion', 'occlusion', 'novelty']
# tasks = ['repeated_noise']
# tasks = ['diffusion']
tasks = ['occlusion']

# # for NOVELTY - moment of introducing novel object
# n_add = 4

# hyperparameter specification
init            = 3

# # print summary
# print(30*'-')
# print('Temporal dynamci(s): '.ljust(30), tempDynamics)
# print('Dataset(s): '.ljust(30), dataset)
# print('Task(s): '.ljust(30), tasks)
# print(30*'-')

# fontsize
fontsize_title          = 20
fontsize_label          = 12
fontsize_tick           = 10
fontsize_legend         = 12

# color
color_dataset = ['deepskyblue', 'dodgerblue', 'royalblue']

# training
offset_iD = [-1, 0, 1]
offset_itDyn = [0, 2]
for task in tasks:

    # set
    if top:

        tempDynamics        = ['add_supp', 'div_norm'] # top
        fig, axs = plt.subplots(1, 5, figsize=(9, 2))
        color_tDyn = ['#ACD39E', '#E49C39'] # top

    else:

        tempDynamics        = ['l_recurrence_A', 'l_recurrence_M'] # bottom
        fig, axs = plt.subplots(1, 4, figsize=(9, 2))
        color_tDyn = ['#225522', '#DF4828'] # bottom

    for itDyn, current_tempDynamics in enumerate(tempDynamics):

        # initiate dataframe to store test performance
        if current_tempDynamics == 'add_supp':
            params_lbls = [r'$\alpha$', r'$\beta$']
            params = np.zeros((init, len(dataset), len(params_lbls)))
        elif current_tempDynamics == 'div_norm':
            params_lbls = [r'$\alpha$', r'$\sigma$', r'$K$']
            params = np.zeros((init, len(dataset), len(params_lbls)))
        elif (current_tempDynamics == 'l_recurrence_A') | (current_tempDynamics == 'l_recurrence_M'):
            params_lbls = [r'$W$', r'$b$']
            params = np.zeros((init, len(dataset), len(params_lbls)))

        for iD, current_dataset in enumerate(dataset):

            # classes the dataset
            if current_dataset == 'mnist': 
                classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
                channels = 1
            elif current_dataset == 'fmnist':
                classes = ('T-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
                channels = 1
            elif current_dataset == 'cifar':   
                classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                channels = 3

            for iInit in range(init):

                # initiate model
                if (task == 'novelty') | (task == 'novelty_augmented'):
                    model = cnn_feedforward_novelty(channels, task, current_dataset)
                else:
                    model = cnn_feedforward(channels, task, current_dataset)

                # introduce temporal dynamics
                model.init_t_steps()
                if tempDynamics != 'none':
                    model.initialize_tempDynamics(current_tempDynamics)

                # load weights
                model.load_state_dict(torch.load(root + 'weights/' + task + '/' + task + '_' + current_tempDynamics + '_' + current_dataset + '_' + str(iInit+1)))
                model.to(device)
                model.eval()

                if current_tempDynamics == 'add_supp':
                    params[iInit, iD, 0] = model.sconv1.alpha
                    params[iInit, iD, 1] = model.sconv1.beta
                elif current_tempDynamics == 'div_norm':
                    params[iInit, iD, 0] = model.sconv1.alpha
                    params[iInit, iD, 1] = model.sconv1.sigma
                    params[iInit, iD, 2] = model.sconv1.K
                elif (current_tempDynamics == 'l_recurrence_A') | (current_tempDynamics == 'l_recurrence_M'):
                    params[iInit, iD, 0] = torch.mean(model.sconv1.weight.flatten()).detach().cpu().numpy()
                    params[iInit, iD, 1] = torch.mean(model.sconv1.bias.flatten()).detach().cpu().numpy()

        # visualize
        for iP, param in enumerate(params_lbls):
            for iD, current_dataset in enumerate(dataset):

                # select data
                data_current = params[:, iD, iP]

                # compute averages
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(init)

                # visualize
                axs[offset_itDyn[itDyn]+iP].scatter(offset_iD[iD], data_mean, color=color_dataset[iD])
                axs[offset_itDyn[itDyn]+iP].plot([offset_iD[iD], offset_iD[iD]], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=-10)
                sns.stripplot(x=np.ones(init)*(offset_iD[iD]), y=data_current, jitter=True, ax=axs[offset_itDyn[itDyn]+iP], color='silver', size=3, native_scale=True, zorder=-1)

            # adjust axes
            axs[offset_itDyn[itDyn]+iP].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            axs[offset_itDyn[itDyn]+iP].spines['top'].set_visible(False)
            axs[offset_itDyn[itDyn]+iP].spines['right'].set_visible(False)
            axs[offset_itDyn[itDyn]+iP].set_title(param, fontsize=fontsize_title, color=color_tDyn[itDyn])
            axs[offset_itDyn[itDyn]+iP].set_xlim(-1.5, 1.5)
            axs[offset_itDyn[itDyn]+iP].set_xticks(offset_iD)
            axs[offset_itDyn[itDyn]+iP].set_xticklabels([' ', ' ', ' '])#, rotation=45, fontsize=fontsize_label)

            # perform statistical testing
            print('\nTemp. dynamics: ', current_tempDynamics, param)
            res = f_oneway(params[:, 0, iP], params[:, 1, iP], params[:, 2, iP])
            print(res)
            if res[1] < 0.05:
                res = tukey_hsd(params[:, 0, iP], params[:, 1, iP], params[:, 2, iP])
                print(res)

    # save figure
    plt.tight_layout()
    if top:
        plt.savefig(root + 'visualization/params/param_' + task + '_top')
        plt.savefig(root + 'visualization/params/param_' + task + '_top.svg')
    else:
        plt.savefig(root + 'visualization/params/param_' + task + '_bottom')
        plt.savefig(root + 'visualization/params/param_' + task + '_bottom.svg')