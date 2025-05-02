import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

from model_analyse_occlusion_utils import *
from model_analyse_occlusion_vis import *

# set the seed
seed = 28
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '' # ADD home directory

# hyperparameter specification
batch_size      = 1

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']

######## ---------------- TASK
task = 'occlusion'

######## ---------------- DATASET
# dataset = ['mnist', 'fmnist', 'cifar']
dataset = ['cifar']

# other
adapters = ['same', 'different']

# contrast
contrast = 1
if contrast == 1:
    contrast_lbl = 'high'
else:
    contrast_lbl = 'low'

# initaite dataloaders
for iD, current_dataset in enumerate(dataset):

    # define number of timesteps
    if current_dataset == 'cifar':
        t_steps = 10
    else:
        t_steps = 9

    # plot settings
    fig, axs = plt.subplots(len(tempDynamics)+1, t_steps)#, figsize=(10, 4))

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

    # load stimuli
    _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

    for a, (imgs, lbls) in enumerate(ldrTest):

        if (a != 0):
            break

        # create sequence
        ax = sequence_test(imgs, 'out_distribution_different', contrast)

        for t in range(t_steps):

            # plot activations
            if current_dataset != 'cifar':
                axs[0, t].imshow(ax[t][0, :, :, :].permute(1, 2, 0), cmap='gray')
            else:
                axs[0, t].imshow(ax[t][0, :, :, :].permute(1, 2, 0))

            # adjust axes
            axs[iD, t].axis('off')


        for itDyn, current_tempDynamics in enumerate(tempDynamics):

            # initiate model
            model = cnn_feedforward(channels, task, current_dataset)

            # introduce temporal dynamics
            model.init_t_steps(t_steps)
            if tempDynamics != 'none':
                model.initialize_tempDynamics(current_tempDynamics)

            # load weights
            model.load_state_dict(torch.load(root + 'weights/' + task + '/' + task + '_' + current_tempDynamics + '_' + current_dataset + '_' + str(1)))
            model.to(device)
            model.eval()

            # forward pass
            testoutp = model.forward(ax)

            for t in range(t_steps):

                # get activations for last timestep
                fmap = model.actvs[0][t].detach().cpu().mean(1) # first layer last timestep (containing target digit)

                # plot activations
                axs[itDyn+1, t].imshow(fmap.permute(1, 2, 0), cmap='viridis')

                # adjust axes
                axs[itDyn+1, t].axis('off')


# save figure
plt.tight_layout()
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + contrast_lbl + '_' + current_dataset)
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + contrast_lbl + '_' + current_dataset + '.svg')

