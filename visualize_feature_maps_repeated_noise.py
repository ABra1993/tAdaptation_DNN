import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

from model_analyse_repeated_noise_utils import *
from model_analyse_repeated_noise_vis import *

# set the seed
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '' # ADD home directory

# define number of timesteps
t_steps = 3

# hyperparameter specification
batch_size      = 2

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']

######## ---------------- TASK
task = 'repeated_noise'

######## ---------------- DATASET
# dataset = ['mnist', 'fmnist', 'cifar']
dataset = ['mnist', 'fmnist', 'cifar']

# other
adapters = ['same', 'different']

# contrast
contrast = 1
if contrast == 1:
    contrast_lbl = 'high'
else:
    contrast_lbl = 'low'

# plot settings
fig, axs = plt.subplots(len(dataset), 9, figsize=(10, 4))

# initaite dataloaders
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

    # load stimuli
    _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

    for a, (imgs, lbls) in enumerate(ldrTest):

        if (a != 0):
            break

        # set image
        imgs[1, :, :, :] = imgs[0, :, :, :]

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

            # create sequence
            ax = sequence_test(imgs, 'out_distribution_different', contrast)

            # plot activations
            if current_dataset != 'cifar':
                axs[iD, 0].imshow(ax[-1][0, :, :, :].permute(1, 2, 0), cmap='gray')
            else:
                axs[iD, 0].imshow(ax[-1][0, :, :, :].permute(1, 2, 0))
            axs[iD, 0].axis('off')

            # forward pass
            testoutp = model.forward(ax)

            # get activations for last timestep
            ax_conv1 = model.actvs[0][2].detach().cpu().mean(1).unsqueeze(1) # first layer last timestep (containing target digit)

            for iA in range(len(adapters)):

                # select data
                fmap = ax_conv1[iA]

                # plot activations
                axs[iD, 1+2*itDyn+iA].imshow(fmap.permute(1, 2, 0), cmap='viridis')

                # adjust axes
                axs[iD, 1+2*itDyn+iA].axis('off')
                axs[iD, 1+2*itDyn+iA].axis('off')

# save figure
plt.tight_layout()
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + contrast_lbl)
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + contrast_lbl + '.svg')

