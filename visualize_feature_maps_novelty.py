import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim

# models
from models.cnn_feedforward_novelty import cnn_feedforward_novelty
from model_train_utils import *

from model_analyse_novelty_utils import *
from model_analyse_novelty_vis import *

# set the seed
seed = 44
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '' # ADD home directory

# define number of timesteps
t_steps = 10

# hyperparameter specification
batch_size      = 1

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['none', 'add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']

######## ---------------- TASK
# task = 'novelty'
task = 'novelty_augmented'

######## ---------------- DATASET
# dataset = ['mnist', 'fmnist', 'cifar']
dataset = ['fmnist'] #, 'fmnist', 'cifar']

# contrast
onset = 4

# plot settings
fig, axs = plt.subplots(len(tempDynamics)+1, t_steps)#, figsize=(10, 2*len(dataset)))

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

        # create sequence
        # lbls_t, ax = sequence_train(imgs, task, lbls, current_dataset)
        # lbls_t, ax = sequence_test(imgs, task, 'single_image', lbls, current_dataset, t_steps, onset)
        lbls_t, ax, qdr_idx = sequence_test(imgs, task, 'onset_accu', lbls, current_dataset, t_steps, onset)

        for t in range(t_steps):

            # plot stimulus
            if current_dataset != 'cifar':
                axs[0, t].imshow(ax[0, t, :, :, :].permute(1, 2, 0), cmap='gray')
            else:
                axs[0, t].imshow(ax[0, t, :, :, :].permute(1, 2, 0))
            axs[0, t].axis('off')

        for itDyn, current_tempDynamics in enumerate(tempDynamics):

            # initiate model
            model = cnn_feedforward_novelty(channels, task, current_dataset)

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

            # visualize
            for t in range(t_steps):
            
                # get activations for last timestep
                fmap = model.actvs[0][t].detach().cpu().mean(1) # first layer last timestep (containing target digit)
                # print(fmap.shape)

                # plot activations
                vmin = 0
                vmax = 0.7
                axs[itDyn+1, t].imshow(fmap.permute(1, 2, 0), cmap='viridis', vmin=vmin, vmax=vmax)

                # adjust axes
                axs[itDyn+1, t].axis('off')
                axs[itDyn+1, t].axis('off')

# save figure
# plt.tight_layout()
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + dataset[0], bbox_inches='tight')
plt.savefig(root + 'visualization/feature_maps/feature_map_' + task + '_' + dataset[0] + '.svg', bbox_inches='tight')

