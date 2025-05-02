import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

from model_analyse_occlusion_utils import *
from model_analyse_occlusion_vis import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# computes result as presented in FIG4

# define root
root                = '' ## ADD home directory

# compute or load metrics
preload = True

# hyperparameter specification
init            = 10
batch_size      = 64

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['none', 'add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']
# tempDynamics = ['none', 'div_norm']

######## ---------------- TASK
task = 'occlusion'

######## ---------------- ANAlYSIS
# analysis = ['in_distribution']

analysis = ['in_distribution']
# analysis = ['activations']
# analysis = ['both']

######## ---------------- DATASET
dataset = ['mnist', 'fmnist', 'cifar']
# dataset = ['mnist']

# layers
layers = ['conv1', 'conv2', 'conv3', 'fc1']

if preload == False:

    # initaite dataloaders
    for _, current_dataset in enumerate(dataset):

        # define number of timesteps
        if current_dataset == 'cifar':
            t_steps = 10
        else:
            t_steps = 9

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

        for _, current_analysis in enumerate(analysis):

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS
            for _, current_tempDynamics in enumerate(tempDynamics):

                if (current_analysis == 'decoding'):
                    _, ldrTest = load_dataset(current_dataset, 1000, task, train=False)
                else:
                    _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

                # test performance
                if (current_analysis == 'in_distribution'):
                    accu_current = np.zeros((init, len(contrasts), t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), t_steps))
                elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
                    accu_current = np.zeros((init, len(contrasts), len(layers), t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), len(layers), t_steps))

                for iInit in range(init):

                    if (current_analysis == 'decoding'):
                        _, ldrTest = load_dataset(current_dataset, 1000, task, train=False)
                    else:
                        _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

                    # initiate model
                    model = cnn_feedforward(channels, task, current_dataset)

                    # show summary
                    model.init_t_steps(1)
                    if (current_dataset == 'cifar'):
                        custom_summary(model.to(device), input_size=(batch_size, 3, 32, 32))
                    else:
                        custom_summary(model.to(device), input_size=(batch_size, 1, 28, 28))
                    print('Number of parameters:'.ljust(25), count_parameters(model))

                    # introduce temporal dynamics
                    model.init_t_steps(t_steps)
                    if tempDynamics != 'none':
                        model.initialize_tempDynamics(current_tempDynamics)

                    # load weights
                    model.load_state_dict(torch.load(root + 'weights/' + task + '/' + task + '_' + current_tempDynamics + '_' + current_dataset + '_' + str(iInit+1)))
                    model.to(device)
                    model.eval()
                    
                    # perform inference
                    for iC, contrast in enumerate(contrasts):

                        print('\n')
                        print(30*'-')
                        print('Contrast: ', contrast)
                        print(30*'-')

                        for a, (imgs, lbls) in enumerate(ldrTest):

                            # create sequence
                            ax = sequence_test(imgs, current_analysis, contrast)

                            # forward pass
                            testoutp = model.forward(ax)

                            if (current_analysis == 'in_distribution'):

                                for t in range(t_steps):

                                    # compute accuracy
                                    testoutp = model.decoder(model.actvs[3][t])
                                    predicy = torch.argmax(testoutp, dim=1).to('cpu')

                                    # save accuracies
                                    accu_per_batch[a, iC, t] = (predicy == lbls).sum().item() / float(lbls.size(0))

                            elif (current_analysis == 'activations'):
                                
                                # iterate over layers and timesteps
                                for iL in range(len(layers)):
                                    for t in range(t_steps):

                                        # select activations
                                        ax_conv1 = model.actvs[iL][t].detach().cpu().mean(1).unsqueeze(1)
                                        accu_per_batch[a, iC, iL, t] = ax_conv1.mean()

                            # save losses and print progress
                            if (a%20 == 0) & (a != 0):
                                print('Inference...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- batch ', a, '/', str(len(ldrTest)).ljust(10), ' ----- accuracy: ', accu_per_batch[a].mean().item())

                        # save accuracies
                        if (current_analysis == 'in_distribution'):
                            for iC in range(len(contrasts)):
                                for t in range(t_steps):
                                    accu_current[iInit, iC, t] = accu_per_batch[:, iC, t].mean(0)
                        elif (current_analysis == 'activations'):
                            for iL in range(len(layers)):
                                for t in range(t_steps):
                                    accu_current[iInit, iC, iL, t] = accu_per_batch[:, iC, iL, t].mean(0)

                # save accuracies
                np.save(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics, accu_current)

# import accuracy
if analysis[0] == 'both':

    visualize_both(task, tempDynamics, init, root)

else:

    for iA, current_analysis in enumerate(analysis):

        # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

        contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)

        # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

        # retrieve metric values
        for iD, current_dataset in enumerate(dataset):

            # define number of timesteps
            if current_dataset == 'cifar':
                t_steps = 10
            else:
                t_steps = 9

            # initiate accuracy storage
            if (current_analysis == 'in_distribution'):
                accu = np.zeros((len(tempDynamics), init, len(contrasts), t_steps))
            elif (current_analysis == 'activations'):
                accu = np.zeros((len(tempDynamics), init, len(contrasts), len(layers), t_steps))

            for iT, current_tempDynamics in enumerate(tempDynamics):
                if (current_analysis == 'in_distribution'):
                    temp = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
                    print(temp.shape)
                    accu[iT, :, :, :] =  temp
                elif (current_analysis == 'activations'):
                    accu[iT, :, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')

            # visualize
            if current_analysis == 'in_distribution':
                # visualize_in_distribution(task, accu, tempDynamics, current_analysis, current_dataset, init, root, t_steps)
                visualize_in_distribution_withoutAvg(task, accu, tempDynamics, current_analysis, current_dataset, init, root, t_steps)
            elif current_analysis == 'activations':
                visualize_activations(task, accu, tempDynamics, current_analysis, current_dataset, init, root, t_steps)