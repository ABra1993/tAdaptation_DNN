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

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# computes result as presented in FIG2

# define root
root                = '' ## ADD home directory

# compute or load metrics
preload = True

# define number of timesteps
t_steps = 3

# hyperparameter specification
init            = 5
batch_size      = 64

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['none', 'add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']

######## ---------------- TASK
task = 'repeated_noise'

######## ---------------- ANAlYSIS
# analysis = ['in_distribution', 'out_distribution_contrast', 'out_distribution_std', 'out_distribution_offset', 'out_distribution_noise', 'out_distribution_shift', 'out_distribution_different']
# analysis = ['out_distribution_std', 'out_distribution_offset', 'out_distribution_noise', 'out_distribution_shift', 'out_distribution_different']

analysis = ['in_distribution']
# analysis = ['out_distribution_contrast']

# analysis = ['out_distribution_std', 'out_distribution_offset', 'out_distribution_noise']

# analysis = ['out_distribution_shift']
# analysis = ['out_distribution_different']
# analysis = ['out_distribution_shift', 'out_distribution_different']

######## ---------------- DATASET
# dataset = ['mnist', 'fmnist', 'cifar']
dataset = ['fmnist', 'cifar']
# dataset = ['cifar']

# other
adapters = ['same', 'different']

if preload == False:

    # initaite dataloaders
    for _, current_dataset in enumerate(dataset):

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

        for _, current_analysis in enumerate(analysis):

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)
            print(n_bin)
            print(depVar)
            print(n_samples)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            for _, current_tempDynamics in enumerate(tempDynamics):

                # test performance
                if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                    accu_current = np.zeros((init, len(contrasts)))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts)))
                elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise') | (current_analysis == 'out_distribution_shift'):
                    accu_current = np.zeros((init, len(contrasts), len(depVar)))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), len(depVar)))
                elif (current_analysis == 'out_distribution_different'):
                    accu_current = np.zeros((init, len(contrasts), len(adapters)))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), len(adapters)))
                
                for iInit in range(init):

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
                            if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast') | (current_analysis == 'out_distribution_different'):
                                ax = sequence_test(imgs, current_analysis, contrast)
                            else:
                                ax = sequence_test(imgs, current_analysis, contrast, n_bin, n_samples, depVar)

                            # forward pass
                            testoutp = model.forward(ax)
                            predicy = torch.argmax(testoutp, dim=1).to('cpu')

                            # save accuracies
                            if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                                accu_per_batch[a, iC] = (predicy == lbls).sum().item() / float(lbls.size(0))
                            elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise') | (current_analysis == 'out_distribution_shift'):
                                for i in range(n_bin):
                                    accu_per_batch[a, iC, i] = (predicy[i*n_samples:i*n_samples+n_samples] == lbls[i*n_samples:i*n_samples+n_samples]).sum().item() / float(lbls[i*n_samples:i*n_samples+n_samples].size(0))
                            elif (current_analysis == 'out_distribution_different'):
                                accu_per_batch[a, iC, 0] = (predicy[:int(imgs.shape[0]/2)] == lbls[:int(imgs.shape[0]/2)]).sum().item() / float(lbls[:int(imgs.shape[0]/2)].size(0))
                                accu_per_batch[a, iC, 1] = (predicy[int(imgs.shape[0]/2):] == lbls[int(imgs.shape[0]/2):]).sum().item() / float(lbls[int(imgs.shape[0]/2):].size(0))

                                # accu_per_batch[a, iC, 0] = (predicy == lbls).sum().item() / float(lbls.size(0))
                                # accu_per_batch[a, iC, 1] = (predicy == lbls).sum().item() / float(lbls.size(0))

                            # save losses and print progress
                            if (a%20 == 0) & (a != 0):
                                print('Inference...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- batch ', a, '/', str(len(ldrTest)).ljust(10), ' ----- accuracy: ', accu_per_batch[a].mean().item())

                        # save accuracies
                        if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                            accu_current[iInit, iC] = np.mean(accu_per_batch[:, iC])
                        elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise') | (current_analysis == 'out_distribution_shift'):
                            for i in range(n_bin):
                                accu_current[iInit, iC, i] = np.mean(accu_per_batch[:, iC, i])
                        elif (current_analysis == 'out_distribution_different'):
                            accu_current[iInit, iC, 0] = np.mean(accu_per_batch[:, iC, 0])
                            accu_current[iInit, iC, 1] = np.mean(accu_per_batch[:, iC, 1])

                # save accuracies
                np.save(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics, accu_current)

# import accuracy
for iA, current_analysis in enumerate(analysis):

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    # initiate accuracy storage
    if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, len(contrasts)))
    elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise') | (current_analysis == 'out_distribution_shift'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, len(contrasts), len(depVar)))
    elif (current_analysis == 'out_distribution_different'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, len(contrasts), len(adapters)))

    # retrieve metric values
    for iD, current_dataset in enumerate(dataset):
        for iT, current_tempDynamics in enumerate(tempDynamics):
            if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                accu[iD, iT, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
            elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise') | (current_analysis == 'out_distribution_shift') | (current_analysis == 'out_distribution_different'):
                accu[iD, iT, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')

    # visualize
    if current_analysis == 'in_distribution':
        visualize_in_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root)
    elif current_analysis == 'out_distribution_contrast':
        visualize_out_distribution_contrast(task, accu, tempDynamics, current_analysis, dataset, init, root, contrasts)
    elif (current_analysis == 'out_distribution_std') | (current_analysis == 'out_distribution_offset') | (current_analysis == 'out_distribution_noise'):
        visualize_out_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root, contrasts, depVar)
    elif (current_analysis == 'out_distribution_shift'):
        visualize_out_distribution_shift(task, accu, tempDynamics, current_analysis, dataset, init, root, contrasts, adapters)
    elif (current_analysis == 'out_distribution_different'):
        visualize_out_distribution_different(task, accu, tempDynamics, current_analysis, dataset, init, root, contrasts, adapters)





