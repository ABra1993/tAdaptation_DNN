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
from models.cnn_feedforward_novelty import cnn_feedforward_novelty
from model_train_utils import *

# from model_analyse_novelty_utils import *
from model_analyse_novelty_utils import *
from model_analyse_novelty_vis import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# computes result as presented in FIG5

# define root
root                = '' ## ADD home directory

# compute or load metrics
preload = True

# define number of timesteps
t_steps = 20

# height and width (f)MNIST after first convolutional layer
h = 28
w = 28
c = 1
# print(c, h, w)

# define quadrants
quadrants_coord = np.array([[[0, h], [0, w]], 
                            [[0, h], [w, 2*w]], 
                            [[h, 2*h], [0, w]], 
                            [[h, 2*h], [w, 2*w]]], dtype=int)

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
task = 'novelty'
# task = 'novelty_augmented'

######## ---------------- ANAlYSIS
# analysis = ['in_distribution', 'out_distribution_contrast', 'decoding']

analysis = ['in_distribution']
# analysis = ['single_image']

# analysis = ['onset_accu']
# analysis = ['onset_activations']

# analysis = ['contrast_accu']
# analysis = ['contrast_activations']

# analysis = ['intervention']

######## ---------------- DATASET
# dataset = ['mnist', 'fmnist', 'cifar']
# dataset = ['fmnist', 'cifar']
dataset = ['mnist', 'fmnist']
# dataset = ['fmnist']

# layers
layers = ['conv1', 'conv2', 'conv3', 'fc1']

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

        for _, current_analysis in enumerate(analysis):

            _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            depVar, _, _, _ = create_samples(current_analysis, batch_size)
            print(depVar)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            for _, current_tempDynamics in enumerate(tempDynamics):

                # test performance
                if (current_analysis == 'in_distribution') | (current_analysis == 'single_image'):
                    accu_current = np.zeros((init, t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), t_steps))
                elif (current_analysis == 'onset_accu'):
                    accu_current = np.zeros((len(depVar), init, t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), t_steps))
                elif (current_analysis == 'onset_activations'):
                    accu_current = np.zeros((len(depVar), init, 2, t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), 2, t_steps))

                for iInit in range(init):
                    for idepVar, depVar_current in enumerate(depVar): # iterate over onsets

                        # initiate model
                        model = cnn_feedforward_novelty(channels, task, current_dataset)

                        # show summary
                        print('\n', current_tempDynamics, current_dataset)
                        model.init_t_steps(1)
                        if (current_dataset == 'cifar'):
                            custom_summary(model.to(device), input_size=(batch_size, 3, 64, 64))
                        else:
                            custom_summary(model.to(device), input_size=(batch_size, 1, 56, 56))

                        # introduce temporal dynamics
                        model.init_t_steps(t_steps)
                        if tempDynamics != 'none':
                            model.initialize_tempDynamics(current_tempDynamics)
                        print('Number of parameters:'.ljust(25), count_parameters(model))

                        # load weights
                        model.load_state_dict(torch.load(root + 'weights/' + task + '/' + task + '_' + current_tempDynamics + '_' + current_dataset + '_' + str(iInit+1)))
                        model.to(device)
                        model.eval()

                        for a, (imgs, lbls) in enumerate(ldrTest):

                            # if (a != 0):
                            #     break

                            # create sequence
                            if current_analysis == 'in_distribution':
                                lbls_t, ax = sequence_train(imgs, task, lbls, current_dataset)
                            elif (current_analysis == 'onset_accu') | (current_analysis == 'onset_activations'): 
                                lbls_t, ax, qdr_idx = sequence_test(imgs, task, current_analysis, lbls, current_dataset, t_steps, depVar_current)
                            else:
                                lbls_t, ax = sequence_test(imgs, task, current_analysis, lbls, current_dataset, t_steps, depVar_current)

                            # compute step
                            outp = model.forward(ax)

                            if (current_analysis == 'onset_activations') | (current_analysis == 'contrast_activations'): 
                                
                                for t in range(t_steps):

                                    # select activations                                    
                                    ax_conv1 = model.actvs[0][t].detach().cpu().mean(1) # first layer last timestep (containing target digit)
                                    # print(ax_conv1.shape)

                                    # save accuracy
                                    for iQ, current_quadrant in enumerate(qdr_idx):
                                        # print(current_quadrant)
                                        
                                        # select
                                        ax_conv1_current = ax_conv1[:, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]]
                                        # print(ax_conv1_current.shape)

                                        # average
                                        accu_per_batch[a, iQ, t] = ax_conv1_current.mean()

                            else:

                                # compute accuracy
                                for t in range(lbls_t.shape[1]):
                                    predicy = torch.argmax(outp[:, :, t], dim=1).to('cpu')
                                    accu_per_batch[a, t] = (predicy == lbls_t[:, t]).sum().item() / float(lbls_t.shape[0])

                                    # acc = (predicy == lbls_t[:, t]).sum().item() / float(lbls_t.shape[0])
                                    # print(acc)

                            # save losses and print progress
                            if (a%20 == 0) & (a != 0):
                                print('Inference...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- batch ', a, '/', str(len(ldrTest)).ljust(10), ' ----- accuracy: ', accu_per_batch[a].mean().item())

                        # save accuracies
                        if (current_analysis == 'in_distribution')  | (current_analysis == 'single_image'):
                            accu_current[iInit, :] = accu_per_batch.mean(0)
                        elif (current_analysis == 'onset_accu'):
                            accu_current[idepVar, iInit, :] = accu_per_batch.mean(0)
                        elif (current_analysis == 'onset_activations'):
                            accu_current[idepVar, iInit, :, :] = accu_per_batch.mean(0)

                # save accuracies
                np.save(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics, accu_current)

# import accuracy
for iA, current_analysis in enumerate(analysis):

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    depVar, _, _, _ = create_samples(current_analysis, batch_size)

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    # initiate accuracy storage
    if (current_analysis == 'in_distribution') | (current_analysis == 'single_image'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, t_steps))
    elif (current_analysis == 'onset_accu'):
        accu = np.zeros((len(dataset), len(tempDynamics), len(depVar), init, t_steps))
    elif (current_analysis == 'onset_activations'):
        accu = np.zeros((len(dataset), len(tempDynamics), len(depVar), init, 2, t_steps))
    
    # retrieve metric values
    for iD, current_dataset in enumerate(dataset):
        for iT, current_tempDynamics in enumerate(tempDynamics):
            if (current_analysis == 'in_distribution') | (current_analysis == 'single_image'):
                accu[iD, iT, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
            elif (current_analysis == 'onset_accu'):
                accu[iD, iT, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
            elif (current_analysis == 'onset_activations'):
                accu[iD, iT, :, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
    
    # visualize
    if (current_analysis == 'in_distribution') | (current_analysis == 'single_image'):
        visualize_in_distribution(task, accu, tempDynamics, current_analysis, dataset, init, root, t_steps)
    elif (current_analysis == 'onset_accu'):
        onset_accu(task, accu, tempDynamics, current_analysis, dataset, init, root, t_steps, depVar)
    elif (current_analysis == 'onset_activations'):
        onset_activations(task, accu, tempDynamics, current_analysis, dataset, init, root, t_steps, depVar)
    

