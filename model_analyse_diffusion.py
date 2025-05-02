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

from model_analyse_diffusion_utils import *
from model_analyse_diffusion_vis import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# computes result as presented in FIG3

# define root
root                = '' ## ADD home directory

# compute or load metrics
preload = True

# define number of timesteps
t_steps = 10

# hyperparameter specification
init            = 10
batch_size      = 64

######## ---------------- TEMPORAL DYNAMICS
tempDynamics = ['none', 'add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics =  ['l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['none', 'add_supp', 'div_norm']
# tempDynamics = ['none', 'add_supp']

######## ---------------- TASK
task = 'diffusion'

######## ---------------- ANAlYSIS
# analysis = ['in_distribution', 'out_distribution_contrast', 'decoding']

analysis = ['in_distribution']
# analysis = ['out_distribution_contrast']
# analysis = ['decoding']
# analysis = ['activations']

######## ---------------- DATASET
dataset = ['mnist', 'fmnist', 'cifar']
# dataset = ['fmnist', 'cifar']
dataset = ['mnist']

# layers
layers = ['conv1', 'conv2', 'conv3', 'fc1']

# initiate accuracy storage
accu_repeated_noise = np.zeros((len(dataset), len(tempDynamics), 5, 2))
for iD, current_dataset in enumerate(dataset):
    for iT, current_tempDynamics in enumerate(tempDynamics):
            accu_repeated_noise[iD, iT, :, :] = np.load(root + 'metrics/repeated_noise/' + current_dataset + '_in_distribution_' + current_tempDynamics + '.npy')

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

            if (current_analysis == 'decoding'):
                _, ldrTest = load_dataset(current_dataset, 1000, task, train=False)
            else:
                _, ldrTest = load_dataset(current_dataset, batch_size, task, train=False)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)

            # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

            for _, current_tempDynamics in enumerate(tempDynamics):

                # test performance
                if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                    accu_current = np.zeros((init, len(contrasts), t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), t_steps))
                elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
                    accu_current = np.zeros((init, len(contrasts), len(layers), t_steps))
                    accu_per_batch = np.zeros((len(ldrTest), len(contrasts), len(layers), t_steps))
                
                for iInit in range(init):

                    # initiate model
                    model = cnn_feedforward(channels, task, current_dataset)

                    # show summary
                    print('\n', current_tempDynamics, current_dataset)
                    # model.init_t_steps(1)
                    # if (current_dataset == 'cifar'):
                    #     custom_summary(model.to(device), input_size=(batch_size, 3, 32, 32))
                    # else:
                    #     custom_summary(model.to(device), input_size=(batch_size, 1, 28, 28))

                    # introduce temporal dynamics
                    model.init_t_steps(t_steps)
                    if tempDynamics != 'none':
                        model.initialize_tempDynamics(current_tempDynamics)
                    print('Number of parameters:'.ljust(25), count_parameters(model))

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

                            if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):

                                for t in range(t_steps):

                                    # compute accuracy
                                    testoutp = model.decoder(model.actvs[3][t])
                                    predicy = torch.argmax(testoutp, dim=1).to('cpu')

                                    # save accuracies
                                    accu_per_batch[a, iC, t] = (predicy == lbls).sum().item() / float(lbls.size(0))

                            elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
                                
                                # iterate over layers and timesteps
                                for iL in range(len(layers)):
                                # for iL in range(1, 2):
                                    for t in range(t_steps):

                                        # select activations
                                        ax_conv1 = model.actvs[iL][t].detach().cpu().mean(1).unsqueeze(1)

                                        if (current_analysis == 'decoding'):
                                            n_samples = ax_conv1.shape[0]
                                            ax_conv1_flat = ax_conv1.view(n_samples, -1)  # flatten to (n_samples, n_features)

                                            # convert to numpy for sklearn compatibility
                                            X = ax_conv1_flat.numpy()
                                            y = lbls.numpy()

                                            # normalize the data
                                            scaler = StandardScaler()
                                            X_scaled = scaler.fit_transform(X)

                                            # initialize SVM classifier
                                            svm = SVC(kernel='linear')

                                            # perform cross-validation (5-fold cross-validation)
                                            cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
                                            # print(cv_scores)

                                            # store
                                            accu_per_batch[a, iC, iL, t] = np.mean(cv_scores)

                                        else:

                                            accu_per_batch[a, iC, iL, t] = ax_conv1.mean()

                            # save losses and print progress
                            if (a%20 == 0) & (a != 0):
                                print('Inference...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- batch ', a, '/', str(len(ldrTest)).ljust(10), ' ----- accuracy: ', accu_per_batch[a].mean().item())

                        # save accuracies
                        if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                            for iC in range(len(contrasts)):
                                for t in range(t_steps):
                                    accu_current[iInit, iC, t] = accu_per_batch[:, iC, t].mean(0)
                        elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
                            for iL in range(len(layers)):
                                for t in range(t_steps):
                                    accu_current[iInit, iC, iL, t] = accu_per_batch[:, iC, iL, t].mean(0)

                # save accuracies
                np.save(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics, accu_current)

# import accuracy
for iA, current_analysis in enumerate(analysis):

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    contrasts, n_bin, depVar, n_samples = create_samples(current_analysis, batch_size)

    # ------------------------------------------------------------- SETTINGS FOR EACH ANALYSIS

    # initiate accuracy storage
    if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, len(contrasts), t_steps))
    elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
        accu = np.zeros((len(dataset), len(tempDynamics), init, len(contrasts), len(layers), t_steps))

    # retrieve metric values
    for iD, current_dataset in enumerate(dataset):
        for iT, current_tempDynamics in enumerate(tempDynamics):
            if (current_analysis == 'in_distribution') | (current_analysis == 'out_distribution_contrast'):
                accu[iD, iT, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
            elif (current_analysis == 'decoding') | (current_analysis == 'activations'):
                accu[iD, iT, :, :, :, :] = np.load(root + 'metrics/' + task + '/' + current_dataset + '_' + current_analysis + '_' + current_tempDynamics + '.npy')
    
    # visualize
    if current_analysis == 'in_distribution':
        visualize_in_distribution(task, accu, accu_repeated_noise, tempDynamics, current_analysis, dataset, init, root, t_steps)
    elif current_analysis == 'out_distribution_contrast':
        visualize_out_distribution_contrast(task, accu, tempDynamics, current_analysis, dataset, init, root, contrasts, t_steps)
    elif current_analysis == 'decoding':
        visualize_decoding(task, accu, tempDynamics, current_analysis, dataset, init, root, layers, t_steps)
    elif current_analysis == 'activations':
        visualize_activations(task, accu, tempDynamics, current_analysis, dataset, init, root, layers, t_steps, contrasts)


