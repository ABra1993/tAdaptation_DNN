import torch
import torchvision

from torch.utils.data import DataLoader
import neptune.new as neptune
import torch.nn as nn
from torch import optim
from torchsummary import summary

# models
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_novelty import cnn_feedforward_novelty
from model_train_utils import *
from visualize_utils import *

# # set the seed
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '' ## ADD home directory

# temporal dynamics
tempDynamics        = ['none', 'add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics        = ['add_supp', 'div_norm', 'l_recurrence_A', 'l_recurrence_M']
# tempDynamics = ['l_recurrence_A', 'l_recurrence_M']
tempDynamics = ['div_norm']

# dataset
# dataset = ['mnist', 'fmnist', 'cifar']
# dataset = ['mnist', 'fmnist', 'cifar']
dataset = ['mnist', 'fmnist']

# tasks = ['core', 'repeated_noise', 'diffusion', 'occlusion', 'novelty']
# tasks = ['repeated_noise']
# tasks = ['diffusion']
# tasks = ['occlusion']
tasks = ['novelty', 'novelty_augmented']
# tasks = ['novelty_augmented']
# tasks = ['novelty_augmented_extend_adaptation']

# hyperparameter specification
init            = 1
batch_size      = 64
lr              = 0.001

# print summary
print(30*'-')
print('Temporal dynamci(s): '.ljust(30), tempDynamics)
print('Dataset(s): '.ljust(30), dataset)
print('Task(s): '.ljust(30), tasks)
print(30*'-')

# training
for current_dataset in dataset:

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

    for task in tasks:

        # set number of epochs
        if (task == 'novelty') | (task == 'novelty_augmented')  | (task == 'novelty_augmented_extend_adaptation'):
            n_epochs        = 10
        else:
            n_epochs        = 20

        # initaite dataloaders
        ldrTrain, ldrTest = load_dataset(current_dataset, batch_size, task, train=True)

        # training + inference
        for _, current_tempDynamics in enumerate(tempDynamics):

            # initiate dataframe to store test performance
            accu_current = np.zeros((init))

            for iInit in range(init):

                # initiate model
                if (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):
                    model = cnn_feedforward_novelty(channels, task, current_dataset)
                else:
                    model = cnn_feedforward(channels, task, current_dataset)

                # initiate recurrence
                if tempDynamics != 'none':
                    model.initialize_tempDynamics(current_tempDynamics)
                model.to(device)

                # loss function and optimizer
                lossfunct = nn.CrossEntropyLoss()   
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # train
                model.train()

                for epoch in range(n_epochs): # images and labels
                    for a, (imgs, lbls) in enumerate(ldrTrain):

                        if (a != 0) | (epoch != 0):
                            break

                        # create sequence
                        if (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):
                            lbls_t, ax = sequence_train(imgs, task, lbls)
                        elif (task != 'repeated_noise') | (task != 'diffusion') | (task != 'occlusion'):
                            ax = sequence_train(imgs, task)

                        # visualize sequence
                        if a == 0:
                            if (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):
                                model.init_t_steps(ax.shape[1])
                                visualize_sequence(ax, imgs, lbls_t, classes, task, current_dataset)
                            else:
                                model.init_t_steps(len(ax))
                                visualize_sequence(ax, imgs, lbls, classes, task, current_dataset)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # compute step
                        # if (task == 'novelty_augmented'):
                        #     outp = model.forward(a)
                        # else:
                        #     outp = model.forward(ax)
                        outp = model.forward(ax)

                        # compute loss
                        if (task == 'core') | (task == 'repeated_noise') | (task == 'diffusion')  | (task == 'occlusion'):
                            losses = lossfunct(outp, lbls.to(device))
                        elif (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):
                            losses = 0
                            for t in range(lbls_t.shape[1]):
                                losses += lossfunct(outp[:, :, t], lbls_t[:, t].to(device))

                        # backprop and optimization
                        losses.backward() 
                        optimizer.step()

                        # print progress
                        if (a%100 == 0) & (a != 0):
                            print('Training...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- epoch ', epoch+1, '/', str(n_epochs).ljust(10), ' ----- batch ', a, '/', str(len(ldrTrain)).ljust(10), ' ----- loss: ', losses.detach().cpu().numpy())
                
                # save model
                torch.save(model.state_dict(), root + 'weights/' + task + '/' + task + '_' + current_tempDynamics + '_' + current_dataset + '_' + str(iInit+1))














            #     # inference
            #     model.eval()
            #     print('\n\n')

            #     # store accuracies
            #     accu_per_batch = np.zeros(len(ldrTest))

            #     # loop and retrieve accuracies
            #     for a, (imgs, lbls) in enumerate(ldrTest):

            #         # create sequence
            #         ax = sequence_train(imgs, task)
                    
            #         # validate
            #         testoutp = model.forward(ax)

            #         # compute accuraices
            #         if (task == 'core') | (task == 'repeated_noise') | (task == 'diffusion') | (task == 'occlusion'):
            #             predicy = torch.argmax(testoutp, dim=1).to('cpu')
            #             accu_per_batch[a] = (predicy == lbls).sum().item() / float(lbls.size(0))

            #         # save losses and print progress
            #         if (a%20 == 0) & (a != 0):
            #             print('Inference...      ----- dynamics ', current_tempDynamics.ljust(10), ' ----- dataset ', current_dataset.ljust(10), ' ----- task ', task.ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- epoch ', epoch+1, '/', str(n_epochs).ljust(10), ' ----- batch ', a, '/', str(len(ldrTest)).ljust(10), ' ----- accuracy: ', accu_per_batch[a].item())

            #     print('\n\n')

            #     # report accuracy
            #     accu_current[iInit] = np.mean(accu_per_batch)

            # # save performance
            # np.save(root + 'accu/' + task + '/' + current_tempDynamics + '_' + current_dataset, accu_current)


