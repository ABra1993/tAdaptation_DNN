# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

# start time
startTime = time.time()

# define root
linux = True # indicates whether script is run on mac or linux
if linux:
    dir = '/home/amber/ownCloud/'
else:
    dir = '/Users/a.m.brandsuva.nl/surfdrive/'

# track model training on neptune
run_init = True
random_init = 1

# set hypterparameters
numepchs = 1
batchsiz = 100
lr = 0.001

# noise pattern
noise = 'different'
contrast = 'lcontrast'

# load training set
noise_imgs = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + 'train_imgs_' + contrast)
noise_lbls = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + 'train_lbls_' + contrast)
traindt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape training set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# load test set
noise_imgs = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + 'test_imgs_' + contrast)
noise_lbls = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/' + noise + '_' + 'test_lbls_' + contrast)
testdt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape test set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# dataloader
ldrs = load_data(traindt, testdt, batch_size=batchsiz, shuffle=True, num_workers=0)
print('\nNumber of training batches: ', len(ldrs['train']), '\n')
print('\nNumber of test batches: ', len(ldrs['test']), '\n')

# determine number of timesteps
t_steps = len(noise_imgs[0, :, 0, 0])
print('\nNumber of timesteps: ', t_steps)

# run n random initializations
accuracies = torch.zeros(random_init)
for i in range(random_init):

    print(30*'-')
    print('Random init: ', i+1)

    if run_init:
        run = neptune.init(
            project="abra1993/adapt-dnn",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
        )  # your credentials

        params = {"name:": 'adaptation-' + noise + '-' + contrast + '-' + str(i), "learning_rate": lr} 
        run["parameters"] = params

    # initiate model
    model = cnn_feedforward(t_steps=t_steps)
    # print(summary(model))

    lossfunct = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=lr)   

    # train model
    if run_init:
        losses, run = train(numepchs, model, ldrs, lossfunct, optimizer, batchsiz, t_steps, run)
    else:
        run = train(numepchs, model, ldrs, lossfunct, optimizer, batchsiz, t_steps)

    # # test model
    if run_init:
        accu, run = test(model, ldrs, t_steps, run=run)
    else:
        accu = test(model, ldrs, t_steps)

    # save accuracies
    accuracies[i] = torch.mean(accu)

    # save neptune session
    if run_init:
        run.stop()

# save model
torch.save(model.state_dict(), dir+'Documents/code/nAdaptation_DNN/weights/weights_feedforward_' + noise + '_' + contrast + '.pth')
print('Weights saved!')

# save accuracies
torch.save(accuracies, 'accu/feedforward_' + noise + '_' + contrast)

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))



