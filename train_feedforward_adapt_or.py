# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from models.cnn_feedforward_div_norm import cnn_feedforward_div_norm
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

# start time
startTime = time.time()

# define root
dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

# track model training on neptune
run_init = True
random_init = 30

# set hypterparameters
numepchs = 1
batchsiz = 100
lr = 0.001
sample_rate = 64

# noise pattern
contrast = 'lcontrast'
# adapt = 'exp_decay'
adapt = 'div_norm'

# load training set
imgs = torch.load(dir+'datasets/noiseMNIST/data/train_imgs_' + contrast)
lbls = torch.load(dir+'datasets/noiseMNIST/data/train_lbls_' + contrast)
traindt = noiseMNIST_dataset(imgs, lbls)
print('Shape training set: ', imgs.shape, ', ', lbls.shape)

# load test set
imgs = torch.load(dir+'datasets/noiseMNIST/data/test_imgs_' + contrast)
lbls = torch.load(dir+'datasets/noiseMNIST/data/test_lbls_' + contrast)
testdt = noiseMNIST_dataset(imgs, lbls)
print('Shape test set: ', imgs.shape, ', ', lbls.shape)

# dataloader
ldrs = load_data(traindt, testdt, batch_size=batchsiz, shuffle=True, num_workers=1)
print('\nNumber of training batches: ', len(ldrs['train']), '\n')
print('\nNumber of test batches: ', len(ldrs['test']), '\n')

# determine number of timesteps
t_steps = len(imgs[0, :, 0, 0])
print('\nNumber of timesteps: ', t_steps)

# initiate pandas dataframe to store initial values
df = pd.DataFrame(columns=['Init', 'tau1_init', 'tau2_init', 'sigma_init'])

# run n random initializations
accuracies = torch.zeros(random_init)
for i in range(random_init):

    print(30*'-')
    print('Random init: ', i+1)

    # choose initial values DN model
    tau1_init = torch.rand(1)
    tau2_init = torch.rand(1)
    sigma_init = torch.rand(1)

    df.loc[i, 'Init'] = 1
    df.loc[i, ['tau1_init', 'tau2_init', 'sigma_init']] = [tau1_init, tau2_init, sigma_init]
    print(df)

    if run_init: # track training with Neptune
        run = neptune.init(
            project="abra1993/adapt-dnn",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
        )  # your credentials

        params = {"name:": adapt + '-' + '-' + contrast + '-' + str(i), "learning_rate": lr} 
        run["parameters"] = params

    # initiate model
    if adapt == 'exp_decay':
        model = cnn_feedforward_exp_decay(t_steps=t_steps)
    elif adapt == 'div_norm':
        model = cnn_feedforward_div_norm(tau1_init, tau2_init, sigma_init, batchsiz=batchsiz, t_steps=t_steps, sample_rate=sample_rate)
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

    print(30*'--')
    print('Mean accuracy: ', torch.round(accuracies[i], decimals=2))
    print(30*'--')

    # save model
    if accuracies[i] > 0.2: # above chance level
        next(model.parameters()).device
        if adapt == 'exp_decay':
            torch.save(model.state_dict(), dir+'/weights/weights_feedforward_' + adapt  + '_' + contrast + '.pth')
        elif adapt == 'div_norm':
            torch.save(model.state_dict(), dir+'weights/weights_feedforward_' + adapt +  '_' + contrast + '.pth')
        print('Weights saved!')

        # save accuracies
        if adapt == 'exp_decay':
            torch.save(accuracies, 'accu/feedforward_' + adapt + '_' + contrast)
        elif adapt == 'div_norm':
            torch.save(accuracies, 'accu/feedforward_' + adapt + '_' + contrast)

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
