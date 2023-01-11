# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

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

# 22   269       same   tensor([16.])   0.034268   0.540629    0.006827  0.722532  0.733280  0.718949  0.715366
# 23   269  different   tensor([16.])   0.034268   0.540629    0.006827  0.677979  0.688694  0.669984  0.675259

# set hypterparameters
contrast = 'lcontrast'
numepchs = 1
batchsiz = 64
lr = 0.0001
sample_rate = 16

# track model training on neptune (CHANGE THESE VALUES)
run_init = True
random_init = 1

save_weight = True
save_accu = False

# param = 'fixed'
param = 'trained'

# noise = 'different'
noise = 'same'

# adapt = 'exp_decay'
adapt = 'div_norm'

# define number of timesteps
t_steps = 10
print('\nNumber of timesteps: ', t_steps)

if param == 'trained':

    # div. norm.
    train_tau1 = True
    train_tau2 = True
    train_sigma = True

    # exp. decay.
    train_alpha = True
    train_beta = True

else:

    # div. norm.
    train_tau1 = False
    train_tau2 = False
    train_sigma = False

    # exp. decay.
    train_alpha = False
    train_beta = False

# load training set
noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + 'train_imgs_' + contrast)
noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + 'train_lbls_' + contrast)
traindt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape training set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# load test set
noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + 'test_imgs_' + contrast)
noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + 'test_lbls_' + contrast)
testdt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape test set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# dataloader
ldrs = load_data(traindt, testdt, batch_size=batchsiz, shuffle=True, num_workers=1)
print('\nNumber of training batches: ', len(ldrs['train']), '\n')
print('\nNumber of test batches: ', len(ldrs['test']), '\n')

# initiate pandas dataframe to store initial values
df = pd.DataFrame(columns=['Init', 'tau1_init', 'tau2_init', 'sigma_init'])

# run n random initializations
accuracies = torch.zeros(random_init)
for i in range(random_init):

    print(30*'-')
    print('Random init: ', i+1)

    # choose initial values DN model
    # tau1_init = torch.rand(1)
    # tau2_init = torch.rand(1)
    # sigma_init = torch.rand(1)
    tau1_init = torch.Tensor([0.034268])
    tau2_init = torch.Tensor([0.540629])
    sigma_init = torch.Tensor([0.006827])

    df.loc[i, 'Init'] = 1
    df.loc[i, ['tau1_init', 'tau2_init', 'sigma_init']] = [tau1_init, tau2_init, sigma_init]
    print(df)

    if run_init: # track training with Neptune
        run = neptune.init(
            project="abra1993/adapt-dnn",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
        )  # your credentials
        if adapt == 'div_norm':
            params = {"name:": adapt + '-' + '-' + contrast + '-' + str(i), "learning_rate": lr, 'train_tau1': train_tau1, 'train_tau2': train_tau2, 'train_sigma': train_sigma} 
        elif adapt == 'exp_decay':
            params = {"name:": adapt + '-' + '-' + contrast + '-' + str(i), "learning_rate": lr, 'train_alpha': train_alpha, 'train_beta': train_beta} 
        run["parameters"] = params

    # initiate model
    if adapt == 'exp_decay':
        model = cnn_feedforward_exp_decay(train_alpha, train_beta, t_steps=t_steps)
    elif adapt == 'div_norm':
        model = cnn_feedforward_div_norm(tau1_init, train_tau1, tau2_init, train_tau2, sigma_init, train_sigma, batchsiz=batchsiz, t_steps=t_steps, sample_rate=sample_rate)

    lossfunct = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=lr)   

    # train model
    if run_init:
        losses, run = train(numepchs, model, ldrs, lossfunct, optimizer, batchsiz, t_steps, run)
    else:
        run = train(numepchs, model, ldrs, lossfunct, optimizer, batchsiz, t_steps)

    # # test model
    if run_init:
        accu, run = test(model=model, ldrs=ldrs, t_steps=t_steps, batch_size=batchsiz, run=run)
    else:
        accu = test(model=model, ldrs=ldrs, t_steps=t_steps, batch_size=batchsiz)

    # save accuracies
    accuracies[i] = torch.mean(accu)

    # save neptune session
    if run_init:
        run.stop()

# save model weights
if save_weight:
    next(model.parameters()).device
    if adapt == 'exp_decay':
        torch.save(model.state_dict(), dir+'/weights/weights_feedforward_' + adapt + '_' + noise + '_' + contrast + '_' + param + '.pth')
    elif adapt == 'div_norm':
        torch.save(model.state_dict(), dir+'weights/weights_feedforward_' + adapt + '_' + noise + '_' + contrast + '_' + param + '.pth')
    print('Weights saved!')

# save accuracies
if save_accu:    
    if adapt == 'exp_decay':
        torch.save(accuracies, 'accu/feedforward_' + adapt + '_' + noise + '_' + contrast + '_' + param)
    elif adapt == 'div_norm':
        torch.save(accuracies, 'accu/feedforward_' + adapt + '_' + noise + '_' + contrast + '_' + param)

print(30*'--')
print('Mean accuracy: ', torch.round(torch.mean(accuracies), decimals=2))
print(30*'--')

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
