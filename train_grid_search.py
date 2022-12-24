# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# simulate model and save activations
idx = torch.randint(10000, (1,))
print('Index image number: ', int(idx[0]))

# track model training on neptune
run_init = False

# set hypterparameters
random_init =               300
runs =                      3
numepchs =                  1
batchsiz =                  64
lr =                        0.0001
contrast =                  'lcontrast'
sample_rates =              [16, 32, 64, 128, 256, 512]
noise_patterns =            ['same', 'different']

# define number of timesteps
t_steps = 10
dur_train = [5, 3]
start_train = [1, 7]
print('\nNumber of timesteps: ', t_steps, '\n')

# load training set
noise_same_imgs_train = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_same_train_imgs_' + contrast)
noise_same_lbls_train = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_same_train_lbls_' + contrast)
traindt_same = noiseMNIST_dataset(noise_same_imgs_train, noise_same_lbls_train)
print('Shape training set (same):'.ljust(50), noise_same_imgs_train.shape, ', ', noise_same_lbls_train.shape)

# load test set
noise_same_imgs_test = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_same_test_imgs_' + contrast)
noise_same_lbls_test = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_same_test_lbls_' + contrast)
testdt_same = noiseMNIST_dataset(noise_same_imgs_test, noise_same_lbls_test)
print('Shape test set (same):'.ljust(50), noise_same_imgs_test.shape, ', ', noise_same_lbls_test.shape)

ldrs_same = load_data(traindt_same, testdt_same, batch_size=batchsiz, shuffle=True, num_workers=1)

# load training set
noise_different_imgs_train = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_different_train_imgs_' + contrast)
noise_different_lbls_train = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_different_train_lbls_' + contrast)
traindt_different = noiseMNIST_dataset(noise_different_imgs_train, noise_different_lbls_train)
print('Shape training set (different):'.ljust(50), noise_different_imgs_train.shape, ', ', noise_different_lbls_train.shape)

# load test set
noise_different_imgs_test = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_different_test_imgs_' + contrast)
noise_different_lbls_test = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_different_test_lbls_' + contrast)
testdt_different = noiseMNIST_dataset(noise_different_imgs_test, noise_different_lbls_test)
print('Shape test set (different):'.ljust(50), noise_different_imgs_test.shape, ', ', noise_different_lbls_test.shape)

ldrs_different = load_data(traindt_different, testdt_different, batch_size=batchsiz, shuffle=True, num_workers=1)

# adapt = 'exp_decay'
# layers_plot_exp_decay = ['conv1', 'conv2', 'conv3', 'fc1']

adapt = 'div_norm'
layers= ['conv1', 'sconv1', 'conv2', 'conv3', 'fc1']

cmap = plt.get_cmap('plasma')
alpha = np.linspace(0, 1, len(layers))

train_tau1 = False
train_tau2 = False
train_sigma = False

# initiate pandas dataframe to store initial values
runs_labels = []
for run in range(runs):
    runs_labels.append('acc' + str(run+1))
df = pd.DataFrame(columns=['Init', 'sample_rate', 'tau1_init', 'tau2_init', 'sigma_init', 'acc'] + runs_labels)

# run n random initializations
accuracies = torch.zeros(random_init)
for i in range(random_init):

    # initiate dataframe to store accuracies
    accuracies = torch.zeros(random_init)

    # choose initial values DN model
    tau1_init = torch.rand(1)
    tau2_init = torch.rand(1)
    sigma_init = torch.rand(1)
    sample_rate = resample(sample_rates, replace=True, n_samples=1)[0]
    noise = resample(noise_patterns, replace=True, n_samples=1)[0]

    # tau1_init = torch.Tensor([0.0075])
    # tau2_init = torch.Tensor([0.3741])
    # sigma_init = torch.Tensor([0.1711])
    # sample_rate = 256
    # noise = 'same'

    df.loc[i, 'Init'] = i+1
    df.loc[i, ['sample_rate', 'tau1_init', 'tau2_init', 'sigma_init']] = [sample_rate, float(tau1_init[0]), float(tau2_init[0]), float(sigma_init[0])]

    for j in range(runs):

        if run_init: # track training with Neptune
            run = neptune.init(
                project="abra1993/adapt-dnn",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
            )  # your credentials

            params = {"name:": adapt + '-' + '-' + contrast + '-' + str(i), "learning_rate": lr, 'train_tau1': train_tau1, 'train_tau2': train_tau2, 'train_sigma': train_sigma} 
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
            if noise == 'same':
                losses, run = train(numepchs, model, ldrs_same, lossfunct, optimizer, batchsiz, t_steps, run=run, rand_init=i, run_num=j)
            else:
                losses, run = train(numepchs, model, ldrs_different, lossfunct, optimizer, batchsiz, t_steps, run=run, rand_init=i, run_num=j)

        else:
            if noise == 'same':
                run = train(numepchs, model, ldrs_same, lossfunct, optimizer, batchsiz, t_steps, rand_init=i, run_num=j)
            else:
                run = train(numepchs, model, ldrs_different, lossfunct, optimizer, batchsiz, t_steps, rand_init=i, run_num=j)

        # # test model
        if run_init:
            if noise == 'same':
                accu, run = test(model=model, ldrs=ldrs_same, t_steps=t_steps, batch_size=batchsiz, run=run, rand_init=i, run_num=j)
            else:
                accu, run = test(model=model, ldrs=ldrs_different, t_steps=t_steps, batch_size=batchsiz, run=run, rand_init=i, run_num=j)

        else:
            if noise == 'same':
                accu = test(model=model, ldrs=ldrs_same, t_steps=t_steps, batch_size=batchsiz, rand_init=i, run_num=j)
            else:
                accu = test(model=model, ldrs=ldrs_different, t_steps=t_steps, batch_size=batchsiz, rand_init=i, run_num=j)

        # save accuracies
        accuracies[j] = torch.mean(accu)

        # save neptune session
        if run_init:
            run.stop()

    # save accuracies
    sum = 0
    for j in range(len(runs_labels)):
        sum += float(accuracies[j])
        df[runs_labels[j]] = float(accuracies[j])
    df['acc'] = sum/runs
    df.to_csv('accu/grid_search/meta.txt', header=True, sep=' ')
    print(df)

    if torch.mean(accuracies) > 0.5:

        # save accuracies
        if adapt == 'exp_decay':
            torch.save(accuracies, 'accu/grid_search/'+ str(i+1) + '_feedforward_' + adapt + '_' + noise + '_' + contrast)
        elif adapt == 'div_norm':
            torch.save(accuracies, 'accu/grid_search/'+ str(i+1) + '_feedforward_' + adapt + '_' + noise + '_' + contrast)

        # prepare image sequence
        imgs_seq = []
        for t in range(t_steps):
            if noise == 'same':
                imgs_seq.append(noise_same_imgs_test[idx, t, :, :, :])
            else:
                imgs_seq.append(noise_different_imgs_test[idx, t, :, :, :])

        # make model prediction
        model.eval()
        with torch.no_grad():
            testoutp_div_norm = model(imgs_seq)

        # plot activations
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title('noise, {}: sample rate, {}; accuracy, {}; \n tau1, {}; tau2, {}; sigma, {}'.format('noise', sample_rate, np.round(float(torch.mean(accuracies)), 4), np.round(float(tau1_init[0]), 4), np.round(float(tau2_init[0]), 4), np.round(float(sigma_init[0]), 4)))
        ax.set_xlabel('Model step')
        ax.set_xticks(np.arange(t_steps)+1)
        ax.set_ylabel('Model activations (a.u.)')

        # plot stimulus
        plt.axvspan(start_train[0], start_train[0]+dur_train[0], color='grey', alpha=0.05)
        plt.axvspan(start_train[1], start_train[1]+dur_train[1], color='grey', alpha=0.05)

        # extract activations
        activations = torch.zeros((len(layers), t_steps))
        for j in range(len(layers)):
            for t in range(t_steps):
                activations[j, t] = torch.nanmean(testoutp_div_norm[j][t])

            # plot activations
            plt.plot(np.arange(t_steps)+1, activations[j, :], color=cmap(alpha[j]), label=layers[j])

        # save figure
        plt.legend()
        plt.savefig('visualizations/grid_search/' + str(i+1), dpi=300)
        plt.close()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
