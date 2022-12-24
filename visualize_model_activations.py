# %%

import torch
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import numpy as np

print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from models.cnn_feedforward_div_norm import cnn_feedforward_div_norm
from models.cnn_feedforward_div_norm_rec import cnn_feedforward_div_norm_rec
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune

# add model to devic
device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 

# start time
startTime = time.time()

# define root
dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

# intrinsic suppression
layers = ['conv1', 'conv2', 'conv3', 'fc1']
layers_plot = ['fc1']
layers_plot = ['conv1', 'conv2', 'conv3', 'fc1']
# layer = 'conv1'
# layer_idx = layers.index(layer)

# adapt = 'exp_decay'
layers_exp_decay = ['conv1', 'conv2', 'conv3', 'fc1']
# layers_plot_exp_decay = ['fc1']
layers_plot_exp_decay = ['conv1', 'conv2', 'conv3', 'fc1']
# layer_exp_decay = 'conv1'
# layer_idx_exp_decay = layers_exp_decay.index(layer_exp_decay)

# adapt = 'div_norm'
layers_div_norm = ['conv1', 'conv2', 'conv3', 'fc1']
layers_plot_div_norm = ['fc1']
# layers_plot_div_norm = ['conv1', 'sconv1', 'conv2', 'conv3', 'fc1']
layers_plot_div_norm = ['conv1', 'conv2', 'conv3', 'fc1']
# layers_plot_div_norm = 'conv1'
# layer_idx_div_norm = layers_div_norm.index(layer_div_norm)

# contrast and dataset
noise_patterns = ['same', 'different', 'no_adaptation']
noise = 'same'
contrast = 'lcontrast'
dataset = 'test'
plot = True
batchsiz = 1
sample_rate = 256

colors = ['crimson', 'deepskyblue', 'orange']

# define number of timesteps
t_steps = 10
print('\nNumber of timesteps: ', t_steps)

linestyles = ['dotted', (0, (5, 10)), 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
linestyles_exp_decay = ['dotted', (0, (5, 10)), 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
linestyles_div_norm = ['dotted', (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10)), 'dashed', 'dashdot']

cmap = plt.get_cmap('plasma')
alpha = np.linspace(0, 1, len(layers))
alpha_exp_decay = np.linspace(0, 1, len(layers_exp_decay))
alpha_div_norm = np.linspace(0, 1, len(layers_div_norm))

# t_steps = 15
# dur = 10
# start = [2]

# t_steps_train = 10
# dur_train = 3
# start_train = [1, 7]

t_steps_train = 10
dur_train = [5, 3]
start_train = [1, 7]
        
# select random img
idx = torch.randint(10000, (1,))
print('Index image number: ', idx)
idx = 423

print(30*'#')
print(noise)
print(30*'#')

# load stimuli
if noise == 'no_adaptation':
    noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/same_' + dataset + '_imgs_' + contrast)
    noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/same_' + dataset + '_lbls_' + contrast)
elif (noise == 'same') | (noise == 'different'):
    noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_imgs_' + contrast)
    noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/' + str(t_steps) + '_' + noise + '_' + dataset + '_lbls_' + contrast)
dt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape training set: ', noise_imgs.shape, ', ', noise_lbls.shape)

tau1_init = torch.Tensor([0.0075])
tau2_init = torch.Tensor([0.3741])
sigma_init = torch.Tensor([0.1711])

# initiate models
model = cnn_feedforward(t_steps=t_steps)
model.load_state_dict(torch.load(dir+'weights/weights_feedforward_' + noise + '_' + contrast + '.pth'))

model_exp_decay = cnn_feedforward_exp_decay(t_steps=t_steps)
model_exp_decay.load_state_dict(torch.load(dir+'weights/weights_feedforward_exp_decay_' + noise + '_' + contrast + '.pth'))    

model_div_norm = cnn_feedforward_div_norm(tau1_init, tau2_init, sigma_init, batchsiz=batchsiz, t_steps=t_steps, sample_rate=sample_rate)
model_div_norm.load_state_dict(torch.load(dir+'weights/weights_feedforward_div_norm_' + noise + '_' + contrast + '.pth'))    

print('Models loaded!')

# model prediction
print('Input shape: ', noise_imgs.shape, '\n') 
imgs_seq = []
with torch.no_grad():

    # prepare image sequence
    for t in range(t_steps):
        imgs_seq.append(noise_imgs[idx, t, :, :, :])

    # forward sweep
    testoutp = model(imgs_seq, batch=False)
    print('Model done!')
    testoutp_exp_decay = model_exp_decay(imgs_seq, batch=False)
    print('Model done!')
    testoutp_div_norm = model_div_norm(imgs_seq, batch=False)
    print('Model done!')

# create stimuli across timepoints
if plot:
    fig2, axs = plt.subplots(1, t_steps)
    for t in range(t_steps):    
        axs[t].imshow(imgs_seq[t].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[t].axis('off')
        axs[t].set_title('t =  ' + str(t+1), fontsize=5)
    fig2.savefig(dir+'visualizations/stimulus_timecourse.svg', type='svg')
    plt.close()

# initiate figure
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

# feedforward
lw = 3
print('\n FF')
for i in range(len(layers)):
    if layers[i] in layers_plot:

        # extract activations
        activations = torch.zeros(t_steps)

        for t in range(t_steps):
            activations[t] = torch.mean(testoutp[i][t])

        # plot stimulus
        axs[0].axvspan(start_train[0], start_train[0]+dur_train[0], color='grey', alpha=0.05)
        axs[0].axvspan(start_train[1], start_train[1]+dur_train[1], color='grey', alpha=0.05)

        # plot activations
        axs[0].plot(activations, color=cmap(alpha[i]), label=layers[i], lw=lw) #, alpha=alpha[i] , linestyle=linestyles[i])
        
    # adjust axis
    axs[0].set_title('Feedforward')
    axs[0].set_ylabel('Model output (a.u.)')
    axs[0].set_xlabel('Model timesteps')
    axs[0].legend()
    # axs[0].set_ylim(0, 0.20)

    # if i == len(layers) - 1:
    # print(layers[i], ': ', activations)

# feedforward with exp. decay.
print('\n FF with exp. decay')
for i in range(len(layers_exp_decay)):
    if layers_exp_decay[i] in layers_plot_exp_decay:

        # extract activations
        activations = torch.zeros(t_steps)

        for t in range(t_steps):
            activations[t] = torch.mean(testoutp_exp_decay[i][t])

        # plot stimulus
        axs[1].axvspan(start_train[0], start_train[0]+dur_train[0], color='grey', alpha=0.05)
        axs[1].axvspan(start_train[1], start_train[1]+dur_train[1], color='grey', alpha=0.05)

        # plot activations
        axs[1].plot(activations, color=cmap(alpha_exp_decay[i]), label=layers_exp_decay[i], lw=lw) # , alpha=alpha_exp_decay[i], linestyle=linestyles[i])
        
    # adjust axis
    axs[1].set_title('Feedforward with exp. decay')
    axs[1].set_xlabel('Model timesteps')
    axs[1].legend()
    # axs[1].set_ylim(0, 0.20)

    # if i == len(layers_exp_decay) - 1:
    # print(layers_exp_decay[i], ': ', activations)

# feedforward wiht div. norm.
print('\n FF with div. norm')
for i in range(len(layers_div_norm)):
    if layers_div_norm[i] in layers_plot_div_norm:

        # extract activations
        activations = torch.zeros(t_steps)

        for t in range(t_steps):
            activations[t] = torch.nanmean(testoutp_div_norm[i][t])
            # activations[t] = testoutp_div_norm[i][t][0, 0, 0, 0]

        # plot stimulus
        axs[2].axvspan(start_train[0], start_train[0]+dur_train[0], color='grey', alpha=0.05)
        axs[2].axvspan(start_train[1], start_train[1]+dur_train[1], color='grey', alpha=0.05)

        # plot activations
        axs[2].plot(activations, color=cmap(alpha_div_norm[i]), label=layers_div_norm[i], lw=lw) #, alpha=alpha_div_norm[i]) , linestyle=linestyles[i])

    # adjust axis
    axs[2].set_title('Feedforward with div. norm.')
    axs[2].set_xlabel('Model timesteps')
    axs[2].legend()
    # axs[2].set_ylim(0, 0.20)

    # if i == len(layers_div_norm) - 1:
    # print(layers_div_norm[i], ': ', activations)

# plot readouts
readout = list(testoutp.values())[-1]
readout_exp_decay = list(testoutp_exp_decay.values())[-1]
readout_div_norm = list(testoutp_div_norm.values())[-1]

# plot
axs[3].axvspan(noise_lbls[idx] - 0.5 + 1, noise_lbls[idx] + 0.5 + 1, color='grey', alpha=0.2, label='Ground truth')
axs[3].scatter(torch.arange(10)+1, readout, label = 'no adaptation', c='white', edgecolors='black', marker='o', s=100)
axs[3].scatter(torch.arange(10)+1, readout_exp_decay, label = 'exp. decay', c='lightgrey', edgecolors='black', marker='v', s=100)
axs[3].scatter(torch.arange(10)+1, readout_div_norm, label = 'div. norm.', c='black', edgecolors='black', marker='s', s=100)

# adjust axis
axs[3].set_title('Decoder')
axs[3].set_xlabel('Classes')
axs[3].legend(bbox_to_anchor=(1, 0.5))

# show plot
fig.savefig(dir+'visualizations/activations_readouts')
fig.savefig(dir+'visualizations/activations_readouts.svg', format='svg')
# plt.show()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('\nExecution time in seconds: ' + str(executionTime))


