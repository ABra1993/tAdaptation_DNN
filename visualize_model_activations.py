# %%

import torch
from torchsummary import summary
import time
import matplotlib.pyplot as plt

print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from models.cnn_feedforward_div_norm import cnn_feedforward_div_norm
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
layer = 'conv1'
layer_idx = layers.index(layer)

# adapt = 'exp_decay'
layers_exp_decay = ['conv1', 'conv2', 'conv3', 'fc1']
layer_exp_decay = 'conv1'
layer_idx_exp_decay = layers_exp_decay.index(layer_exp_decay)

# adapt = 'div_norm'
layers_div_norm = ['conv1', 'sconv1', 'conv2', 'conv3', 'fc1']
layer_div_norm = 'conv1'
layer_idx_div_norm = layers_div_norm.index(layer_div_norm)

# contrast and dataset
noise_patterns = ['same', 'different', 'no_adaptation']
noise = 'same'
contrast = 'lcontrast'
dataset = 'test'
plot = False
batchsiz = 1

colors = ['crimson', 'deepskyblue', 'orange']
linestyles = ['dotted', (0, (5, 10)), 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]

# timecourse
t_steps = 150
dur = 40
start = [20, 90]

# t_steps = 15
# dur = 10
# start = [2]

# t_steps = 3
# dur = 1
# start = [0, 2]

# initiate figure
fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        
# select random img
idx = torch.randint(10000, (1,))
print('Index image number: ', idx)

print(30*'#')
print(noise)
print(30*'#')

# load stimuli
if noise == 'no_adaptation':
    noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/same_' + dataset + '_imgs_' + contrast)
    noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/same_' + dataset + '_lbls_' + contrast)
elif (noise == 'same') | (noise == 'different'):
    noise_imgs = torch.load(dir+'datasets/noiseMNIST/data/' + noise + '_' + dataset + '_imgs_' + contrast)
    noise_lbls = torch.load(dir+'datasets/noiseMNIST/data/' + noise + '_' + dataset + '_lbls_' + contrast)
dt = noiseMNIST_dataset(noise_imgs, noise_lbls)
print('Shape training set: ', noise_imgs.shape, ', ', noise_lbls.shape)

# create stimuli across timepoints
noise_imgs, noise_lbls = create_stim_timecourse(dt, idx, t_steps, dur, start, noise_lbls)
if plot:
    fig2, axs = plt.subplots(1, t_steps)
    for t in range(t_steps):    
        axs[t].imshow(noise_imgs[t, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[t].set_title('t =  ' + str(t+1), fontsize=5)
    plt.show()
    plt.close()

# initiate models
model = cnn_feedforward(t_steps=t_steps)
model.load_state_dict(torch.load(dir+'weights/weights_feedforward_' + noise + '_' + contrast + '.pth'))

model_exp_decay = cnn_feedforward_exp_decay(t_steps=t_steps)
model_exp_decay.load_state_dict(torch.load(dir+'weights/weights_feedforward_exp_decay_' + noise + '_' + contrast + '.pth'))    

model_div_norm = cnn_feedforward_div_norm(batchsiz=batchsiz, t_steps=t_steps)
# model_div_norm.load_state_dict(torch.load(dir+'weights/weights_feedforward_div_norm_' + noise + '_' + contrast + '.pth'))    
print('Models loaded!')

# model prediction
print('Input shape: ', noise_imgs.shape, '\n') 
with torch.no_grad():

    # compute activations
    imgs_seq = []
    for t in range(t_steps):
        imgs_seq.append(noise_imgs[t, :, :, :, :]) # (t, b, c, w, h)

    # # forward sweep
    testoutp = model(imgs_seq, batch=False)
    testoutp_exp_decay = model_exp_decay(imgs_seq, batch=False)
    testoutp_div_norm = model_div_norm(imgs_seq, batch=False)

# feedforward
for i in range(len(layers)):

    # extract activations
    activations = torch.zeros(t_steps)

    for t in range(t_steps):
        activations[t] = torch.mean(testoutp[i][t])

    # plot activations
    axs[0].plot(activations, color=colors[0], linestyle=linestyles[i], label=layers[i])
    
    # adjust axis
    axs[0].set_title('Feedforward')
    axs[0].set_ylabel('Model output (a.u.)')
    axs[0].set_xlabel('Model timesteps')
    axs[0].legend()

    if i == len(layers) - 1:
        print(activations)

# feedforward with exp. decay.
for i in range(len(layers_exp_decay)):

    # extract activations
    activations = torch.zeros(t_steps)

    for t in range(t_steps):
        activations[t] = torch.mean(testoutp_exp_decay[i][t])

    # plot activations
    axs[1].plot(activations, color=colors[1], linestyle=linestyles[i], label=layers_exp_decay[i])
    
    # adjust axis
    axs[1].set_title('Feedforward with exp. decay')
    axs[1].set_xlabel('Model timesteps')
    axs[1].legend()

    if i == len(layers_exp_decay) - 1:
        print(activations)

# feedforward wiht div. norm.
for i in range(len(layers_div_norm)):

    # extract activations
    activations = torch.zeros(t_steps)

    for t in range(t_steps):
        activations[t] = torch.mean(testoutp_div_norm[i][t])

    # plot activations
    axs[2].plot(activations, color=colors[2], linestyle=linestyles[i], label=layers_div_norm[i])

    # adjust axis
    axs[2].set_title('Feedforward with div. norm.')
    axs[2].set_xlabel('Model timesteps')
    axs[2].legend()

    if i == len(layers_div_norm) - 1:
        print(activations)

# plot readouts
readout = list(testoutp.values())[-1]
readout_exp_decay = list(testoutp_exp_decay.values())[-1]
readout_div_norm = list(testoutp_div_norm.values())[-1]

# plot
axs[3].axvspan(noise_lbls - 0.5 + 1, noise_lbls + 0.5 + 1, color='grey', alpha=0.2, label='Ground truth')
axs[3].scatter(torch.arange(10)+1, readout, label = 'no adaptation', color=colors[0])
axs[3].scatter(torch.arange(10)+1, readout_exp_decay, label = 'exp. decay', color=colors[1])
axs[3].scatter(torch.arange(10)+1, readout_div_norm, label = 'div. norm.', color=colors[2])

# adjust axis
axs[3].set_title('Decoder')
axs[3].set_xlabel('Classes')
axs[3].legend(bbox_to_anchor=(1, 0.5))

# show plot
fig.savefig(dir+'visualizations/activations_readouts')
plt.show()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('\nExecution time in seconds: ' + str(executionTime))


