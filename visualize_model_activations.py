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

# layer names
layers = ['sconv1', 'conv1', 'conv2', 'conv3', 'fc1']
layer = 'sconv1'
layer_idx = layers.index(layer)

# adapt = 'exp_decay'
adapt = 'div_norm'

# contrast and dataset
noise_patterns = ['same', 'different', 'no_adaptation']
noise = 'same'
contrast = 'lcontrast'
dataset = 'test'
plot = False
batchsiz = 1

t_steps = 15
dur = 10
start = [2]
        
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

# fig = plt.figure()
# plt.plot(noise_imgs[:, 0, 0, 0, 0], 'k')
# plt.show()

# initiate model
if (noise == 'same') | (noise == 'different'):
    # model_exp_decay = cnn_feedforward_exp_decay(t_steps=t_steps)
    model = cnn_feedforward_div_norm(batchsiz=batchsiz, t_steps=t_steps)
    # model_norm_div.load_state_dict(torch.load(dir+'Documents/code/nAdaptation_DNN/weights/weights_feedforward_' + adapt + '_' + noise + '_' + contrast + '.pth'))    
elif (noise == 'no_adaptation'):
    model = cnn_feedforward(t_steps=t_steps)
    model.load_state_dict(torch.load(dir+'weights/weights_feedforward_same_' + contrast + '.pth'))
print('Weights loaded!')

# fig = plt.figure()
# plt.title(idx)
print('noise img shape: ', noise_imgs.shape)
with torch.no_grad():

    # compute activations
    imgs_seq = []
    for t in range(t_steps):
        imgs_seq.append(noise_imgs[t, :, :, :, :]) # (t, b, c, w, h)

    # # forward sweep
    testoutp_exp_decay = model(imgs_seq, batch=False)

# # extract activations from proper layer
# testoutp_plot_exp_decay = testoutp_exp_decay[layer_idx]

# activations_exp_decay = torch.zeros(t_steps)
# activations = torch.zeros(t_steps)
# for t in range(t_steps):
#     activations_exp_decay[t] = torch.mean(testoutp_plot_exp_decay[t])
# # print(activations_exp_decay)

# # plot activatios
# plt.plot(activations_exp_decay, label=noise)
# print(activations_exp_decay)

# # adjust axis
# plt.ylabel('Model output (a.u.)')
# plt.xlabel('Model timesteps')
# plt.legend()

# # show plot
# fig.savefig(dir+'visualizations/activations_single_' + adapt + '_' + layer)
# plt.show()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('\nExecution time in seconds: ' + str(executionTime))


