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
layers = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
layer = 'conv1'
layer_idx = layers.index(layer)

# adapt = 'exp_decay'
layers = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
layer_exp_decay = 'conv1'
layer_idx_exp_decay = layers.index(layer_exp_decay)

# adapt = 'div_norm'
layers = ['conv1', 'sconv1', 'conv2', 'conv3', 'fc1', 'fc2']
layer_div_norm = 'sconv1'
layer_idx_div_norm = layers.index(layer_div_norm)

# contrast and dataset
noise_patterns = ['same', 'different', 'no_adaptation']
noise = 'same'
contrast = 'lcontrast'
dataset = 'test'
plot = False
batchsiz = 1

# timecourse
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

# initiate models
model = cnn_feedforward(t_steps=t_steps)
model.load_state_dict(torch.load(dir+'weights/weights_feedforward_' + noise + '_' + contrast + '.pth'))

model_exp_decay = cnn_feedforward_exp_decay(t_steps=t_steps)
model_exp_decay.load_state_dict(torch.load(dir+'weights/weights_feedforward_exp_decay_' + noise + '_' + contrast + '.pth'))    

model_div_norm = cnn_feedforward_div_norm(batchsiz=batchsiz, t_steps=t_steps)
# model_div_norm.load_state_dict(torch.load(dir+'weights/weights_feedforward_div_norm_' + noise + '_' + contrast + '.pth'))    

# model prediction
print('Input shape: ', noise_imgs.shape) 
with torch.no_grad():

    # compute activations
    imgs_seq = []
    for t in range(t_steps):
        imgs_seq.append(noise_imgs[t, :, :, :, :]) # (t, b, c, w, h)

    # # forward sweep
    testoutp = model(imgs_seq, batch=False)
    testoutp_exp_decay = model_exp_decay(imgs_seq, batch=False)
    testoutp_div_norm = model_div_norm(imgs_seq, batch=False)

# extract activations from proper layer
testoutp_plot = testoutp[layer_idx]
testoutp_plot_exp_decay = testoutp_exp_decay[layer_idx_exp_decay]
testoutp_plot_div_norm = testoutp_div_norm[layer_idx_div_norm]

# extract activations
activations = torch.zeros(t_steps)
activations_exp_decay = torch.zeros(t_steps)
activations_div_norm = torch.zeros(t_steps)
for t in range(t_steps):
    activations[t] = torch.mean(testoutp_plot[t])
    activations_exp_decay[t] = torch.mean(testoutp_plot_exp_decay[t])
    activations_div_norm[t] = torch.mean(testoutp_plot_div_norm[t])

# plot activatios
fig = plt.figure()

plt.plot(activations, label = 'no adaptation, ' + layer)
plt.plot(activations_exp_decay, label = 'exp. decay, ' + layer_exp_decay)
plt.plot(activations_div_norm, label = 'div. norm., ' + layer_div_norm)

# adjust axis
plt.ylabel('Model output (a.u.)')
plt.xlabel('Model timesteps')
plt.legend()

# show plot
fig.savefig(dir+'visualizations/activations')
plt.show()

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('\nExecution time in seconds: ' + str(executionTime))


