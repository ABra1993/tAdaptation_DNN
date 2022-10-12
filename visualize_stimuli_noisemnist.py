# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')

from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
import matplotlib.pyplot as plt

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *

def main():

    # define root
    linux = True # indicates whether script is run on mac or linux
    if linux:
        dir = '/home/amber/ownCloud/'
    else:
        dir = '/Users/a.m.brandsuva.nl/surfdrive/'

    # noise settings
    noise_patterns = ['same', 'different']
    contrast = 'lcontrast'

    # load SAME test set
    noise_same_imgs = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/same_test_imgs_' + contrast)
    noise_same_lbls = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/same_test_lbls_' + contrast)
    print('Shape test set: ', noise_same_imgs.shape, ', ', noise_same_lbls.shape)

    # load SAME test set
    noise_diff_imgs = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/different_test_imgs_' + contrast)
    noise_diff_lbls = torch.load(dir+'Documents/code/nAdaptation_DNN/datasets/noiseMNIST/data/different_test_lbls_' + contrast)
    print('Shape test set: ', noise_diff_imgs.shape, ', ', noise_diff_lbls.shape)

    # join images
    imgs = [noise_same_imgs, noise_diff_imgs]

    # define input shape
    input_shape = noise_same_imgs[0][0].shape
    w = input_shape[1]
    h = input_shape[2]
    c = input_shape[0]
    print('Input shape: ', input_shape)

    # select random img
    img_num = torch.randint(len(noise_same_lbls), (1,))
    # img_num = 0

    # determine number of timesteps
    t_steps = len(noise_same_imgs[0, :, 0, 0])
    print('\nNumber of timesteps: ', t_steps)

    # plot images
    for i, noise in enumerate(noise_patterns):

        print(noise)

        # initiate figure
        _, axs = plt.subplots(1, 3, figsize=(10, 3))
        # axs[1].set_title('Noise pattern: ' + noise)

        # select image
        img_current = torch.squeeze(imgs[i][img_num, :, :, :, :], 0)
        print(img_current.shape)

        for t in range(t_steps):
            axs[t].imshow(torch.squeeze(img_current[t, : , :, :], 0).reshape(w, h, c), cmap='gray', vmin=0, vmax=1)
        
        # show plot
        plt.tight_layout()
        plt.savefig(dir+'Documents/code/nAdaptation_DNN/visualizations/stim_' + noise + '_' + contrast)
        plt.show()

if __name__ == '__main__':
    main()