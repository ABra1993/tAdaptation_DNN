import torch as torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torchvision.transforms as T

def sequence_test(imgs, task, current_analysis, lbls, current_dataset, *args):

    # height and width
    h = imgs[0].shape[2]
    w = imgs[0].shape[3]
    c = imgs[0].shape[1]
    # print(c, h, w)

    # number of timesteps
    n_timestep      = args[0]

    # define quadrants
    quadrants_coord = np.array([[[0, h], [0, w]], 
                                [[0, h], [w, 2*w]], 
                                [[h, 2*h], [0, w]], 
                                [[h, 2*h], [w, 2*w]]], dtype=int)
    
    # # initiate figure
    # fig, axs = plt.subplots(1, n_timestep)

    # create inputs
    if current_analysis == 'single_image':
        if task == 'novelty':
            
            # initiate
            ax = torch.zeros(imgs[0].shape[0], c, w*2, h*2)

            # set count
            count_img = 0

            # sample quadrant
            quadrants = [0, 1, 2, 3]
            q = np.random.randint(0, len(quadrants), 1)[0]
            current_quadrant = quadrants[q]

            # add image
            ax[:, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = imgs[count_img]

            # add image
            ax = ax.repeat(n_timestep, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
            lbls_t = lbls[count_img].repeat(n_timestep, 1).permute(1, 0)

        elif task == 'novelty_augmented':

            # initiate dataframe
            ax = torch.zeros(imgs[0].shape[0], n_timestep, c, w*2, h*2)
            lbls_t = torch.zeros(imgs[0].shape[0], n_timestep, dtype=int)

            # set count
            count_img = 0

            # precompute augmentations for each timestep
            augmentations = get_augmentations(n_timestep)

            # initiate quadrant
            quadrants = [0, 1, 2, 3]
            quadrants_order = []

            # sample quadrant
            q = np.random.randint(0, len(quadrants), 1)[0]
            current_quadrant = quadrants[q]
            quadrants_order.append(current_quadrant)
            quadrants.remove(quadrants[q])
                
            # select and adjust the current image
            img = imgs[0].clone()  # copy the original image to avoid modifying it
            img = augmentations[0](img)  # apply augmentations

            # add image
            ax[:, 0, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

            for t in range(1, n_timestep):

                # first image
                img = imgs[0].clone()  # copy the original image to avoid modifying it
                img = augmentations[t](img)  # apply augmentations

                # add image
                ax[:, t, :, quadrants_coord[quadrants_order[0]][0][0]:quadrants_coord[quadrants_order[0]][0][1], quadrants_coord[quadrants_order[0]][1][0]:quadrants_coord[quadrants_order[0]][1][1]] = img

            # add labels
            lbls_t = lbls[count_img].repeat(n_timestep, 1).permute(1, 0)

        # # visualize
        # for t in range(n_timestep):
        #     axs[t].imshow(ax[0, t, :, :, :].permute(1, 2, 0), cmap='gray')
        #     axs[t].axis('off')
        # plt.savefig('visualization/' + task + '/' + current_analysis + '_input_' + current_dataset)
        # plt.savefig('visualization/' + task + '/' + current_analysis + '_input' + current_dataset + '.svg')
        # plt.close()
    
        # retrun
        return lbls_t, ax
        
    elif (current_analysis == 'onset_accu') | (current_analysis == 'onset_activations'):

        # retrieve onset
        onset = args[1]
            
        # initiate dataframe
        ax = torch.zeros(imgs[0].shape[0], n_timestep, c, w*2, h*2)
        lbls_t = torch.zeros(imgs[0].shape[0], n_timestep, dtype=int)

        # sample quadrants
        qdr_idx = random.sample([0, 1, 2, 3], 2)
        # qdr_idx = [0, 2]

        # set count
        count_img = 0

        # precompute augmentations for each timestep
        if task == 'novelty_augmented':
            augmentations = get_augmentations(n_timestep)

        # add image
        current_quadrant = qdr_idx[0]
        ax[:, 0, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = imgs[count_img]

        # add lbl
        lbls_t[:, 0] = lbls[count_img]

        # increment count
        count_img += 1

        for t in range(1, n_timestep):

            # clone previous image
            if task == 'novelty':
                ax[:, t, :, :, :] = ax[:, t-1, :, :, :]#.clone()
            else:

                # update quadrant
                current_quadrant = qdr_idx[0]

                # first image
                img = imgs[0]
                img = augmentations[t](img)  # apply augmentations

                # add image
                ax[:, t, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

            # add second image
            if t >= onset:
                    
                # update quadrant
                current_quadrant = qdr_idx[1]

                # first image
                img = imgs[1].clone()
                if task == 'novelty_augmented':
                    img = augmentations[t](img)  # apply augmentations

                # add image
                ax[:, t, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

                # add lbl
                lbls_t[:, t] = lbls[1]
            
            else:

                # add lbl
                lbls_t[:, t] = lbls[0]

        # # visualize
        # for t in range(n_timestep):
        #     axs[t].imshow(ax[0, t, :, :, :].permute(1, 2, 0), cmap='gray')
        #     axs[t].axis('off')
        # plt.savefig('visualization/' + task + '/' + current_analysis + '_input')
        # plt.savefig('visualization/' + task + '/' + current_analysis + '_input.svg')
        # plt.close()
    
        # retrun
        return lbls_t, ax, qdr_idx

def get_augmentations(n_timestep):
    augmentations = []
    for _ in range(n_timestep):
        augmentation = T.Compose([
            T.RandomRotation(degrees=(-12, 12)),  # Use a tuple for the range
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            T.GaussianBlur(3)
        ])
        augmentations.append(augmentation)
    return augmentations

def create_samples(current_analysis, batch_size):

    if current_analysis == 'in_distribution':
        return [0], None, None, None
    elif current_analysis == 'single_image':
        return [0], None, None, None
    elif current_analysis == 'onset_accu':
        onsets = np.array([3, 6, 12], dtype=int)
        return onsets, None, None, None
    elif current_analysis == 'onset_activations':
        onsets = np.array([3, 6, 12], dtype=int)
        return onsets, None, None, None
    elif current_analysis == 'intervention':
        onsets = np.array([5], dtype=int)
        return onsets, None, None, None