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

from model_train_dataloader_novelty import PairsDataset

#@torch.jit.script

def sequence_train(imgs, task, *args):

    # # set the seed
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # initiate input
    ax = []

    if task == 'core':

        # number of timesteps
        n_timestep = 1

        # create sequence
        for t in range(n_timestep):
            ax.append(imgs)

    elif task == 'repeated_noise':

        # set contrasts
        contrasts       = torch.Tensor([0.2, 1])
        batch_size      = int(imgs.shape[0]/len(contrasts))

        # create noise pattern
        noise = torch.randn_like(imgs)*.2

        # number of timesteps
        n_timestep = 3

        # compute mean of test images
        mean_imgs = torch.mean(imgs, dim=[2, 3], keepdim=True).expand(imgs.shape)

        # adjust contrast
        imgs_adjusted = imgs.clone()
        for iC, contrast in enumerate(contrasts):
            imgs_adjusted[iC*batch_size:(iC+1)*batch_size, :, :, :] = (imgs[iC*batch_size:(iC+1)*batch_size, :, :, :] - mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]) * contrast # + mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]

        # add mean to noise
        noise_adjusted = (noise + mean_imgs)

        # create sequence
        for t in range(n_timestep):
            if t == 0:
                ax.append(noise_adjusted.clamp(0, 1))
            elif t == 1:
                blank = torch.ones_like(imgs) * mean_imgs
                ax.append(blank)
            elif t == 2:
                test = noise_adjusted + imgs_adjusted
                ax.append(test.clamp(0, 1))

    elif task == 'diffusion':

        # number of timesteps
        n_timestep = 10

        # compute mean of test images
        mean_imgs = torch.mean(imgs, dim=[2, 3], keepdim=True).expand(imgs.shape)

        # create weights
        weights = torch.linspace(0, 1, n_timestep)

        # create noise pattern
        noise = torch.randn_like(imgs)*.5

        for t in range(n_timestep):
            test = torch.multiply(1 - weights[t], mean_imgs + noise) + torch.multiply(weights[t], imgs)
            test = mean_imgs + (test - mean_imgs)
            ax.append(test.clamp(0, 1))

    elif task == 'occlusion':

        # compute mean of test images
        mean_per_image = imgs.mean(dim=[1, 2, 3], keepdim=True) 
        mean_imgs = mean_per_image.expand(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
        # print(mean_imgs[0, :, :, :])

        # initiate sequence storage
        ax = []

        # patch
        patch_width     = 12
        patch_height    = 12

        # height and with
        height  = imgs.shape[2]
        width   = imgs.shape[3]

        # create a tensor with values [10, 11, 12] to set the height
        min          = 10
        max          = 13
        values = torch.arange(min, max, dtype=torch.float32)
        indices = values.multinomial(num_samples=imgs.shape[0], replacement=True)
        # print(values)
        # print(indices)

        # occlude images
        for iB in range(imgs.shape[0]): # iterate over batch
            for t in range(int((width - patch_width)/2+1)):
                
                # replicate img tensor to store 
                imgs_occluded = imgs.clone()

                # apply the occlusion
                if iB == 0:
                    imgs_occluded[iB, :, int(values[indices[iB]].item()):int(values[indices[iB]].item())+patch_height, t*2:t*2+patch_width] = mean_imgs[iB, :, :patch_height, :patch_width]
                    ax.append(imgs_occluded)
                else:
                    ax[t][iB, :, int(values[indices[iB]].item()):int(values[indices[iB]].item())+patch_height, t*2:t*2+patch_width] = mean_imgs[iB, :, :patch_height, :patch_width]

        # normalize in the range [0, 1]
        for t in range(len(ax)):
            ax[t] = torch.clamp(ax[t], 0, 1)

    elif (task == 'novelty'):

        # labels
        lbls = args[0]

        # height and width
        h = imgs[0].shape[2]
        w = imgs[0].shape[3]
        c = imgs[0].shape[1]
        # print(c, h, w)

        # number of timesteps
        n_timestep      = 20

        # create tensor to store
        ax = torch.zeros(imgs[0].shape[0], n_timestep, c, w*2, h*2)
        lbls_t = torch.zeros(imgs[0].shape[0], n_timestep, dtype=int)

        # define quadrants
        quadrants_coord = np.array([[[0, h], [0, w]], 
                                   [[0, h], [w, 2*w]], 
                                   [[h, 2*h], [0, w]], 
                                   [[h, 2*h], [w, 2*w]]], dtype=int)
        

        # sample contrast
        contrast = np.random.uniform(0.2, 1, 4)
        # print(contrast)

        # mean imgs
        mean_imgs = []
        for iImg in range(len(imgs)):
            temp = torch.mean(imgs[iImg], dim=[2, 3], keepdim=True).expand(imgs[iImg].shape)
            mean_imgs.append(temp)
            # print(temp.shape)

        # precompute augmentations for each timestep
        augmentations = get_augmentations(n_timestep)

        # set count
        count_img = 0

        # sample timesteps
        t_add = np.random.choice(np.arange(1, n_timestep).tolist(), size=3, replace=False)
        # print(t_add)

        # initiate quadrant
        quadrants = [0, 1, 2, 3]

        # sample quadrant
        q = np.random.randint(0, len(quadrants), 1)[0]
        current_quadrant = quadrants[q]
        quadrants.remove(quadrants[q])

        # augment image
        if c == 3:
            img = (imgs[count_img] - mean_imgs[count_img]) * contrast[count_img] + mean_imgs[count_img]
        else:
            img = (imgs[count_img] - mean_imgs[count_img]) * contrast[count_img]

        # add image
        ax[:, 0, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

        # add lbl
        lbls_t[:, 0] = lbls[count_img]

        # increment count
        count_img += 1

        for t in range(1, n_timestep):

            # clone previous image
            ax[:, t, :, :, :] = ax[:, t-1, :, :, :]#.clone()

            if t in np.array(t_add):

                # sample quadrant    
                q = np.random.randint(0, len(quadrants), 1)[0]
                current_quadrant = quadrants[q]
                quadrants.remove(quadrants[q])

                # augment image
                if c == 3:
                    img = (imgs[count_img] - mean_imgs[count_img]) * contrast[count_img] + mean_imgs[count_img]
                else:
                    img = (imgs[count_img] - mean_imgs[count_img]) * contrast[count_img]

                # add image
                ax[:, t, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

                # add lbl
                lbls_t[:, t] = lbls[count_img]

                # increment count
                count_img += 1

            else:

                # add lbl
                lbls_t[:, t] = lbls_t[:, t-1]

        # normalize in the range [0, 1]
        for t in range(n_timestep):
            ax[:, t, :, :, :] = torch.clamp(ax[:, t, :, :, :], 0, 1)

        # retrun
        return lbls_t, ax
    
    elif (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):

        # labels
        lbls = args[0]

        # height and width
        h = imgs[0].shape[2]
        w = imgs[0].shape[3]
        c = imgs[0].shape[1]
        # print(c, h, w)

        # number of timesteps
        n_timestep      = 20

        # create tensor to store
        ax = torch.zeros(imgs[0].shape[0], n_timestep, c, w*2, h*2)
        lbls_t = torch.zeros(imgs[0].shape[0], n_timestep, dtype=int)

        # define quadrants
        quadrants_coord = np.array([[[0, h], [0, w]], 
                                [[0, h], [w, 2*w]], 
                                [[h, 2*h], [0, w]], 
                                [[h, 2*h], [w, 2*w]]], dtype=int)
        

        # mean imgs
        mean_imgs = []
        for iImg in range(len(imgs)):
            temp = torch.mean(imgs[iImg], dim=[2, 3], keepdim=True).expand(imgs[iImg].shape)
            mean_imgs.append(temp)
            # print(temp.shape)

        # precompute augmentations for each timestep
        augmentations = get_augmentations(n_timestep)

        # set count
        count_img = 0

        # sample timesteps
        t_add = np.random.choice(np.arange(1, n_timestep).tolist(), size=3, replace=False)
        # print(t_add)

        # initiate quadrant
        quadrants = [0, 1, 2, 3]
        quadrants_order = []

        # sample quadrant
        q = np.random.randint(0, len(quadrants), 1)[0]
        current_quadrant = quadrants[q]
        quadrants_order.append(current_quadrant)
        quadrants.remove(quadrants[q])

        # select and adjust the current image
        img = imgs[0]
        img = augmentations[0](img)

        # add image
        ax[:, 0, :, quadrants_coord[current_quadrant][0][0]:quadrants_coord[current_quadrant][0][1], quadrants_coord[current_quadrant][1][0]:quadrants_coord[current_quadrant][1][1]] = img

        # add lbl
        lbls_t[:, 0] = lbls[count_img]

        # increment count
        count_img += 1

        for t in range(1, n_timestep):

            if t in np.array(t_add):

                # sample quadrant
                q = np.random.randint(0, len(quadrants), 1)[0]
                current_quadrant = quadrants[q]
                quadrants_order.append(current_quadrant)
                quadrants.remove(quadrants[q])

                # add lbl
                lbls_t[:, t] = lbls[count_img]

                # increment count
                count_img += 1

                # adjust frame
                for i in range(count_img):

                    # first image
                    img = imgs[i]
                    img = augmentations[t](img)

                    # add image
                    ax[:, t, :, quadrants_coord[quadrants_order[i]][0][0]:quadrants_coord[quadrants_order[i]][0][1], quadrants_coord[quadrants_order[i]][1][0]:quadrants_coord[quadrants_order[i]][1][1]] = img

            else:

                for i in range(count_img):

                    # first image
                    img = imgs[i]
                    img = augmentations[t](img)  # apply augmentations

                    # add image
                    ax[:, t, :, quadrants_coord[quadrants_order[i]][0][0]:quadrants_coord[quadrants_order[i]][0][1], quadrants_coord[quadrants_order[i]][1][0]:quadrants_coord[quadrants_order[i]][1][1]] = img

                # add lbl
                lbls_t[:, t] = lbls_t[:, t-1]

        # normalize in the range [0, 1]
        for t in range(n_timestep):
            ax[:, t, :, :, :] = torch.clamp(ax[:, t, :, :, :], 0, 1)

        # retrun
        return lbls_t, ax
    
    return ax

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

def load_dataset(dataset, batch_size, task, train, download=False):

    # # set the seed
    # seed = 29
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # root
    root_data = './data'

    # settings
    download        = False
    shuffle         = True

    # transform
    transform = transforms.Compose([
    transforms.ToTensor()])

    # download data
    if dataset == 'mnist':
        trainData           = datasets.MNIST(root=root_data, train=True, download=download, transform=transform)
        testData            = datasets.MNIST(root=root_data, train=False, download=download, transform=transform)
    elif dataset == 'fmnist':
        trainData           = datasets.FashionMNIST(root=root_data, train=True, download=download, transform=transform)
        testData            = datasets.FashionMNIST(root=root_data, train=False, download=download, transform=transform)
    elif dataset == 'cifar':
        trainData           = torchvision.datasets.CIFAR10(root=root_data, train=True, download=download, transform=transform)
        testData            = torchvision.datasets.CIFAR10(root=root_data, train=False, download=download, transform=transform)

    # create dataloader
    if (task == 'novelty') | (task == 'novelty_augmented') | (task == 'novelty_augmented_extend_adaptation'):
        dataloader_custom = PairsDataset(trainData)
        ldrTrain = DataLoader(dataloader_custom, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        dataloader_custom = PairsDataset(testData)
        ldrTest = DataLoader(dataloader_custom, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        ldrTrain = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        ldrTest = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    # # print sizes
    # print('Test set: ', len(ldrTrain))
    # print('Training set: ', len(ldrTest))

    return ldrTrain, ldrTest

# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# custom summary
def custom_summary(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params": sum(p.numel() for p in module.parameters())
            }
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)

    with torch.no_grad():
        model(torch.zeros(1, *input_size))

    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print("----------------------------------------------------------------")
