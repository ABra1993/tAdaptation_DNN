#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from utils import h1, h2, torch_convolve, torch_cross_val_norm, torch_cross_val

# define root
dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

def h1(tau1, t_steps, sample_rate, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    y = input * torch.exp(-input/tau1)

    return y

def h2(tau2, t_steps, sample_rate, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    y = torch.exp(-input/tau2)

    return y

# input
b = 1 # batch-size
c = 2 # number channels
w = 2 # height
h = 2 # width

# parameter settings (r1 = minimal value, r2 = maximal value)
r1 = 0.1
r2 = 0.1
tau1    = (r1 - r2) * torch.rand(c, w, h) + r2
tau1 = 0.1

r1 = 0.1
r2 = 0.1
tau2    = (r1 - r2) * torch.rand(c, w, h) + r2
tau2 = 0.1

r1 = 0.1
r2 = 0.1
sigma   = (r1 - r2) * torch.rand(c, w, h) + r2
sigma = 0.1

r1 = 2
r2 = 2
n    = (r1 - r2) * torch.rand(c, w, h) + r2
n = 1.5

# timecourse
sample_rate = 16
t_steps = 80
dur = 20
start = [20, 50]

# sample_rate = 1
# t_steps = 3
# dur = 1
# start = [0, 2]

x = torch.zeros([t_steps, b, c, w, h])
value = [1, 1]
for i in range(len(start)):
    x[start[i]:start[i]+dur, :, :, :, :] = value[i]

# create timepoints based on sample rate
t = np.arange(t_steps)/sample_rate

# compute inversed IRFs (used for cross-validation)
irf = h1(tau1, t_steps, sample_rate, b, c, w, h)
# irf_inv = irf.reshape(t_steps, b*c*w*h)
irf_inv = torch.flip(irf, [-1]).unsqueeze(0).unsqueeze(0)
print('IRF: ', irf_inv.shape)

irf_norm = h2(tau2, t_steps, sample_rate, b, c, w, h)
# irf_norm_inv = irf_norm.reshape(t_steps, b*c*w*h)
irf_norm_inv = torch.flip(irf_norm, [-1]).unsqueeze(0).unsqueeze(0)
print('IRF norm: ', irf_norm_inv.shape)

# prep activations
x = x.reshape(t_steps, b*c*w*h).transpose(0,1)
x_pad = F.pad(x, [t_steps, 0, 0, 0]).unsqueeze(1)
print('x: ', x.shape)
print('x_pad: ', x_pad.shape)

# convolution input drive
conv_input_drive = F.conv1d(x.unsqueeze(1), irf_inv, padding=t_steps-1, stride=1, groups=1)
conv_input_drive_clip = conv_input_drive[:, 0, 0:t_steps]
input_drive = torch.pow(torch.abs(conv_input_drive_clip), n)
print('conv1d: ', conv_input_drive_clip.shape)

# convolution normalisation response
conv_normrsp = F.conv1d(conv_input_drive_clip.unsqueeze(1), irf_norm_inv, padding=t_steps-1, stride=1, groups=1)
conv_normrsp_clip = conv_normrsp[:, 0, 0:t_steps]
normrsp = torch.add(torch.pow(torch.abs(conv_normrsp_clip), n), sigma)
print('conv1d norm: ', conv_input_drive_clip.shape)

# reshape
x = x.squeeze(1).transpose(0, 1).reshape(t_steps, b, c, w, h)

conv_input_drive = conv_input_drive_clip.transpose(0, 1).reshape(t_steps, b, c, w, h)
input_drive = input_drive.transpose(0, 1).reshape(t_steps, b, c, w, h)

conv_normrsp = conv_normrsp_clip.transpose(0, 1).reshape(t_steps, b, c, w, h)
normrsp = normrsp.transpose(0, 1).reshape(t_steps, b, c, w, h)

# DN model response
r = torch.div(input_drive, normrsp)

# plot canonical computations
fig, axs = plt.subplots(2, 4, figsize=(20, 5))
count = 0
lw = 1
for i in range(len(x[0, :, 0, 0, 0])):
    for j in range(len(x[0, 0, :, 0, 0])):
        for k in range(len(x[0, 0, 0, :, 0])):
            for l in range(len(x[0, 0, 0, 0, :])):

                # determine index
                if count > 3:
                    idx_row = 1
                else:
                    idx_row = 0
                idx_col = (count % 4)

                # plot impulse response functions for first batch
                if i == 0:

                    # # retrieve parameter values
                    # torch.set_printoptions(precision=2)
                    # tau1_current = str(torch.round(tau1[j, k, l], decimals=2))
                    # tau2_current = str(torch.round(tau2[j, k, l], decimals=2))
                    # sigma_current = str(torch.round(sigma[j, k, l], decimals=2))
                    # n_current = str(torch.round(n[j, k, l], decimals=2))

                    # # set title
                    # axs[idx_row, idx_col].set_title(r'$\tau_{1} = $' + tau1_current + r', $\tau_{2} = $' +  tau2_current + r', $\sigma = $' +  sigma_current + r', $n = $' +  n_current, fontsize=6)
                    axs[idx_row, idx_col].set_title(r'$\tau_{1} = $' + str(tau1) + r', $\tau_{2} = $' +  str(tau2) + r', $\sigma = $' +  str(sigma) + r', $n = $' +  str(n), fontsize=6)

                    # activation
                    # axs[idx_row, idx_col].plot(x[:, i, j, k, l], 'k', label='$S/a_{c,h,w}$', lw=lw, alpha=0.7)
                    for m in range(len(start)):
                        axs[idx_row, idx_col].axvspan(start[m], start[m]+dur, color='grey', alpha=0.1)
                    
                    # impulse response functiona
                    axs[idx_row, idx_col].plot(t-max(t), irf_inv[0, 0, :], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')

                    axs[idx_row, idx_col].plot(conv_input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
                    axs[idx_row, idx_col].plot(input_drive[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')

                    axs[idx_row, idx_col].plot(conv_normrsp[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='red')
                    axs[idx_row, idx_col].plot(normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
                    
                    axs[idx_row, idx_col].plot(r[:, i, j, k, l], label=r'$r_{DN}$', color='red')

                # increment count
                count+=1

axs[0,0].legend(fontsize=8)
plt.tight_layout()

file = 'dn_conv1d'
plt.savefig(dir+'visualizations/DN_in_DNN/' + file)
plt.show()

# def torch_cross_val_norm(x, y, b, c, w, h, sigma, n, t_steps):

#     # preprocess
#     x_resh = x.reshape(t_steps, b*c*w*h)
#     y_resh = y.reshape(t_steps, b*c*w*h)

#     # fix stimulus
#     x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

#     # reshape sigma
#     n_resh = n.reshape(c*w*h).repeat(b)
#     sigma_resh = sigma.reshape(c*w*h).repeat(b)
#     sigma_pow_resh = torch.pow(sigma_resh, n_resh)
    
#     convnl = torch.Tensor(t_steps, b*c*w*h)
#     normrsp = torch.Tensor(t_steps, b*c*w*h)
#     for t in range(t_steps):

#         # add padding
#         y_shift = F.pad(y_resh, [0, 0, t, t_steps-t])

#         # sliding dot product
#         convnl[t, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
#         normrsp[t, :] = torch.add(convnl[t, :], sigma_pow_resh)

#     return convnl.reshape(t_steps, b, c, w, h), normrsp.reshape(t_steps, b, c, w, h)
