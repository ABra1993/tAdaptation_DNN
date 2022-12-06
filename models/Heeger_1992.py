
# %%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from utils import h1, h2, torch_convolve, torch_cross_val_norm, torch_cross_val

# define root
dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

# input
b = 1 # batch-size
c = 2 # number channels
w = 2 # height
h = 2 # width

# parameter settings (r1 = minimal value, r2 = maximal value)
# r1 = 0.05
# r2 = 0.05
# tau1    = (r1 - r2) * torch.rand(c, w, h) + r2
tau1 = 0.5

# r1 = 0.1
# r2 = 0.1

# tau2    = (r1 - r2) * torch.rand(c, w, h) + r2
tau2 = 0.1

# r1 = 0.1
# r2 = 0.1
# sigma   = (r1 - r2) * torch.rand(c, w, h) + r2
sigma = 0.1

# r1 = 2
# r2 = 2
# n    = (r1 - r2) * torch.rand(c, w, h) + r2
n = 1

# timecourse
sample_rate = 16
t_steps = 160
dur = 80
start = [20]
tmpts = np.arange(t_steps)/sample_rate

x = torch.zeros(t_steps)
for i in range(len(start)):
    x[start[i]:start[i]+dur] = 0.3

# # model parameters
# tau1 = 0.8
# tau2 = 0.07
sigma = 0.1
# n = 1.5

alpha_input_drive = 0.9
alpha_normalisation = 0.7

beta_input_drive = 0.7
beta_normalisation = 0.6

# sample_rate = 1
# t_steps = 3
# dur = 1
# start = [0, 2]

# x = torch.zeros([t_steps, b, c, w, h])
# value = [1, 2]
# for i in range(len(start)):
#     x[start[i]:start[i]+dur, :, :, :, :] = value[i]

input_drive = np.zeros(t_steps)

right_side = np.zeros(t_steps)
normalisation = np.zeros(t_steps)

s_normalisation = np.zeros(t_steps)
s_input_drive = np.zeros(t_steps)

r = np.zeros(t_steps)

exp_decay = np.zeros(t_steps)

for t in range(1, t_steps):

    # compute suppression states
    s_input_drive[t] = alpha_input_drive * s_input_drive[t-1] + (1 - alpha_input_drive) * x[t-1]
    s_normalisation[t] = alpha_normalisation * s_normalisation[t-1] + (1 - alpha_normalisation) * x[t-1]
    if t == 1:
        print('Normalisation')
        print(s_normalisation[t])
        print(s_normalisation[t-1])
        print(x[t-1])

    # delayed normalisation
    input_drive[t] = (beta_input_drive * s_input_drive[t])
    normalisation[t] = sigma + (beta_normalisation * s_normalisation[t])

    # compute response
    r[t] = input_drive[t] / normalisation[t]
    if t == 1:
        print(input_drive[t])
        print(normalisation[t])
        print('r', r[t])

print(normalisation[0])
fig = plt.figure()

# stimulus timecourse
plt.plot(tmpts, x, color='black')
plt.plot(tmpts, input_drive, color='green', label='input drive')
plt.plot(tmpts, normalisation, color='blue', label='norm.')
plt.plot(tmpts, r, color='red', label='R')
plt.plot(tmpts, exp_decay, color='red', linestyle='--')
plt.legend()
plt.show()

# plot canonical computations
# fig, axs = plt.subplots(2, 4, figsize=(20, 5))
# count = 0
# lw = 1
# for i in range(len(x[0, :, 0, 0, 0])):
#     for j in range(len(x[0, 0, :, 0, 0])):
#         for k in range(len(x[0, 0, 0, :, 0])):
#             for l in range(len(x[0, 0, 0, 0, :])):

                # # determine index
                # if count > 3:
                #     idx_row = 1
                # else:
                #     idx_row = 0
                # idx_col = (count % 4)

                # # plot impulse response functions for first batch
                # if i == 0:


                    # # # set title
                    # # axs[idx_row, idx_col].set_title(r'$\tau_{1} = $' + tau1_current + r', $\tau_{2} = $' +  tau2_current + r', $\sigma = $' +  sigma_current + r', $n = $' +  n_current, fontsize=6)
                    # axs[idx_row, idx_col].set_title(r'$\tau_{1} = $' + str(tau1) + r', $\tau_{2} = $' +  str(tau2) + r', $\sigma = $' +  str(sigma) + r', $n = $' +  str(n), fontsize=6)

                    # # activation
                    # # axs[idx_row, idx_col].plot(x[:, i, j, k, l], 'k', label='$S/a_{c,h,w}$', lw=lw, alpha=0.7)
                    # for m in range(len(start)):
                    #     axs[idx_row, idx_col].axvspan(start[m], start[m]+dur, color='grey', alpha=0.1)
                    
                    # # impulse response functiona
                    # axs[idx_row, idx_col].plot(t-max(t), irf_inv[0, 0, :], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')

                    # axs[idx_row, idx_col].plot(conv_input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
                    # axs[idx_row, idx_col].plot(input_drive[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')

                    # axs[idx_row, idx_col].plot(conv_normrsp[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='red')
                    # axs[idx_row, idx_col].plot(normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
                    
                    # axs[idx_row, idx_col].plot(r[:, i, j, k, l], label=r'$r_{DN}$', color='red')

#                 # increment count
#                 count+=1

# axs[0,0].legend(fontsize=8)
# plt.tight_layout()

file = 'dn_heeger92'
plt.savefig(dir+'visualizations/DN_in_DNN/' + file)
# plt.show()

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
