#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import h1, h2, torch_convolve, torch_cross_val_norm, torch_cross_val

# input
b = 2 # batch-size
c = 2 # number channels
w = 2 # height
h = 2 # width

# parameter settings (r1 = minimal value, r2 = maximal value)
r1 = 0.8
r2 = 0.12
sigma   = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 0.05
r2 = 0.1
tau1    = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 0.1
r2 = 0.2
tau2    = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 1
r2 = 2
n    = (r1 - r2) * torch.rand(c, w, h) + r2

# timecourse
# t_steps = 7
# x = torch.zeros([t_steps, b, c, w, h])
# x[3:5, :, :, :, :] = 1

t_steps = 1000
x = torch.zeros([t_steps, b, c, w, h])
x[300:800, :, :, :, :] = 1

# create timepoints based on sample rate
sample_rate = 512
t = np.arange(t_steps)/sample_rate

# compute inversed IRFs (used for cross-validation)
irf = h1(tau1, t_steps, sample_rate, b, c, w, h)
irf_inv = irf.reshape(t_steps, b*c*w*h)
irf_inv = torch.flip(irf_inv.reshape(t_steps, b*c*w*h), [0])
irf_inv = irf_inv.reshape(t_steps, b, c, w, h)

irf_norm = h2(tau2, sample_rate, t_steps, b, c, w, h)
irf_norm_inv = irf_norm.reshape(t_steps, b*c*w*h)
irf_norm_inv = torch.flip(irf_norm_inv.reshape(t_steps, b*c*w*h), [0])
irf_norm_inv = irf_norm_inv.reshape(t_steps, b, c, w, h)

# linear convolution
# input_drive = torch_convolve(x, irf, b, c, w, h, n, t_steps)
input_drive_cross = torch_cross_val(x, irf_inv, b, c, w, h, n, t_steps)

# # nonlinear convolution and computation of normalisation response
convnl, normrsp = torch_cross_val_norm(input_drive_cross, irf_norm_inv, b, c, w, h, sigma, n, t_steps)

# # DN model response
r = torch.div(input_drive_cross, normrsp)

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

                    # retrieve parameter values
                    torch.set_printoptions(precision=2)
                    tau1_current = str(torch.round(tau1[j, k, l], decimals=2))
                    tau2_current = str(torch.round(tau2[j, k, l], decimals=2))
                    sigma_current = str(torch.round(sigma[j, k, l], decimals=2))
                    n_current = str(torch.round(n[j, k, l], decimals=2))

                    # set title
                    axs[idx_row, idx_col].set_title(r'$\tau_{1} = $' + tau1_current + r', $\tau_{2} = $' +  tau2_current + r', $\sigma = $' +  sigma_current + r', $n = $' +  n_current, fontsize=6)

                    # activation
                    axs[idx_row, idx_col].plot(t, x[:, i, j, k, l], 'k', label='$S/a_{c,h,w}$', lw=lw, alpha=0.7)
                    
                    # impulse response functiona
                    # axs[idx_row, idx_col].plot(t, irf[:, i, j, k, l], label=r'$irf$', lw=lw, alpha=0.5, color='green', linestyle='--')
                    # axs[idx_row, idx_col].plot(t-max(t), irf_inv[:, i, j, k, l], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')
                    # axs[idx_row, idx_col].plot(t-max(t), irf_norm_inv[:, i, j, k, l], label=r'$irf_{norm}$', lw=lw, alpha=0.7, color='orange', linestyle='--')
                      
                    # axs[idx_row, idx_col].plot(t, input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
                    axs[idx_row, idx_col].plot(t, input_drive_cross[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')
                    axs[idx_row, idx_col].plot(t, convnl[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='orange')
                    axs[idx_row, idx_col].plot(t, normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
                    
                    axs[idx_row, idx_col].plot(t, r[:, i, j, k, l], label=r'$r_{DN}$', color='red')

                # increment count
                count+=1

axs[0,0].legend(fontsize=8)
plt.tight_layout()

file = 'dn_torch_all'
plt.savefig('visualizations/DN_in_DNN/' + file)
plt.show()
