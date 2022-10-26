#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from utils import h1, h2, torch_convolve, torch_cross_val_norm, torch_cross_val

def h1(tau1, t_steps, sample_rate, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    input = input.repeat(b*c*w*h, 1)
    tau1_resh = torch.transpose(tau1.reshape(c*w*h).repeat(b).repeat(t_steps, 1), 0, 1)

    # compute impulse response function
    y = input * torch.exp(-input/tau1_resh)
    # y = y/torch.sum(y)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)

def h2(tau2, sample_rate, t_steps, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    input = input.repeat(b*c*w*h, 1)
    tau2_resh = torch.transpose(tau2.reshape(c*w*h).repeat(b).repeat(t_steps, 1), 0, 1)

    # compute impulse response function
    y = torch.exp(-input/tau2_resh)
    # y = y/torch.sum(y)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)

# def torch_convolve(x, y, b, c, w, h, n, t_steps):
#     ''' Not relevant for DN model, includes convolution. '''

#     # preprocess
#     x_resh = x.reshape(t_steps, b*c*w*h)
#     y_resh = y.reshape(t_steps, b*c*w*h)

#     # reshape parameters
#     n_resh = n.reshape(c*w*h).repeat(b)

#     output = torch.Tensor(t_steps, b*c*w*h)
#     for t in range(t_steps):
#         if t == 0:
#             continue

#         # shift y
#         y_shift = torch.zeros(t_steps, b*c*w*h)
#         y_shift[t:, :] = y_resh[:t_steps-t, :] 

#         # sliding dot product
#         output[t, :] = torch.pow(torch.abs(torch.tensordot(x_resh, y_shift)), n_resh)

#     return output.reshape(t_steps, b, c, w, h)

def torch_cross_val(x, y, b, c, w, h, n, t_steps):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # reshape parameters
    n_resh = n.reshape(c*w*h).repeat(b)

    # fix stimulus
    x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

    output = torch.Tensor(t_steps, b*c*w*h)
    for t in range(t_steps):

        # add padding
        y_shift = F.pad(y_resh, [0, 0, t, t_steps-t])

        # sliding dot product
        output[t, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)

    return output.reshape(t_steps, b, c, w, h)

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

# input
b = 2 # batch-size
c = 2 # number channels
w = 2 # height
h = 2 # width

# parameter settings (r1 = minimal value, r2 = maximal value)
r1 = 0.1
r2 = 0.1
tau1    = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 0.1
r2 = 0.1
tau2    = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 0.1
r2 = 0.1
sigma   = (r1 - r2) * torch.rand(c, w, h) + r2

r1 = 1
r2 = 1
n    = (r1 - r2) * torch.rand(c, w, h) + r2

# timecourse
t_steps = 40
dur = 10
start = [5, 20]

x = torch.zeros([t_steps, b, c, w, h])
for i in range(len(start)):
    x[start[i]:start[i]+dur, :, :, :, :] = 0.25

# create timepoints based on sample rate
sample_rate = 16
t = np.arange(t_steps)/sample_rate

# compute inversed IRFs (used for cross-validation)
irf = h1(tau1, t_steps, sample_rate, b, c, w, h)
irf_inv = irf.reshape(t_steps, b*c*w*h)
irf_inv = torch.flip(irf_inv, [0])
irf_inv = irf_inv.reshape(t_steps, b, c, w, h)

irf_norm = h2(tau2, sample_rate, t_steps, b, c, w, h)
irf_norm_inv = irf_norm.reshape(t_steps, b*c*w*h)
irf_norm_inv = torch.flip(irf_norm_inv, [0])
irf_norm_inv = irf_norm_inv.reshape(t_steps, b, c, w, h)
print(torch.max(irf_norm_inv[:, 0, 0, 0, 0]))


conv_input_drive = torch.Tensor(t_steps, b, c, w, h)
conv_normrsp = torch.Tensor(t_steps, b, c, w, h)
for i in range(b):
    for j in range(c):
        for k in range(w):
            for l in range(h):

                # convolution (cross-validation)
                x_cur = x[:, i, j, k, l].view(1, 1, -1)
                irf_inv_cur = torch.flip(irf[:, i, j, k, l], (0,)).view(1, 1, -1)
                conv1d = F.conv1d(x_cur, irf_inv_cur, padding=t_steps-1).view(-1)[0:t_steps]
                conv_input_drive[:, i, j, k, l] = conv1d/torch.max(conv1d)
                
                # convolution (cross-validation)
                conv_input_drive_cur = conv_input_drive[:, i, j, k, l].view(1, 1, -1)
                irf_norm_inv_cur = irf_norm_inv[:, i, j, k, l].view(1, 1, -1)
                conv1d = F.conv1d(conv_input_drive_cur, irf_norm_inv_cur, padding=t_steps-1).view(-1)[0:t_steps]
                conv_normrsp[:, i, j, k, l] = conv1d/torch.max(conv1d)

# normalise
# conv_input_drive = conv_input_drive/torch.sum(conv_input_drive)
# conv_normrsp = conv1d_normrsp/torch.sum(conv_normrsp)

# compute input drive and normalisation pool
sigma_resh = sigma.reshape(c*w*h).repeat(b)
n_resh = n.reshape(c*w*h).repeat(b)

conv_input_drive = conv_input_drive.reshape(t_steps, b*c*w*h)
input_drive = torch.abs(torch.pow(conv_input_drive, n_resh))
# input_drive = input_drive/torch.max(input_drive, dim=0) 
input_drive = input_drive.reshape(t_steps, b, c, w, h)
conv_input_drive = conv_input_drive.reshape(t_steps, b*c*w*h).reshape(t_steps, b, c, w, h)

conv_normrsp = conv_normrsp.reshape(t_steps, b*c*w*h)
conv_exp_normrsp = torch.abs(torch.pow(conv_normrsp, n_resh))
# conv_exp_normrsp = conv_exp_normrsp/torch.max(conv_exp_normrsp)
normrsp = torch.add(conv_exp_normrsp, sigma_resh).reshape(t_steps, b, c, w, h)

conv_exp_normrsp = conv_normrsp.reshape(t_steps, b, c, w, h)
conv_normrsp = conv_normrsp.reshape(t_steps, b, c, w, h)

# a_batch = x.reshape(b, c*w*h, t_steps)
# b_batch_temp = irf_inv.reshape(b, c*w*h, t_steps)

# b_batch = torch.Tensor(1, 4, t_steps)
# b_batch[0, :, :] = b_batch_temp[0, :, :]
# # b_batch[:, 0, :] = b_batch_temp[0, :, :]

# input_drive_cross_batch = torch.nn.functional.conv1d(a_batch, b_batch, padding=t_steps)
# print(input_drive_cross_batch.shape)

# plt.axvspan(200, 400, color='grey', alpha=0.3)
# plt.axvspan(600, 800, color='grey', alpha=0.3)
# plt.plot(input_drive_cross[:, 0, 0, 0, 0].detach().numpy())
# plt.plot(input_drive_cross_batch[1, 0, :].detach().numpy())
# plt.plot(x[:, 0, 0, 0, 0])
# plt.plot(irf[:, 0, 0, 0, 0])
# plt.plot(input_drive_cross[1, 1, :].detach().numpy())
# plt.plot(input_drive_cross[1, 2, :].detach().numpy())

# plt.show()

# # linear convolution
# # input_drive = torch_convolve(x, irf, b, c, w, h, n, t_steps)
input_drive_cross = torch_cross_val(x, irf_inv, b, c, w, h, n, t_steps)

# # # nonlinear convolution and computation of normalisation response
# convnl, normrsp = torch_cross_val_norm(input_drive_cross, irf_norm_inv, b, c, w, h, sigma, n, t_steps)

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
                    for m in range(len(start)):
                        axs[idx_row, idx_col].axvspan(t[start[m]], t[start[m]+dur], color='grey', alpha=0.1)
                    
                    # impulse response functiona
                    # axs[idx_row, idx_col].plot(t, irf[:, i, j, k, l], label=r'$irf$', lw=lw, alpha=0.5, color='green', linestyle='--')
                    axs[idx_row, idx_col].plot(t-max(t), irf_inv[:, i, j, k, l], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')
                    axs[idx_row, idx_col].plot(t-max(t), irf_norm_inv[:, i, j, k, l], label=r'$irf_{norm}$', lw=lw, alpha=0.7, color='orange', linestyle='--')
                    # axs[idx_row, idx_col].plot(t, conv_normrsp[:, i, j, k, l], label=r'$L \ast h_{2}$', lw=lw, alpha=0.7, color='orange')

                    # axs[idx_row, idx_col].plot(t, conv_input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
                    axs[idx_row, idx_col].plot(t, input_drive[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')
                    # axs[idx_row, idx_col].plot(t, input_drive_cross[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='purple')

                    # axs[idx_row, idx_col].plot(t, conv_exp_normrsp[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='red')
                    axs[idx_row, idx_col].plot(t, normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
                    
                    axs[idx_row, idx_col].plot(t, r[:, i, j, k, l], label=r'$r_{DN}$', color='red')

                # increment count
                count+=1

axs[0,0].legend(fontsize=8)
plt.tight_layout()

file = 'dn_conv1d'
plt.savefig('visualizations/DN_in_DNN/' + file)
plt.show()
