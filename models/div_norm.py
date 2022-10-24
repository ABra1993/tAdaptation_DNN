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
    y = y/torch.sum(y)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)

def h2(tau2, sample_rate, t_steps, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    input = input.repeat(b*c*w*h, 1)
    tau2_resh = torch.transpose(tau2.reshape(c*w*h).repeat(b).repeat(t_steps, 1), 0, 1)

    # compute impulse response function
    y = torch.exp(-input/tau2_resh)
    y = y/torch.sum(y)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)

def torch_convolve(x, y, b, c, w, h, n, t_steps):
    ''' Not relevant for DN model, includes convolution. '''

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # reshape parameters
    n_resh = n.reshape(c*w*h).repeat(b)

    output = torch.Tensor(t_steps, b*c*w*h)
    for t in range(t_steps):
        if t == 0:
            continue

        # shift y
        y_shift = torch.zeros(t_steps, b*c*w*h)
        y_shift[t:, :] = y_resh[:t_steps-t, :] 

        # sliding dot product
        output[t, :] = torch.pow(torch.abs(torch.tensordot(x_resh, y_shift)), n_resh)

    return output.reshape(t_steps, b, c, w, h)

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

def torch_cross_val_norm(x, y, b, c, w, h, sigma, n, t_steps):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # fix stimulus
    x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

    # reshape sigma
    n_resh = n.reshape(c*w*h).repeat(b)
    sigma_resh = sigma.reshape(c*w*h).repeat(b)
    sigma_pow_resh = torch.pow(sigma_resh, n_resh)
    
    convnl = torch.Tensor(t_steps, b*c*w*h)
    normrsp = torch.Tensor(t_steps, b*c*w*h)
    for t in range(t_steps):

        # add padding
        y_shift = F.pad(y_resh, [0, 0, t, t_steps-t])

        # sliding dot product
        convnl[t, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
        normrsp[t, :] = torch.add(convnl[t, :], sigma_pow_resh)

    return convnl.reshape(t_steps, b, c, w, h), normrsp.reshape(t_steps, b, c, w, h)

# input
b = 3 # batch-size
c = 1 # number channels
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
t_steps = 10
x = torch.zeros([t_steps, b, c, w, h])
x[2:6, :, :, :, :] = 1
# x[2, :, :, :, :] = 1

t_steps = 1000
x = torch.zeros([t_steps, b, c, w, h])
x[200:400, :, :, :, :] = 1
x[600:800, :, :, :, :] = 1

# create timepoints based on sample rate
sample_rate = 512
t = np.arange(t_steps)/sample_rate

# compute inversed IRFs (used for cross-validation)
irf = h1(tau1, t_steps, sample_rate, b, c, w, h)
# irf_inv = irf.reshape(t_steps, b*c*w*h)
# irf_inv = torch.flip(irf_inv, [0])
# irf_inv = irf_inv.reshape(t_steps, b, c, w, h)

irf_norm = h2(tau2, sample_rate, t_steps, b, c, w, h)
# irf_norm_inv = irf_norm.reshape(t_steps, b*c*w*h)
# irf_norm_inv = torch.flip(irf_norm_inv, [0])
# irf_norm_inv = irf_norm_inv.reshape(t_steps, b, c, w, h)

input_drive_cross = torch.Tensor(t_steps, b, c, w, h)
conv = torch.Tensor(t_steps, b, c, w, h)
for i in range(b):
    for j in range(c):
        for k in range(w):
            for l in range(h):

                # convolution (cross-validation)
                aa = x[:, i, j, k, l]
                bb = irf[:, i, j, k, l]

                a1 = aa.view(1, 1, -1)
                b1 = torch.flip(bb, (0,)).view(1, 1, -1)
                conv1d = torch.nn.functional.conv1d(a1, b1, padding=t_steps).view(-1)[0:t_steps]
                input_drive_cross[:, i, j, k, l] = torch.abs(conv1d)

                # convolution (cross-validation)
                aa = input_drive_cross[:, i, j, k, l]
                bb = irf_norm[:, i, j, k, l]

                a1 = aa.view(1, 1, -1)
                b1 = torch.flip(bb, (0,)).view(1, 1, -1)
                conv1d = torch.nn.functional.conv1d(a1, b1, padding=t_steps).view(-1)[0:t_steps]
                
                sigma_resh = sigma.reshape(c*w*h).repeat(b)      
                conv[:, i, j, k, l] = torch.abs(conv1d) 

conv = conv.reshape(t_steps, b*c*w*h)
sigma_resh = sigma.reshape(c*w*h).repeat(b)
normrsp = torch.add(conv, sigma_resh).reshape(t_steps, b, c, w, h)

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
# input_drive_cross = torch_cross_val(x, irf_inv, b, c, w, h, n, t_steps)

# # # nonlinear convolution and computation of normalisation response
# convnl, normrsp = torch_cross_val_norm(input_drive_cross, irf_norm_inv, b, c, w, h, sigma, n, t_steps)

# DN model response
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
                    # axs[idx_row, idx_col].plot(t, x[:, i, j, k, l], 'k', label='$S/a_{c,h,w}$', lw=lw, alpha=0.7)
                    
                    # impulse response functiona
                    # axs[idx_row, idx_col].plot(t, irf[:, i, j, k, l], label=r'$irf$', lw=lw, alpha=0.5, color='green', linestyle='--')
                    # axs[idx_row, idx_col].plot(t-max(t), irf_inv[:, i, j, k, l], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')
                    # axs[idx_row, idx_col].plot(t-max(t), irf_norm_inv[:, i, j, k, l], label=r'$irf_{norm}$', lw=lw, alpha=0.7, color='orange', linestyle='--')
                      
                    # axs[idx_row, idx_col].plot(t, input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
                    axs[idx_row, idx_col].plot(t, input_drive_cross[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')
                    # axs[idx_row, idx_col].plot(t, convnl[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='orange')
                    axs[idx_row, idx_col].plot(t, normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
                    
                    axs[idx_row, idx_col].plot(t, r[:, i, j, k, l], label=r'$r_{DN}$', color='red')

                # increment count
                count+=1

axs[0,0].legend(fontsize=8)
plt.tight_layout()

file = 'dn_conv1d'
plt.savefig('visualizations/DN_in_DNN/' + file)
plt.show()
