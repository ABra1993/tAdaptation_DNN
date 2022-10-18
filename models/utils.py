import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F


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

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    output = torch.Tensor(t_steps, b*c*w*h)
    for t in range(t_steps):
        if t == 0:
            continue

        # shift y
        y_shift = torch.zeros(t_steps, b*c*w*h)
        y_shift[t:, :] = y_resh[:t_steps-t, :] 

        # sliding dot product
        output[t, :] = torch.tensordot(x_resh, y_shift)

    return output.reshape(t_steps, b, c, w, h)

def torch_cross_val(x, y, b, c, w, h, n, t_steps):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # fix stimulus
    x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

    output = torch.Tensor(t_steps, b*c*w*h)
    for t in range(t_steps):
        if t == 0:
            continue

        # add padding
        y_shift = F.pad(y_resh, [0, 0, t, t_steps-t])

        # sliding dot product
        output[t, :] = torch.tensordot(x_tau, y_shift)

    return output.reshape(t_steps, b, c, w, h)

def torch_cross_val_norm(x, y, b, c, w, h, sigma, n, t_steps):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # fix stimulus
    x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

    # reshape sigma
    sigma_resh = sigma.reshape(c*w*h).repeat(b)

    convnl = torch.Tensor(t_steps, b*c*w*h)
    normrsp = torch.Tensor(t_steps, b*c*w*h)
    normrsp[0, :] = sigma_resh
    for t in range(t_steps):
        if t == 0:
            continue

        # add padding
        y_shift = F.pad(y_resh, [0, 0, t, t_steps-t])

        # sliding dot product
        convnl[t, :] = torch.tensordot(x_tau, y_shift)
        normrsp[t, :] = torch.add(convnl[t, :], sigma_resh)

    return convnl.reshape(t_steps, b, c, w, h), normrsp.reshape(t_steps, b, c, w, h)
