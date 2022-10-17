import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import numpy as np
import math


def h1(tau1, t_steps, sample_rate, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    input = input.repeat(b*c*w*h, 1)
    print(input.shape)
    tau1_resh = torch.transpose(tau1.reshape(c*w*h).repeat(b).repeat(t_steps, 1), 0, 1)
    print(tau1_resh.shape)

    # compute impulse response function
    y = input * torch.exp(-input/tau1_resh)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)


def h2(tau2, sample_rate, t_steps, b, c, w, h):

    # preprocess tensors
    input = torch.arange(t_steps)/sample_rate
    input = input.repeat(b*c*w*h, 1)
    print(input.shape)
    tau2_resh = torch.transpose(tau2.reshape(c*w*h).repeat(b).repeat(t_steps, 1), 0, 1)
    print(tau2_resh.shape)

    # compute impulse response function
    y = torch.exp(-input/tau2_resh)

    return y.transpose(0,1).reshape(t_steps, b, c, w, h)

def torch_convolve(x, y, b, c, w, h, n, t_steps, t_current, true=False):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # remake irf
    y_shift = torch.zeros(t_steps, b*c*w*h)
    y_shift[-t_current:, :] = y_resh[:t_current, :] 

    n_resh = n.reshape(c*w*h).repeat(b)

    # sliding dot product
    cv = torch.tensordot(x_resh, y_shift)

    output = cv**n_resh

    return output.reshape(b, c, w, h)

def torch_convolve_norm(x, y, b, c, w, h, sigma, n, t_steps, t_current):

    # preprocess
    x_resh = x.reshape(t_steps, b*c*w*h)
    y_resh = y.reshape(t_steps, b*c*w*h)

    # remake irf
    y_shift = torch.zeros(t_steps, b*c*w*h)
    y_shift[-t_current:, :] = y_resh[:t_current, :] 

    # reshape sigma
    sigma_resh = sigma.reshape(c*w*h).repeat(b)
    n_resh = n.reshape(c*w*h).repeat(b)

    # sliding dot product
    cv = torch.zeros(b*c*w*h)
    for i in range(b*c*w*h):
        cv[i] = torch.dot(x_resh[:, i], y_shift[:, i])

    normrsp = torch.abs(cv)**n_resh + sigma_resh**n_resh

    return cv.reshape(b, c, w, h), normrsp.reshape(b, c, w, h)
