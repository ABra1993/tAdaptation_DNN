#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class module_div_norm(nn.Module):
    
    def __init__(self, batchsiz, height, width, channels, sample_rate, t_steps, tau1_init, train_tau1, tau2_init, train_tau2, sigma_init, train_sigma):
        super().__init__()

        # set dimensions
        self.b = batchsiz
        self.h = height
        self.w = width
        self.c = channels

        self.sample_rate = sample_rate
        self.t_steps = t_steps

        # self.tau1        = nn.Parameter(torch.rand(self.c, self.w, self.h), requires_grad=False)
        # self.tau2        = nn.Parameter(torch.rand(self.c, self.w, self.h), requires_grad=False)
        # self.sigma       = nn.Parameter(torch.rand(self.c, self.w, self.h), requires_grad=False)

        self.tau1        = nn.Parameter(tau1_init, requires_grad=train_tau1)
        self.tau2        = nn.Parameter(tau2_init, requires_grad=train_tau2)
        self.sigma       = nn.Parameter(sigma_init, requires_grad=train_sigma)

    def h1(self):

        # preprocess tensors
        input = torch.arange(self.t_steps)/self.sample_rate
        y = input * torch.exp(-input/self.tau1)
        y = y/torch.sum(y)

        return y

    def h2(self):

        # preprocess tensors
        input = torch.arange(self.t_steps)/self.sample_rate
        y = torch.exp(torch.divide(-input, self.tau2))
        y = y/torch.sum(y)

        return y

    def forward(self, x):

        # compute impulse response functions
        irf = self.h1()
        irf_inv = torch.flip(irf, [-1]).unsqueeze(0).unsqueeze(0)

        irf_norm = self.h2()
        irf_norm_inv = torch.flip(irf_norm, [-1]).unsqueeze(0).unsqueeze(0)

        # reshape 
        x = x.reshape(self.t_steps, self.b*self.c*self.w*self.h).transpose(0,1)

        # convolution input drive
        conv_input_drive = F.conv1d(x.unsqueeze(1), irf_inv, padding=self.t_steps-1)
        conv_input_drive_clip = conv_input_drive[:, 0, 0:self.t_steps]
        # input_drive = torch.pow(torch.abs(conv_input_drive_clip), self.n)
        # input_drive = torch.pow(conv_input_drive_clip, self.n)

        # convolution normalisation response
        conv_normrsp = F.conv1d(conv_input_drive_clip.unsqueeze(1), irf_norm_inv, padding=self.t_steps-1)
        conv_normrsp_clip = conv_normrsp[:, 0, 0:self.t_steps]
        # normrsp = torch.add(torch.pow(torch.abs(conv_normrsp_clip), self.n), self.sigma)
        # normrsp = torch.add(torch.pow(conv_normrsp_clip, self.n), self.sigma)
        normrsp = torch.add(conv_normrsp_clip, self.sigma)

        # reshape
        # x = x.squeeze(1).transpose(0, 1).reshape(self.t_steps, self.b, self.c, self.w, self.h)
        conv_input_drive = conv_input_drive_clip.transpose(0, 1).reshape(self.t_steps, self.b, self.c, self.w, self.h)
        normrsp = normrsp.transpose(0, 1).reshape(self.t_steps, self.b, self.c, self.w, self.h)

        # DN model response
        r = torch.div(conv_input_drive, normrsp)

        return r




