#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class module_div_norm(nn.Module):
    
    def __init__(self, batchsiz, height, width, channels):
        super().__init__()

        # set dimensions
        self.b = batchsiz
        self.h = height
        self.w = width
        self.c = channels

        self.sample_rate = 1

        if height == 'none':

            # parameter settings (r1 = minimal value, r2 = maximal value)
            r1 = 0.05
            r2 = 0.1
            self.tau1    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

            r1 = 0.1
            r2 = 0.2
            self.tau2    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

            r1 = 0.8
            r2 = 0.12
            self.sigma   = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

            r1 = 1
            r2 = 2
            self.n    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)
        
        else:

            # parameter settings (r1 = minimal value, r2 = maximal value)
            r1 = 0.05
            r2 = 0.1
            # self.tau1    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
            self.tau1    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
            # self.tau1    = torch.rand(self.c, self.w, self.h)
            # self.tau1    = (r1 - r2) * self.tau1 + r2   

            r1 = 0.1
            r2 = 0.2
            # self.tau2    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
            self.tau2    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
            # self.tau2    = torch.rand(self.c, self.w, self.h)
            # self.tau2    = (r1 - r2) * self.tau2 + r2 

            r1 = 0.8
            r2 = 0.12
            # self.sigma   = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
            self.sigma   = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
            # self.sigma    = torch.rand(self.c, self.w, self.h)
            # self.sigma    = (r1 - r2) * self.sigma + r2 

            r1 = 1
            r2 = 2
            # self.n    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
            self.n    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
            # self.n    = torch.rand(self.c, self.w, self.h)
            # self.n    = (r1 - r2) * self.n + r2
            self.height = height
            self.width = width
            self.channels = channels
            self.alpha   = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0))
            self.beta    = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0))
 

    def h1(self, t_steps):

        # preprocess tensors
        input = torch.arange(t_steps)/self.sample_rate
        input = input.repeat(self.b*self.c*self.w*self.h, 1)
        tau1_resh = torch.transpose(self.tau1.reshape(self.c*self.w*self.h).repeat(self.b).repeat(t_steps, 1), 0, 1)
        
        # compute impulse response function
        y = input * torch.exp(-input/tau1_resh)
        y = y/torch.sum(y)

        return y.transpose(0,1).reshape(t_steps, self.b, self.c, self.w, self.h)

    def h2(self, t_steps):

        # preprocess tensors
        input = torch.arange(t_steps)/self.sample_rate
        input = input.repeat(self.b*self.c*self.w*self.h, 1)
        tau2_resh = torch.transpose(self.tau2.reshape(self.c*self.w*self.h).repeat(self.b).repeat(t_steps, 1), 0, 1)

        # compute impulse response function
        y = torch.exp(-input/tau2_resh)
        y = y/torch.sum(y)

        return y.transpose(0,1).reshape(t_steps, self.b, self.c, self.w, self.h)    
            
    def torch_cross_val(self, x, y, t, t_steps):

        # preprocess
        x_resh = x.reshape(t, self.b*self.c*self.w*self.h)
        y_resh = y.reshape(t_steps, self.b*self.c*self.w*self.h)

        # reshape parameters
        n_resh = self.n.reshape(self.c*self.w*self.h).repeat(self.b)

        # fix stimulus
        x_tau = F.pad(x_resh, [0, 0, t_steps, t_steps-t])

        output = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
        for tmp in range(t_steps):

            # add padding
            y_shift = F.pad(y_resh, [0, 0, tmp, t_steps-tmp])

            # sliding dot product
            # output[tmp, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
            output[tmp, :] = torch.pow(torch.tensordot(x_tau, y_shift), n_resh)

        return output.reshape(t_steps, self.b, self.c, self.w, self.h)

    def torch_cross_val_norm(self, x, y, t, t_steps):

        # preprocess
        x_resh = x.reshape(t_steps, self.b*self.c*self.w*self.h)
        y_resh = y.reshape(t_steps, self.b*self.c*self.w*self.h)

        # fix stimulus
        x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

        # reshape sigma
        n_resh = self.n.reshape(self.c*self.w*self.h).repeat(self.b)
        sigma_resh = self.sigma.reshape(self.c*self.w*self.h).repeat(self.b)
        sigma_pow_resh = torch.pow(sigma_resh, n_resh)
        
        convnl = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
        normrsp = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
        for tmp in range(t_steps):

            # add padding
            y_shift = F.pad(y_resh, [0, 0, tmp, t_steps-tmp])

            # sliding dot product
            # convnl[tmp, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
            convnl[tmp, :] = torch.pow(torch.tensordot(x_tau, y_shift), n_resh)
            normrsp[tmp, :] = torch.add(convnl[tmp, :], sigma_pow_resh)

        return normrsp.reshape(t_steps, self.b, self.c, self.w, self.h)

    def forward(self, x, t, t_steps):

        # compute inversed IRFs (used for cross-validation)
        irf = self.h1(t_steps)
        irf_inv = irf.reshape(t_steps, self.b*self.c*self.w*self.h)
        irf_inv = torch.flip(irf_inv, [0])
        irf_inv = irf_inv.reshape(t_steps, self.b, self.c, self.w, self.h)
        # print(irf_inv)

        irf_norm = self.h2(t_steps)
        irf_norm_inv = irf_norm.reshape(t_steps, self.b*self.c*self.w*self.h)
        irf_norm_inv = torch.flip(irf_norm_inv, [0])
        irf_norm_inv = irf_norm_inv.reshape(t_steps, self.b, self.c, self.w, self.h)        
        # print(irf_norm_inv)

        # # compute input drive and normalisation pool
        # input_drive_cross = self.torch_cross_val(x, irf_inv, t, t_steps)
        # normrsp = self.torch_cross_val_norm(input_drive_cross, irf_norm_inv, t, t_steps)

        # DN model response
        # r = torch.div(input_drive_cross, normrsp)
        r = torch.div(irf,irf_norm)

        return r[t, :, :, :, :]








