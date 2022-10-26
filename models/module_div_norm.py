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

        # if height == 'none':

        #     # parameter settings (r1 = minimal value, r2 = maximal value)
        #     r1 = 0.05
        #     r2 = 0.1
        #     self.tau1    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

        #     r1 = 0.1
        #     r2 = 0.2
        #     self.tau2    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

        #     r1 = 0.8
        #     r2 = 0.12
        #     self.sigma   = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)

        #     r1 = 1
        #     r2 = 2
        #     self.n    = nn.Parameter((r1 - r2) * torch.rand(self.c) + r2)
        
        # else:

        # parameter settings (r1 = minimal value, r2 = maximal value)
        r1 = 0.05
        r2 = 0.05
        self.tau1    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
        # self.tau1    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
        # self.tau1    = torch.rand(self.c, self.w, self.h)
        # self.tau1    = (r1 - r2) * self.tau1 + r2   

        r1 = 0.1
        r2 = 0.1
        self.tau2    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
        # self.tau2    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
        # self.tau2    = torch.rand(self.c, self.w, self.h)
        # self.tau2    = (r1 - r2) * self.tau2 + r2 

        r1 = 0.1
        r2 = 0.1
        self.sigma   = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
        # self.sigma   = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
        # self.sigma    = torch.rand(self.c, self.w, self.h)
        # self.sigma    = (r1 - r2) * self.sigma + r2 

        r1 = 2
        r2 = 2
        self.n    = (r1 - r2) * torch.rand(self.c, self.w, self.h) + r2
        # self.n    = nn.Parameter((r1 - r2) * torch.rand(self.c, self.w, self.h) + r2)
        # self.n    = torch.rand(self.c, self.w, self.h)
        # self.n    = (r1 - r2) * self.n + r2


    def h1(self, t_steps):

        # preprocess tensors
        input = torch.arange(t_steps)/self.sample_rate
        input = input.repeat(self.b*self.c*self.w*self.h, 1)
        tau1_resh = torch.transpose(self.tau1.reshape(self.c*self.w*self.h).repeat(self.b).repeat(t_steps, 1), 0, 1)
        
        # compute impulse response function
        y = input * torch.exp(-input/tau1_resh)
        # y = y/torch.sum(y)

        return y.transpose(0,1).reshape(t_steps, self.b, self.c, self.w, self.h)

    def h2(self, t_steps):

        # preprocess tensors
        input = torch.arange(t_steps)/self.sample_rate
        input = input.repeat(self.b*self.c*self.w*self.h, 1)
        tau2_resh = torch.transpose(self.tau2.reshape(self.c*self.w*self.h).repeat(self.b).repeat(t_steps, 1), 0, 1)

        # compute impulse response function
        y = torch.exp(-input/tau2_resh)
        # y = y/torch.sum(y)

        return y.transpose(0,1).reshape(t_steps, self.b, self.c, self.w, self.h)    
            
    # def torch_cross_val(self, x, y, t, t_steps):

    #     # preprocess
    #     x_resh = x.reshape(t, self.b*self.c*self.w*self.h)
    #     y_resh = y.reshape(t_steps, self.b*self.c*self.w*self.h)

    #     # reshape parameters
    #     n_resh = self.n.reshape(self.c*self.w*self.h).repeat(self.b)

    #     # fix stimulus
    #     x_tau = F.pad(x_resh, [0, 0, t_steps, t_steps-t])

    #     output = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
    #     for tmp in range(t_steps):

    #         # add padding
    #         y_shift = F.pad(y_resh, [0, 0, tmp, t_steps-tmp])

    #         # sliding dot product
    #         # output[tmp, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
    #         output[tmp, :] = torch.pow(torch.tensordot(x_tau, y_shift), n_resh)

    #     return output.reshape(t_steps, self.b, self.c, self.w, self.h)

    # def torch_cross_val_norm(self, x, y, t, t_steps):

    #     # preprocess
    #     x_resh = x.reshape(t_steps, self.b*self.c*self.w*self.h)
    #     y_resh = y.reshape(t_steps, self.b*self.c*self.w*self.h)

    #     # fix stimulus
    #     x_tau = F.pad(x_resh, [0, 0, t_steps, 0])

    #     # reshape sigma
    #     n_resh = self.n.reshape(self.c*self.w*self.h).repeat(self.b)
    #     sigma_resh = self.sigma.reshape(self.c*self.w*self.h).repeat(self.b)
    #     sigma_pow_resh = torch.pow(sigma_resh, n_resh)
        
    #     convnl = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
    #     normrsp = torch.Tensor(t_steps, self.b*self.c*self.w*self.h)
    #     for tmp in range(t_steps):

    #         # add padding
    #         y_shift = F.pad(y_resh, [0, 0, tmp, t_steps-tmp])

    #         # sliding dot product
    #         # convnl[tmp, :] = torch.pow(torch.abs(torch.tensordot(x_tau, y_shift)), n_resh)
    #         convnl[tmp, :] = torch.pow(torch.tensordot(x_tau, y_shift), n_resh)
    #         normrsp[tmp, :] = torch.add(convnl[tmp, :], sigma_pow_resh)

    #     return normrsp.reshape(t_steps, self.b, self.c, self.w, self.h)

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

        conv_input_drive = torch.Tensor(t_steps, self.b, self.c, self.w, self.h)
        conv_normrsp = torch.Tensor(t_steps, self.b, self.c, self.w, self.h)
        for i in range(self.b):
            for j in range(self.c):
                for k in range(self.w):
                    for l in range(self.h):

                        # convolution (cross-validation)
                        x_cur = x[:, i, j, k, l].view(1, 1, -1)
                        irf_inv_cur = irf_inv[:, i, j, k, l].view(1, 1, -1)
                        with torch.no_grad():
                            conv1d = F.conv1d(x_cur, irf_inv_cur, padding=t_steps).view(-1)[0:t_steps]
                        conv_input_drive[:, i, j, k, l] = conv1d/torch.max(conv1d)
                        
                        # convolution (cross-validation)
                        conv_input_drive_cur = conv_input_drive[:, i, j, k, l].view(1, 1, -1)
                        irf_norm_inv_cur = irf_norm_inv[:, i, j, k, l].view(1, 1, -1)
                        with torch.no_grad():
                            conv1d = F.conv1d(conv_input_drive_cur, irf_norm_inv_cur, padding=t_steps).view(-1)[0:t_steps]
                        conv_normrsp[:, i, j, k, l] = conv1d/torch.max(conv1d)

        # reshape parameters
        sigma_resh = self.sigma.reshape(self.c*self.w*self.h).repeat(self.b)
        n_resh = self.n.reshape(self.c*self.w*self.h).repeat(self.b)

        conv_input_drive = conv_input_drive.reshape(t_steps, self.b*self.c*self.w*self.h)
        input_drive = torch.abs(torch.pow(conv_input_drive, n_resh)).reshape(t_steps, self.b, self.c, self.w, self.h)
        conv_input_drive = conv_input_drive.reshape(t_steps, self.b, self.c, self.w, self.h)

        conv_normrsp = conv_normrsp.reshape(t_steps, self.b*self.c*self.w*self.h)
        conv_exp_normrsp = torch.abs(torch.pow(conv_normrsp, n_resh))
        # conv_exp_normrsp = conv_exp_normrsp/torch.max(conv_exp_normrsp)
        normrsp = torch.add(conv_exp_normrsp, sigma_resh).reshape(t_steps, self.b, self.c, self.w, self.h)
        conv_normrsp = conv_normrsp.reshape(t_steps, self.b, self.c, self.w, self.h)
    
        # # compute input drive and normalisation pool
        # input_drive_cross = self.torch_cross_val(x, irf_inv, t, t_steps)
        # normrsp = self.torch_cross_val_norm(input_drive_cross, irf_norm_inv, t, t_steps)

        # DN model response
        r = torch.div(input_drive, normrsp)
        r = torch.nan_to_num(r)
        # plt.scatter(t, r[t, 0, 0, 0, 0])
        # # print(r[:, 0, 0, 0, 0])
        # print(conv_input_drive[:, 0, 0, 0, 0])
        # print(conv_normrsp[:, 0, 0, 0, 0])

        # if t == t_steps - 1:
        #     plt.plot(irf_inv[:, 0, 0, 0, 0], label=r'$irf_{inv}$', lw=2, alpha=0.5, color='blue', linestyle='--')
        # print(torch.max(irf_norm_inv[:, 0, 0, 0, 0]))
        #     plt.show()

        # return irf_inv, irf_norm_inv, conv_input_drive, input_drive, conv_normrsp, conv_exp_normrsp, normrsp, r[t, :, :, :, :]



        return r[t, :, :, :, :]







