#%%
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class module_div_norm_rec(nn.Module):

    def __init__(self, height, width, channels):
        super().__init__()

        if height == 'none':
            self.channels   = channels

            self.tau1       = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0))
            self.tau2       = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0))
            self.sigma      = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0))
            # self.n          = nn.Parameter(torch.ones([self.channels]).unsqueeze_(dim=0)*0.5)

        else:
            self.height     = height
            self.width      = width
            self.channels   = channels
            self.alpha      = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.5)
            self.beta       = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.5)

            self.tau1       = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.5)
            self.tau2       = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.5)
            self.sigma      = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0))
            # self.n          = nn.Parameter(torch.ones([self.channels]).unsqueeze_(dim=0)*0.5)

    def forward(self, x, x_previous, input_previous, right_side_previous):
        
        # self.tau1 = 0.9
        # self.tau2 = 0.96
        # self.n = 1.5
        # self.sigma = 0.5

        input_drive = self.alpha * (self.tau1 * input_previous + (1 - self.tau1) * abs(x_previous))
        # input_drive_beta_updt = self.beta * (self.tau1 * input_previous + (1 - self.tau1) * x_previous)
        # input_drive = x - input_drive_beta_updt
        
        
        right_side_beta_updt = self.beta * (self.tau2 * right_side_previous + (1 - self.tau2) * x_previous)
        
        

        # input = self.tau1 * input_previous + (1 - self.tau1) * x_previous
        right_side = self.tau2 * right_side_previous + (1 - self.tau2) * x_previous
        # r = input**self.n / (self.sigma**self.n + right_side**self.n)
        r = torch.div(input_drive, self.sigma + right_side)
        # r = x - right_side_beta_updt
        return input, right_side, r




