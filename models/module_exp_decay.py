import torch
import torch.nn as nn

class module_exp_decay(nn.Module):
    def __init__(self, height, width, channels, train_alpha, train_beta):
        super().__init__()

        if height == 'none':
            self.channels = channels
            # self.alpha   = nn.Parameter(torch.ones([self.channels]).unsqueeze_(dim=0)*0.96, requires_grad=train_alpha)
            # self.beta    = nn.Parameter(torch.ones([self.channels]).unsqueeze_(dim=0)*0.7, requires_grad=train_beta)
            self.alpha   = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0), requires_grad=train_alpha)
            self.beta    = nn.Parameter(torch.zeros([self.channels]).unsqueeze_(dim=0), requires_grad=train_beta)

        else:
            self.height = height
            self.width = width
            self.channels = channels
            # self.alpha   = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.96, requires_grad=train_alpha)
            # self.beta    = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0)*0.7, requires_grad=train_beta)
            self.alpha   = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0), requires_grad=train_alpha)
            self.beta    = nn.Parameter(torch.zeros([self.height, self.width, self.channels]).unsqueeze_(dim=0), requires_grad=train_beta)

    def forward(self, x, x_previous, s_previous):

        s_updt =  (self.alpha * s_previous + (1 - self.alpha) * x_previous)
        s_beta_updt = self.beta * (self.alpha * s_previous + (1 - self.alpha) * x_previous)

        return s_updt, s_beta_updt
