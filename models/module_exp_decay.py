import torch
import torch.nn as nn

class module_exp_decay(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()

        if height == 'none':
            self.channels = channels
            self.alpha   = nn.Parameter(torch.rand([self.channels]).unsqueeze_(dim=0))
            self.beta    = nn.Parameter(torch.ones([self.channels]).unsqueeze_(dim=0))
        else:
            self.height = height
            self.width = width
            self.channels = channels
            self.alpha   = nn.Parameter(torch.rand([self.height, self.width, self.channels]).unsqueeze_(dim=0))
            self.beta    = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0))

    def forward(self, x, x_previous, s_previous, print_val=False):

        # alpha   = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0))*0.5
        # beta    = nn.Parameter(torch.ones([self.height, self.width, self.channels]).unsqueeze_(dim=0))*2.01        
        # s_updt =  (alpha * s_previous + (1 - alpha) * x_previous)
        # s_beta_updt = beta * (alpha * s_previous + (1 - alpha) * x_previous)

        s_updt =  (self.alpha * s_previous + (1 - self.alpha) * x_previous)
        s_beta_updt = self.beta * (self.alpha * s_previous + (1 - self.alpha) * x_previous)

        # if print_val:

            # print(30*'--')
            # print('alpha: ', torch.mean(self.beta))
            # print('beta: ', torch.mean(self.alpha))

        #     print('s_prev: ', torch.mean(s_previous))
        #     print('x_prev: ', torch.mean(x_previous))
        #     print('x: ', torch.mean(x))
        #     print(30*'--')

        return s_updt, s_beta_updt
