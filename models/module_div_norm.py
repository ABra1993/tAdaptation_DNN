import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class module_div_norm(nn.Module):
    def __init__(self):
        super(module_div_norm, self).__init__()

        # prevent zero division
        self.epsilon = 1.e-8

        # parameters
        self.K          = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.alpha      = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.sigma      = nn.Parameter(torch.Tensor([1]), requires_grad=True)
    
    def forward(self, x_previous, g_previous):
        """ x is the current input computed for the linear response, F is the multiplicative feedback
        and G is the feedback signal. """

        # # clamp previous decay
        # g_previous = torch.clamp(g_previous, min=torch.zeros_like(g_previous), max=torch.ones_like(g_previous) * self.K)   # update feedback signal

        # scaling previous decay (maintains relations between pixel values)
        if torch.max(g_previous) > self.K:                                                                    # rescale if feedback signal exceeds maximal attainable response
            g_previous = g_previous/torch.max(g_previous)
            g_previous = g_previous*self.K
        
        # compute feedback signal
        feedback = torch.sqrt(torch.subtract(self.K, g_previous)+self.epsilon)/(self.sigma + self.epsilon)      # multiplicative feedback                                                    # response
        
        # compute decay signal
        g = torch.add(torch.mul(1 - self.alpha, g_previous), torch.mul(self.alpha, x_previous))                 # update feedback signal

        return g, feedback
