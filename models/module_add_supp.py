import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class module_add_supp(nn.Module):
    def __init__(self):
        super(module_add_supp, self).__init__()

        # parameters
        self.alpha  = nn.Parameter(torch.rand(1), requires_grad=True)
        self.beta   = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x_previous, g_previous):

        # update feedback signal
        g = torch.add(torch.mul(g_previous, 1 - self.alpha), torch.mul(x_previous, self.alpha))
        
        # compute feedback for current timestep (using g_previous)
        feedback = torch.mul(self.beta, g)

        return g, feedback
        