import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn

class Positive(nn.Module):
    def forward(self, X):
        return torch.abs(X)