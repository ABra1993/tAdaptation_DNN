import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import torch.nn as nn

class Zero(nn.Module):
    def forward(self, X):
        return X*0