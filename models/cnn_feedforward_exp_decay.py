import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
# print(os.getcwd())

import torch.nn as nn
from models.module_exp_decay import module_exp_decay
import torch.nn.functional as F


class cnn_feedforward_exp_decay(nn.Module):

    def __init__(self, train_alpha, train_beta, t_steps=3):
        super(cnn_feedforward_exp_decay, self).__init__()

        # training variables
        self.t_steps = t_steps

        # layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.sconv1 = module_exp_decay(32, 24, 24, train_alpha, train_beta)
        # self.sconv1 = module_div_norm(32, 24, 24)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.sconv2 = module_exp_decay(32, 8, 8, train_alpha, train_beta)
        # self.sconv2 = module_div_norm(32, 8, 8)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.sconv3 = module_exp_decay(32, 2, 2, train_alpha, train_beta)
        # self.sconv3 = module_div_norm(32, 2, 2)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.sfc1 = module_exp_decay('none', 'none', 1024, train_alpha, train_beta)
        # self.sfc1 = module_div_norm('none', 'none', 1024)

        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train

    def forward(self, input, batch=True):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # initiate activations
        actvsc1 = {}
        actvsc2 = {}
        actvsc3 = {}
        actvsfc1 = {}

        actvs = {}
        actvs[0] = actvsc1
        actvs[1] = actvsc2
        actvs[2] = actvsc3
        actvs[3] = actvsfc1

        # initiate suppression states
        sc1 = {}
        sc2 = {}
        sc3 = {}
        sfc1 = {}
        
        s = {}
        s[0] = sc1
        s[1] = sc2
        s[2] = sc3
        s[3] = sfc1

        # conv1
        x = self.conv1(input[0])
        actvs[0][0] = self.relu(x)
        s[0][0] = torch.zeros(x.shape)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvs[1][0] = self.relu(x)
        s[1][0] = torch.zeros(x.shape)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvs[2][0] = x
        s[2][0] = torch.zeros(x.shape)
        
        # fc1
        x = self.dropout(x)

        if batch:  
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)

        x = self.fc1(x)
        actvs[3][0] = x # shape: batch_size x 1024
        s[3][0] = torch.zeros(x.shape)

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[t+1])               
                s_updt, s_beta_updt = self.sconv1(x, actvs[0][t], s[0][t])
                s[0][t+1] = s_updt
                x = self.relu(torch.subtract(x, s_beta_updt))
                actvs[0][t+1] = x
                x = self.pool(x)
                
                # conv2
                x = self.conv2(x)
                s_updt, s_beta_updt = self.sconv2(x, actvs[1][t], s[1][t])
                s[1][t+1] = s_updt
                x = self.relu(torch.subtract(x, s_beta_updt))
                actvs[1][t+1] = x
                x = self.pool(x)

                # conv3
                x = self.conv3(x)
                s_updt, s_beta_updt = self.sconv3(x, actvs[2][t], s[2][t])
                s[2][t+1] = s_updt
                actvs[2][t+1] = self.relu(torch.subtract(x, s_beta_updt))

                # fc1
                x = self.dropout(actvs[2][t+1]) 

                if batch:  
                    x = x.view(x.size(0), -1)
                else:
                    x = torch.flatten(x)

                x = self.fc1(x)
                s_updt, s_beta_updt = self.sfc1(x, actvs[3][t], s[3][t])
                s[3][t+1] = s_updt
                actvs[3][t+1] = torch.subtract(x, s_beta_updt)

        # only decode last timestep
        actvs[4] = self.decoder(actvs[3][t+1])

        return actvs
