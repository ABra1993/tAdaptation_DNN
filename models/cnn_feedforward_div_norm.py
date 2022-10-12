import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
# print(os.getcwd())
import matplotlib.pyplot as plt

import torch.nn as nn
from models.module_exp_div_norm import module_div_norm
import torch.nn.functional as F


class cnn_feedforward_div_norm(nn.Module):

    def __init__(self, batchsiz=1, t_steps=3):
        super(cnn_feedforward_div_norm, self).__init__()

        # training variables
        self.t_steps = t_steps
        self.batchsiz = batchsiz

        # layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.sconv1 = module_div_norm(32, 24, 24)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        # self.sconv2 = module_div_norm(32, 8, 8)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.sconv3 = module_div_norm(32, 2, 2)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        # self.sfc1 = module_div_norm('none', 'none', 1024)

        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train

    def forward(self, input, batch=True):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # output sizes per layer
        # torch.Size([100, 32, 24, 24])
        # torch.Size([100, 32, 8, 8])
        # torch.Size([100, 32, 2, 2])
        # torch.Size([100, 1024])

        #print progress
        print('-------- Timestep: ', 1)

        # initiate activations
        actvsc1 = torch.zeros([self.t_steps, self.batchsiz, 32, 24, 24])
        actvsc2 = torch.zeros([self.t_steps, self.batchsiz, 32, 8, 8])
        actvsc3 = torch.zeros([self.t_steps, self.batchsiz, 32, 2, 2])
        actvsfc1 = torch.zeros([self.t_steps, self.batchsiz, 1024])

        # conv1
        x = self.conv1(input[0])
        actvsc1[0, :, :, :, :] = self.relu(x)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvsc2[0, :, :, :] = self.relu(x)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvsc3[0, :, :, :] = x
        
        # fc1
        x = self.dropout(x)

        if batch:  
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)

        x = self.fc1(x)
        actvsfc1[0, :, :] = x # shape: batch_size x 1024

        if self.t_steps > 0:
            for t in range(1, self.t_steps):

                # print progress
                print('-------- Timestep: ', t+1)

                # conv1
                x = self.conv1(input[t])               
                x = self.relu(x)                # only non-negative values
                # print('B: ', x.shape)
                # print(x)
                irf, norm_irf, x = self.sconv1(actvsc1[0:t, :, :, :, :], t, self.t_steps)
                plt.figure()
                plt.plot(torch.mean(irf[:, 0, 0, :], axis=0), label='L')
                plt.plot(torch.mean(norm_irf[0, 0, :, :], axis=0), label='norm_irf')
                plt.legend()
                plt.show()
                plt.close()
                # print(x)
                # x = torch.sigmoid(x)
                # print('B: ', x.shape)
                # if t == 2:
                #     print(x)
                # print(x.shape)
                actvsc1[t, :, :, :, :] = x
                x = self.pool(x)
                
                
                # conv2
                # print(x) # NO NaN
                x = self.conv2(x)
                # print(x) # NaN
                # print('A: ', x.shape)
                x = self.relu(x)
                # x = self.sconv1(actvsc2[0:t, :, :, :, :], t)
                actvsc2[t, :, :, :, :] = x
                x = self.pool(x)

                # conv3
                x = self.conv3(x)
                x = self.relu(x)
                # x = self.sconv1(actvsc3[0:t, :, :, :, :], t)
                actvsc3[t, :, :, :, :] = x

                # fc1
                x = self.dropout(actvsc3[t, :, :, :, :]) 

                if batch:  
                    x = x.view(x.size(0), -1)
                else:
                    x = torch.flatten(x)

                x = self.fc1(x) 
                actvsfc1[t, :, :] = x

        # only decode last timestep
        # print(actvsfc1[t, :, :])
        actvs_decoder = self.decoder(actvsfc1[t, :, :])
        # print(actvs_decoder)

        # combine in dictionairy
        actvs = {}
        actvs[0] = actvsc1
        actvs[1] = actvsc2
        actvs[2] = actvsc3
        actvs[3] = actvsfc1
        actvs[4] = actvs_decoder

        return actvs
