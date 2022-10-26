import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import matplotlib.pyplot as plt
from models.Positive import Positive
from models.Zero import Zero
import torch.nn as nn
from models.module_div_norm import module_div_norm
import torch.nn.functional as F
import torch.nn.utils.parametrize as P



class cnn_feedforward_div_norm(nn.Module):

    def __init__(self, batchsiz=100, t_steps=3):
        super(cnn_feedforward_div_norm, self).__init__()

        # training variables
        self.t_steps = t_steps
        self.batchsiz = batchsiz

        # layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        P.register_parametrization(self.conv1, 'weight', Positive())
        P.register_parametrization(self.conv1, 'bias', Zero())
        self.sconv1 = module_div_norm(self.batchsiz, 24, 24, 32)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        # self.sconv2 = module_div_norm(8, 8, 32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.sconv3 = module_div_norm(2, 2, 32)
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

        # initiate activations           
        actvsc1     = torch.zeros(self.t_steps, self.batchsiz, 32, 24, 24)
        actvsc1_s   = torch.zeros(self.t_steps, self.batchsiz, 32, 24, 24)
        actvsc2     = torch.zeros(self.t_steps, self.batchsiz, 32, 8, 8)
        actvsc3     = torch.zeros(self.t_steps, self.batchsiz, 32, 2, 2)
        actvsfc1    = torch.zeros(self.t_steps, self.batchsiz, 1024)

        # conv1
        x = self.conv1(input[0])
        actvsc1[0, :, :, :, :] = self.relu(x)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvsc2[0, :, :, :, :] = self.relu(x)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvsc3[0, :, :, :, :] = x
        
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

                # conv1
                x = self.conv1(input[t])
                actvsc1[t, :, :, :, :] = x
                irf_inv, irf_norm_inv, conv_input_drive, input_drive, conv_normrsp, conv_exp_normrsp, normrsp, r_dn = self.sconv1(actvsc1[0:t, :, :, :, :], t, self.t_steps)
                r_dn = self.sconv1(actvsc1[0:t, :, :, :, :], t, self.t_steps)
                actvsc1_s[t, :, :, :, :] = r_dn
                # r[t, :, :, :, :] = x            
                x = self.relu(r_dn)
                x = self.pool(x) 
                
                # conv2
                x = self.conv2(x)
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


        # # create figure to track DN model
        # fig = plt.figure()
        # axs = plt.gca()
        # i = 0
        # j = 0
        # k = 0
        # l = 0
        # lw = 2

        # # activation
        # axs.plot(actvsc1[:, i, j, k, l], 'k', label='$S/a_{c,h,w}$', lw=lw, alpha=0.7)

        # # impulse response functiona
        # # axs.plot(irf[:, i, j, k, l], label=r'$irf$', lw=lw, alpha=0.5, color='green', linestyle='--')
        # axs.plot(torch.arange(self.t_steps)-self.t_steps+1, irf_inv[:, i, j, k, l], label=r'$irf_{inv}$', lw=lw, alpha=0.5, color='blue', linestyle='--')
        # axs.plot(torch.arange(self.t_steps)-self.t_steps+1, irf_norm_inv[:, i, j, k, l], label=r'$irf_{norm}$', lw=lw, alpha=0.7, color='orange', linestyle='--')
        # # axs.plot(conv_normrsp[:, i, j, k, l], label=r'$L \ast h_{2}$', lw=lw, alpha=0.7, color='orange')

        # axs.plot(conv_input_drive[:, i, j, k, l], label=r'$L$', lw=lw, alpha=0.7, color='green')
        # axs.plot(input_drive[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='blue')
        # # axs.plot(input_drive_cross[:, i, j, k, l], label=r'$|L|^{n}$', lw=lw, alpha=0.7, color='purple')

        # # axs.plot(t, conv_exp_normrsp[:, i, j, k, l], label=r'$|L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='red')
        # axs.plot(normrsp[:, i, j, k, l], label=r'$\sigma^{n} + |L \ast h_{2}|^{n}$', lw=lw, alpha=0.7, color='purple')
        
        # axs.plot(actvsc1_s[:, i, j, k, l], label=r'$r_{DN}$', color='red')
        # plt.legend()
        # plt.show()

        # # only decode last timestep
        # actvs_decoder = self.decoder(actvsfc1[t, :, :])

        # combine in dictionairy
        actvs = {}
        actvs[0] = actvsc1
        actvs[1] = actvsc2
        actvs[2] = actvsc3
        actvs[3] = actvsfc1
        actvs[4] = actvs_decoder


        return actvs
