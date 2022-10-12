import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_feedforward(nn.Module):
    def __init__(self, t_steps=3):
        super(cnn_feedforward, self).__init__()

        # training variables
        self.t_steps = t_steps

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        
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

        # conv1
        x = self.conv1(input[0])
        actvs[0][0] = self.relu(x)
        x = self.pool(x)

        # conv2
        x = self.conv2(x)
        actvs[1][0] = self.relu(x)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvs[2][0] = x

        # fc1
        x = self.dropout(x)  

        if batch:  
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)

        actvs[3][0] = self.fc1(x)

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[t+1])               
                actvs[0][t+1] = self.relu(x)
                x = self.pool(x)

                # conv2
                x = self.conv2(x)
                actvs[1][t+1] = self.relu(x)
                x = self.pool(x)

                # conv3
                x = self.conv3(x)
                actvs[2][t+1] = self.relu(x)

                # fc1
                x = self.dropout(actvs[2][t+1]) 

                if batch:  
                    x = x.view(x.size(0), -1)
                else:
                    x = torch.flatten(x)

                actvs[3][t+1] = self.fc1(x)

        actvs[4] = self.decoder(actvs[3][t+1])

        return actvs
