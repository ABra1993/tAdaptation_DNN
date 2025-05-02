
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module_div_norm import module_div_norm
from models.module_add_supp import module_add_supp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cnn_feedforward_novelty(nn.Module):
    def __init__(self, channels, task, dataset):
        super(cnn_feedforward_novelty, self).__init__()

        # temporal dynamics
        self.task           = task
        self.dataset        = dataset

        # intervention
        self.intervention = None
        self.qdr_idx = None
        self.quadrants_coord = None
        self.onset = None

        # activation functions, pooling and dropout layers
        self.relu                   = nn.ReLU()
        self.dropout                = nn.Dropout()
        self.pool                   = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigmoid                = nn.Softmax()
        self.flatten                = nn.Flatten()

        # conv1
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5)

        # conv2 + conv3
        if self.dataset == 'cifar':
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        else:
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        # fully connected
        if channels == 1:
            self.fc1 = nn.Linear(in_features=1568, out_features=1024)
        elif channels == 3:
            self.fc1 = nn.Linear(in_features=2592, out_features=1024)

    def init_t_steps(self, t_steps=None):

        # set number of timesteps
        self.t_steps = t_steps

        # initiate readout
        self.decoder = nn.Linear(in_features=1024, out_features=10)

        # gpu
        self.decoder.to(device)

    def initialize_tempDynamics(self, tempDynamics):
        ''' Initiates the modules that apply temporal adaptation based on previous model inputs. '''

        # set temporal dynamics
        self.tempDynamics   = tempDynamics

        # initiate custom layer
        if (self.tempDynamics == 'add_supp'):
            self.sconv1 = module_add_supp()
            if self.task == 'novelty_augmented_extend_adaptation':
                self.sconv2 = module_add_supp()
                self.sconv3 = module_add_supp()
                self.sconv4 = module_add_supp()

        elif (self.tempDynamics == 'div_norm'):
            self.sconv1 = module_div_norm()
            if self.task == 'novelty_augmented_extend_adaptation':
                self.sconv2 = module_div_norm()
                self.sconv3 = module_div_norm()
                self.sconv4 = module_div_norm()

        elif (self.tempDynamics == 'l_recurrence_A') | (self.tempDynamics == 'l_recurrence_M'):
            self.sconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            if self.task == 'novelty_augmented_extend_adaptation':
                self.sconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
                self.sconv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
                self.sconv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

        # elif (self.tempDynamics == 't_recurrence_A') | (self.tempDynamics == 't_recurrence_M'):
        #     self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, output_padding=1)
            # self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, output_padding=1)
        
        # elif (self.tempDynamics == 'tl_recurrence_A') | (self.tempDynamics == 'tl_recurrence_M'):
        #     self.sconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        #     self.sconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        #     self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0)
            
    def apply_adaptation(self, t, x, layer_idx):
        ''' Applies temporal adaptation '''

        # temporal adaptation
        if (self.tempDynamics == 'add_supp'):
            self.g[layer_idx][t+1], feedback = self.sconv1(self.actvs[layer_idx][t], self.g[layer_idx][t])
            x = torch.subtract(x, feedback)

        elif (self.tempDynamics == 'div_norm'):
            self.g[layer_idx][t+1], feedback = self.sconv1(self.actvs[layer_idx][t], self.g[layer_idx][t])
            x = torch.multiply(x, feedback)

        elif (self.tempDynamics == 'l_recurrence_A'):
            x_l = self.sconv1(self.actvs[layer_idx][t])
            x = torch.add(x, x_l)
        
        elif (self.tempDynamics == 'l_recurrence_M'):
            x_l = self.sconv1(self.actvs[layer_idx][t])
            x = torch.multiply(x, x_l)
        
        # elif (self.tempDynamics == 't_recurrence_A'):
        #     x_t = self.deconv(x_pool)
        #     x_t = self.deconv(x_t)
        #     x = torch.add(x, x_t)
        
        # elif (self.tempDynamics == 't_recurrence_M'):
        #     x_t = self.deconv(x_pool)
        #     x_t = self.deconv(x_t)
        #     x = torch.multiply(x, x_t)

        # elif (self.tempDynamics == 'tl_recurrence_A'):
        #     x_l = self.sconv1(self.actvs[layer_idx][t])
        #     x = torch.add(x, x_l)
        #     if layer_idx == 0:
        #         x_t = self.deconv2(self.actvs[layer_idx+1][t])
        #         x = torch.add(x, x_t)
                
        # elif (self.tempDynamics == 'tl_recurrence_M'):
        #     x_l = self.sconv1(self.actvs[layer_idx][t])
        #     x = torch.multiply(x, x_l)
        #     if layer_idx == 0:
        #         x_t = self.deconv2(self.actvs[layer_idx+1][t])
        #         x = torch.multiply(x, x_t)
            # print(x)

        return x
    
    def set_intervention(self, value, qdr_idx, quadrants_coord, onset):
        self.intervention = value
        self.qdr_idx = qdr_idx
        self.quadrants_coord = quadrants_coord
        self.onset = int(onset)

    def forward(self, input):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # initiate activations
        actvsc1 = {}
        actvsc2 = {}
        actvsc3 = {}
        actvsc4 = {}
        fc1     = {}

        self.actvs = {}
        self.actvs[0] = actvsc1
        self.actvs[1] = actvsc2
        self.actvs[2] = actvsc3
        self.actvs[3] = actvsc4
        self.actvs[4] = fc1

        # RECURREN-dependent ------ initiate feedback states (idle with NONE and L-, T-, LT-recurrence)
        g1 = {}
        g2 = {}
        g3 = {}
        g4 = {}
        
        self.g = {}
        self.g[0] = g1
        self.g[1] = g2
        self.g[2] = g3
        self.g[3] = g4
        
        # TASK-dependent ------ idle with CORE, REPEATED_NOISE
        outp = torch.zeros(input.shape[0], 10, input.shape[1]).to(device)

        # ------- CONV1
        x = self.conv1(input[:, 0, :, :, :].to(device))
        x = self.relu(x)

        self.g[0][0] = torch.zeros(x.shape).to(device)
        self.actvs[0][0] = x

        # pooling
        x = self.pool(x)
        
        # ------- CONV2
        x = self.conv2(x)
        x = self.relu(x)

        self.g[1][0] = torch.zeros(x.shape).to(device)
        self.actvs[1][0] = x

        # pooling
        x = self.pool(x)

        # ------- CONV3
        x = self.conv3(x)
        x = self.relu(x)

        self.g[2][0] = torch.zeros(x.shape).to(device)
        self.actvs[2][0] = x

        # ------- CONV4
        x = self.conv4(x)
        x = self.relu(x)

        self.g[3][0] = torch.zeros(x.shape).to(device)
        self.actvs[3][0] = x
        
        # dropout
        x = self.dropout(x)

        # flatten output
        x = self.flatten(x)

        # ------- FC
        x = self.fc1(x)
        self.actvs[4][0] = x

        # compute model prediction for current timestep
        outp[:, :, 0] = self.decoder(x)

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # ------- CONV1
                x = self.conv1(input[:, t+1, :, :, :].to(device))
                x = self.apply_adaptation(t=t, x=x, layer_idx=0)
                
                # intervene (swap scaling)
                if (self.intervention):
                    if (int(t) >= int(self.onset - 1)):
                            
                            # Get coordinates for quadrants
                            qdr1 = torch.tensor(self.quadrants_coord[self.qdr_idx[0]]).to(device)
                            qdr2 = torch.tensor(self.quadrants_coord[self.qdr_idx[1]]).to(device)

                            # Extract the two quadrants for the current feature map
                            q_qdr1 = x[:, :, qdr1[0][0]:qdr1[0][1], qdr1[1][0]:qdr1[1][1]]
                            q_qdr2 = x[:, :, qdr2[0][0]:qdr2[0][1], qdr2[1][0]:qdr2[1][1]]

                            # compute mean
                            mean_qdr1 =  q_qdr1.mean(dim=[-1, -2], keepdim=True)
                            mean_qdr2 =  q_qdr2.mean(dim=[-1, -2], keepdim=True)

                            # compute ratio
                            ratio = mean_qdr1/mean_qdr2

                            # compute scaled activations
                            q_qdr1/=ratio + 1e-5
                            q_qdr2*=ratio 

                            # Assign the scaled values back to the tensor
                            x[:, :, qdr1[0][0]:qdr1[0][1], qdr1[1][0]:qdr1[1][1]] = q_qdr1
                            x[:, :, qdr2[0][0]:qdr2[0][1], qdr2[1][0]:qdr2[1][1]] = q_qdr2
                
                x = self.relu(x)
                self.actvs[0][t+1] = x 

                # pooling
                x = self.pool(x)
                
                # ------- CONV2
                x = self.conv2(x)
                if self.task == 'novelty_augmented_extend_adaptation':
                    x = self.apply_adaptation(t=t, x=x, layer_idx=1)
                x = self.relu(x)

                self.actvs[1][t+1] = x

                # pooling
                x = self.pool(x)

                # ------- CONV3
                x = self.conv3(x)
                if self.task == 'novelty_augmented_extend_adaptation':
                    x = self.apply_adaptation(t=t, x=x, layer_idx=2)
                x = self.relu(x)

                self.actvs[2][t+1] = x

                # ------- CONV4
                x = self.conv4(x)
                if self.task == 'novelty_augmented_extend_adaptation':
                    x = self.apply_adaptation(t=t, x=x, layer_idx=3)
                x = self.relu(x)

                self.g[3][t+1] = torch.zeros(x.shape).to(device)
                self.actvs[3][t+1] = x

                # dropout
                x = self.dropout(x) 

                # ------- FC
                x = self.flatten(x)
                x = self.fc1(x)
                self.actvs[3][t+1] = x
                
                # decoder
                outp[:, :, t+1] = self.decoder(x)

        return outp
