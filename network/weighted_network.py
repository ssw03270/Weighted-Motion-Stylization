import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data.sampling as sampling

class WeightedNetwork(nn.Module):

    def __init__(self):
        super(WeightedNetwork, self).__init__()
        self.conv1 = nn.Conv2d(11, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=1)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=1)
        self.conv6 = nn.ConvTranspose2d(64, 11, kernel_size=1)

        self.fc = nn.Linear(8, 21)

        self.actv = nn.functional.leaky_relu

        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, style_vector):
        # encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        # x = sampling.s_down_sampling(x, 1)

        x = self.conv2(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        # x = sampling.s_down_sampling(x, 2)

        x = self.conv3(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        # x = sampling.s_down_sampling(x, 3)

        # add style vector
        style_vector = self.fc(style_vector)
        x = x * style_vector

        # decoder
        x = self.conv4(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        # x = sampling.s_up_sampling(x, 3)

        x = self.conv5(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        # x = sampling.s_up_sampling(x, 2)

        x = self.conv6(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        # x = sampling.s_up_sampling(x, 1)

        return x
