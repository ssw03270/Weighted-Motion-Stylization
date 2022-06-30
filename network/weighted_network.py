import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data.sampling as sampling

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(11, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.ConvTranspose2d(64, 11, kernel_size=1)

        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(8, 10)

        self.actv = nn.functional.leaky_relu

        self.s_down = sampling.S_Down_Sampling()
        self.s_up = sampling.S_Up_Sampling()
        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, style_vector):
        # encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 1)

        x = self.conv2(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 2)

        x = self.conv3(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 3)

        # decoder
        x = self.conv4(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        x = self.s_up(x, 3)

        # add style vector
        style_vector1 = self.fc1(style_vector)
        x = x * style_vector1

        x = self.conv5(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        x = self.s_up(x, 2)

        # add style vector
        style_vector2 = self.fc2(style_vector)
        x = x * style_vector2

        x = self.conv6(x)
        x = self.actv(x, 0.2)
        x = self.t_up(x)
        x = self.s_up(x, 1)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(11, 64, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(8, 2))
        self.conv5 = nn.Conv2d(256, 8, kernel_size=(1, 1))

        self.actv = nn.functional.leaky_relu

        self.fc = nn.Linear(8, 8)

        self.s_down = sampling.S_Down_Sampling()
        self.s_up = sampling.S_Up_Sampling()
        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        # same structure with encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 1)

        x = self.conv2(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 2)

        x = self.conv3(x)
        x = self.actv(x, 0.2)
        x = self.t_down(x)
        x = self.s_down(x, 3)

        #
        x = self.actv(x, 0.2)
        x = self.conv4(x)
        x = self.actv(x, 0.2)
        x = self.conv5(x)

        #
        x = torch.reshape(x, (-1,))
        x = self.fc(x)

        return x