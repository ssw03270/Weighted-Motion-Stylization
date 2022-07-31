import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data.sampling as sampling

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=1)

        self.ResBlk1 = ResBlock(64, 128)
        self.ResBlk2 = ResBlock(128, 256)

        self.TransResBlk1 = TransResBlock(256, 128)
        self.TransResBlk2 = TransResBlock(128, 64)

        self.conv2 = nn.ConvTranspose2d(64, 7, kernel_size=1)

        self.actv = nn.functional.leaky_relu

        self.s_down = sampling.S_Down_Sampling()
        self.s_up = sampling.S_Up_Sampling()
        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)
        self.AdaIN1 = AdaIN(8, 128)
        self.AdaIN2 = AdaIN(8, 64)


    def forward(self, x, style):
        # encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)

        # add style vector
        x = self.IN1(x)

        x = self.ResBlk1(x)
        x = self.t_down(x)
        x = self.s_down(x, 1)

        # add style vector
        x = self.IN2(x)

        x = self.ResBlk2(x)
        x = self.t_down(x)
        x = self.s_down(x, 2)

        # decoder
        x = self.TransResBlk1(x)
        x = self.t_up(x)
        x = self.s_up(x, 1)

        # add style vector
        x = self.AdaIN1(x, style)

        x = self.TransResBlk2(x)
        x = self.t_up(x)
        x = self.s_up(x, 2)

        # add style vector
        x = self.AdaIN2(x, style)

        x = self.conv2(x)
        x = self.actv(x, 0.2)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=1)
        self.ResBlk1 = ResBlock(64, 128)
        self.ResBlk2 = ResBlock(128, 256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(16, 5), stride=1)
        self.conv3 = nn.Conv2d(256, 8, kernel_size=(1, 1))

        self.actv = nn.functional.leaky_relu
        self.actv2 = torch.sigmoid

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(8, 8)

        self.s_down = sampling.S_Down_Sampling()
        self.s_up = sampling.S_Up_Sampling()
        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, y):
        # same structure with encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)

        # add style vector
        x = self.IN1(x)

        x = self.ResBlk1(x)
        x = self.t_down(x)
        x = self.s_down(x, 1)

        x = self.dropout(x)

        # add style vector
        x = self.IN2(x)

        x = self.ResBlk2(x)
        x = self.t_down(x)
        x = self.s_down(x, 2)

        x = self.dropout(x)

        #
        x = self.actv(x, 0.2)
        x = self.conv2(x)
        x = self.actv(x, 0.2)
        x = self.conv3(x)

        #
        x = x.view(x.size(0), -1)
        idx = range(y.size(0))
        x = x[idx, y]
        x = self.actv2(x)
        return x


class StyleEncoder(nn.Module):

    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=1)
        self.ResBlk1 = ResBlock(64, 128)
        self.ResBlk2 = ResBlock(128, 256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(16, 5))
        self.conv3 = nn.Conv2d(256, 8, kernel_size=(1, 1))

        self.actv = nn.functional.leaky_relu
        self.actv2 = torch.sigmoid

        self.fc = nn.Linear(8, 8)

        self.s_down = sampling.S_Down_Sampling()
        self.s_up = sampling.S_Up_Sampling()
        self.t_down = sampling.T_Down_Sampling()
        self.t_up = sampling.T_Up_Sampling()

        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        # same structure with encoder
        x = self.conv1(x)
        x = self.actv(x, 0.2)

        # add style vector
        x = self.IN1(x)

        x = self.ResBlk1(x)
        x = self.t_down(x)
        x = self.s_down(x, 1)

        # add style vector
        x = self.IN2(x)

        x = self.ResBlk2(x)
        x = self.t_down(x)
        x = self.s_down(x, 2)

        #
        x = self.actv(x, 0.2)
        x = self.conv2(x)
        x = self.actv(x, 0.2)
        x = self.conv3(x)

        #
        x = x.view(x.size(0), -1)

        return x


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResBlock, self).__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
        )
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.lRelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.residual_block(x)  # F(x)
        out = out + self.conv(x)  # F(x) + x
        out = self.lRelu(out)
        return out

class TransResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TransResBlock, self).__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_in, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=3, padding=1),
        )
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.lRelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.residual_block(x)  # F(x)
        out = out + self.conv(x)  # F(x) + x
        out = self.lRelu(out)
        return out