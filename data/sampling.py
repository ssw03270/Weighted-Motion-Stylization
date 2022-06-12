import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class S_Down_Sampling(nn.Module):
    def __init__(self, mode=1):
        super(S_Down_Sampling, self).__init__()
        self.node_map_1 = [[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [0, 9],
                      [10, 11, 12],
                      [13, 14],
                      [15, 16],
                      [17, 18],
                      [19, 20]]
        self.node_map_2 = [[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]]
        self.node_map_3 = [[2, 3, 4],
                      [0, 1]]
        self.node_map = []

    def forward(self, x, mode):
        if mode == 1:
            self.node_map = self.node_map_1
        elif mode == 2:
            self.node_map = self.node_map_2
        elif mode == 3:
            self.node_map = self.node_map_3

        xxx = []
        for map in self.node_map:
            xx = x[:, :, map]
            xx = xx.view(1, 1, len(xx), int(len(xx[0])), len(xx[0][0]))
            xx = F.interpolate(xx, scale_factor=(1, 1, 0.5), mode='trilinear', align_corners=True)
            xx = torch.squeeze(xx, 0)
            xx = torch.squeeze(xx, 0)
            xxx.append(xx)
        x = torch.stack(xxx, dim=2)
        x = x.view([len(xx), len(xx[0]), -1])
        return x

class S_Up_Sampling(nn.Module):
    def __init__(self, mode=1):
        super(S_Up_Sampling, self).__init__()
        self.node_map_1 = [[1, 2],
                           [3, 4],
                           [5, 6],
                           [7, 8],
                           [0, 9],
                           [10, 11, 12],
                           [13, 14],
                           [15, 16],
                           [17, 18],
                           [19, 20]]
        self.node_map_2 = [[0, 1],
                           [2, 3],
                           [4, 5],
                           [6, 7],
                           [8, 9]]
        self.node_map_3 = [[2, 3, 4],
                           [0, 1]]
        self.node_map = []

    def forward(self, x, mode):
        if mode == 1:
            self.node_map = self.node_map_1
        elif mode == 2:
            self.node_map = self.node_map_2
        elif mode == 3:
            self.node_map = self.node_map_3

        xxx = []
        for i, map in enumerate(self.node_map):
            xx = x[:, :, i]
            xx = xx.view(1, 1, len(xx), int(len(xx[0])), -1)
            xx = F.interpolate(xx, scale_factor=(1, 1, len(map)), mode='trilinear', align_corners=True)
            xx = torch.squeeze(xx, 0)
            xx = torch.squeeze(xx, 0)
            xxx.append(xx)
        x = torch.cat(xxx, dim=2)
        x = x.view([len(xx), len(xx[0]), -1])
        return x

class T_Down_Sampling(nn.Module):
    def __init__(self):
        super(T_Down_Sampling, self).__init__()

    def forward(self, x):
        x = x.view(1, 1, len(x), int(len(x[0])), len(x[0][0]))
        x = F.interpolate(x, scale_factor=(1, 0.5, 1), mode='trilinear', align_corners=True)
        x = x.squeeze()
        return x

class T_Up_Sampling(nn.Module):
    def __init__(self):
        super(T_Up_Sampling, self).__init__()

    def forward(self, x):
        x = x.view(1, 1, len(x), int(len(x[0])), len(x[0][0]))
        x = F.interpolate(x, scale_factor=(1, 2, 1), mode='trilinear', align_corners=True)
        x = x.squeeze()
        return x
