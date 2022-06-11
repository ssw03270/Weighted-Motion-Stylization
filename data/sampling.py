import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class S_Down_Sampling(nn.Module):
    def __init__(self, mode):
        super(S_Down_Sampling, self).__init__()
        node_map_1 = [[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [0, 9],
                      [10, 11, 12],
                      [13, 14],
                      [15, 16],
                      [17, 18],
                      [19, 20]]
        node_map_2 = [[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]]
        node_map_3 = [[2, 3, 4],
                      [0, 1]]
        self.node_map = []
        if mode == 1:
            self.node_map = node_map_1
        elif mode == 2:
            self.node_map = node_map_2
        elif mode == 3:
            self.node_map = node_map_3

    def forward(self, data2):
        new_data2 = []
        for data in data2:
            new_data = []
            for skeleton_data in data:
                new_skeleton_data = []
                for map in self.node_map:
                    weight_sum = 0
                    for idx in map:
                        weight_sum += skeleton_data[idx]
                    weight_sum /= len(map)
                    new_skeleton_data.append(weight_sum)
                new_data.append(new_skeleton_data)
            new_data2.append(new_data)
        return new_data2

class S_Up_Sampling(nn.Module):
    def __init__(self, mode):
        super(S_Up_Sampling, self).__init__()
        node_map_1 = [[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [0, 9],
                      [10, 11, 12],
                      [13, 14],
                      [15, 16],
                      [17, 18],
                      [19, 20]]
        node_map_2 = [[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]]
        node_map_3 = [[2, 3, 4],
                      [0, 1]]

        self.node_map = []
        if mode == 1:
            self.node_map = node_map_1
        elif mode == 2:
            self.node_map = node_map_2
        elif mode == 3:
            self.node_map = node_map_3

    def forward(self, data2):
        new_data2 = []
        for data in data2:
            new_data = []
            for skeleton_data in data:
                new_skeleton_data = []
                for i in range(len(self.node_map)):
                    for j in range(len(self.node_map[i])):
                        new_skeleton_data.append(skeleton_data[i])
                new_data.append(new_skeleton_data)
            new_data2.append(new_data)
        return new_data2

class T_Down_Sampling(nn.Module):
    def __init__(self):
        super(T_Down_Sampling, self).__init__()

    def forward(self, data2):
        data2 = data2.view(1, 1, len(data2), int(len(data2[0])), len(data2[0][0]))
        data2 = F.interpolate(data2, scale_factor=(1, 0.5, 1), mode='trilinear', align_corners=True)
        # data2 = data2.view(len(data2[0][0]), int(len(data2[0][0][0])), len(data2[0][0][0][0]))
        data2 = data2.squeeze()
        return data2

class T_Up_Sampling(nn.Module):
    def __init__(self):
        super(T_Up_Sampling, self).__init__()

    def forward(self, data2):
        data2 = data2.view(1, 1, len(data2), int(len(data2[0])), len(data2[0][0]))
        data2 = F.interpolate(data2, scale_factor=(1, 2, 1), mode='trilinear', align_corners=True)
        # data2 = data2.view(len(data2[0][0]), int(len(data2[0][0][0])), len(data2[0][0][0][0]))
        data2 = data2.squeeze()
        return data2
