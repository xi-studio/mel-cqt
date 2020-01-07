import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BasicBlock_0(nn.Module):
    def __init__(self, c):
        super(BasicBlock_0, self).__init__()

        self.t = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Tanh(),
        )
        self.s = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        self.end = nn.Conv1d(c, c, 1)


    def forward(self, x):
        identity = x
        out = self.t(x) * self.s(x)
        out += identity
        out = torch.relu(self.end(out))

        return x


class BasicBlock_1(nn.Module):
    def __init__(self, c):
        super(BasicBlock_1, self).__init__()
        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_0(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class BasicBlock_2(nn.Module):
    def __init__(self, c):
        super(BasicBlock_2, self).__init__()

        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_1(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class BasicBlock_3(nn.Module):
    def __init__(self, c):
        super(BasicBlock_3, self).__init__()

        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_2(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.start = nn.Conv1d(128, 256, 1)
        self.conv1 = BasicBlock_2(256)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.conv3 = nn.Conv1d(512, 24, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(self.conv3(x))

        return x


