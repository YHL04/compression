

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class Torso(nn.Module):
    def __init__(self, dim_x=32, dim_y=32, channels=3):
        super(Torso, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(21632, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        return x


class Model(nn.Module):

    def __init__(self, dim_x, dim_y, channels, n_basis=64, n_layers=2):
        super(Model, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        self.torso = Torso(dim_x=dim_x, dim_y=dim_y, channels=channels)

        # params
        self.n_basis = n_basis
        self.n_layers = n_layers

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.channels = channels

        # basis to learn
        self.basis = nn.Parameter(torch.randn((dim_x * dim_y) * n_basis,))

        # model torso
        self.last = nn.Linear(1024, n_basis * channels)

    def forward(self, x):
        """
        Parameters:
            x with shape (B, N)

        Returns:
            linear combination of basis
        """
        B = x.size(0)

        # pass through model
        # x = self.resnet(x)
        x = self.torso(x)
        x = self.last(x).view(B, self.n_basis, self.channels, 1, 1)

        # get learnable basis
        b = self.basis.reshape(1, self.n_basis, 1, self.dim_x, self.dim_y
                               ).repeat(B, 1, 1, 1, 1).repeat(1, 1, self.channels, 1, 1)

        # check shape
        assert x.shape == (B, self.n_basis, self.channels, 1, 1)
        assert b.shape == (B, self.n_basis, self.channels, self.dim_x, self.dim_y)

        # reconstruct image through linear combination of basis
        img = (x * b).sum(dim=1)

        # return reconstructed image, and basis coefficients
        return img, x

