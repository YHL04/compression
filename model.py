

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class Model(nn.Module):

    def __init__(self, N=64, n_basis=64, n_layers=2):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)

        # params
        self.N = N
        self.n_basis = n_basis
        self.n_layers = n_layers

        # basis to learn
        self.basis = nn.Parameter(torch.randn(N * n_basis,))

        # model torso
        self.last = nn.Linear(1024, n_basis)

    def forward(self, x):
        """
        Parameters:
            x with shape (B, N)

        Returns:
            linear combination of basis
        """
        B = x.size(0)

        # pass through model
        x = self.resnet(x)
        x = self.last(x).unflatten(-1)

        # get learnable basis
        b = self.basis.reshape(1, self.n_basis, self.N).repeat(B, 1, 1)

        # check shape
        assert x.shape == (B, self.n_basis)
        assert b.shape == (B, self.n_basis, self.N)

        # reconstruct image through linear combination of basis
        img = x * b

        # return reconstructed image, and basis coefficients
        return img, x

