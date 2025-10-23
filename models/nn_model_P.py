import torch
from torch import nn
import torch.nn.functional as F
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


def create_mlp(input_dim, output_dim, net_arch, activation_fn=Swish):
    """
    returns: a list of nn.Module
    """
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        # modules.append(nn.LayerNorm(net_arch[idx]))
        modules.append(activation_fn())
    last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
    modules.append(nn.Linear(last_layer_dim, output_dim))
    return nn.Sequential(*modules)


class Model_Px(nn.Module):

    def __init__(self, x_dim: int, net_arch: List[int], activation_fn=Swish):
        super().__init__()

        self.x_dim = x_dim
        self.mlp = create_mlp(input_dim=x_dim,
                              output_dim=x_dim ** 2,
                              net_arch=net_arch,
                              activation_fn=activation_fn)

    def forward(self, x):
        """
        Args:
            x : shape(..., x_dim)
        Returns
            P(x) shape(B, x_dim, x_dim)
        """
        P_flat = self.mlp(x)  # shape(..., x_dim**2)
        P_square = P_flat.view(*P_flat.shape[:-1], self.x_dim, self.x_dim)
        P_sym_positive = P_square.transpose(-2, -1) @ P_square
        return P_sym_positive
