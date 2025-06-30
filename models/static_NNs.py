import torch
from torch import nn


class FCNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, act=nn.Tanh):
        super(FCNN, self).__init__()
        print("!!!!! We are using the FCNN of the static_NN file!!!!!!!")
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False), act(),
            nn.Linear(dim_hidden, dim_hidden, bias=False), act(),
            nn.Linear(dim_hidden, dim_out, bias=False)
            # nn.Linear(dim_in, dim_out, bias=False)
        )
        # self.network = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        return self.network(x)
