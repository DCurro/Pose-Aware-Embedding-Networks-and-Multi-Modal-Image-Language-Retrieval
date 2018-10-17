import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

        self.mask = Parameter(torch.ones(1, 128))

    def forward(self, x):
        return x * self.mask.repeat(x.size(0), 1)