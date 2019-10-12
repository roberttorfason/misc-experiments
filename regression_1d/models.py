import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)


class FcNetwork(nn.Module):
    def __init__(self, input_size, activation=nn.ReLU):
        super().__init__()
        self._linear0 = nn.Linear(input_size, 20)
        self._mean = nn.Linear(20, 1)
        self._beta = nn.Linear(20, 1, bias=False)

        self._activation = activation()

        self.apply(_init_weights)

    def forward(self, x):
        out = self._linear0(x)
        out = self._activation(out)

        mean = self._mean(out)
        beta = torch.exp(self._beta(out))

        return mean, beta
