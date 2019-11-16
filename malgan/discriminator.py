from typing import List

import torch
from torch import Tensor
import torch.nn as neural_net

class Discriminator(neural_net.Module):
    EPS = 1e-7

    def __init__(self, width_of_malware: int, hidden_layer_size: List[int], activation_fct: neural_net.Module):
        super().__init__()

        self._layers = neural_net.Sequential()
        for counter, (in_width, out_width) in enumerate(zip([width_of_malware] + hidden_layer_size[:-1], hidden_layer_size)):
            layer = neural_net.Sequential(neural_net.Linear(in_width, out_width), activation_fct)
            self._layers.add_module("FF%02d" % counter, layer)

        layer = neural_net.Sequential(neural_net.Linear(hidden_layer_size[-1], 1), neural_net.Sigmoid())
        self._layers.add_module("FF%02d" % len(hidden_layer_size), layer)

    def forward(self, example_tensor: Tensor) -> Tensor:
        d_theta = self._layers(example_tensor)

        return torch.clamp(d_theta, self.EPS, 1. - self.EPS).view(-1)
