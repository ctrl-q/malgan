from typing import List, Tuple

import torch
import torch.nn as neural_net
from torch import Tensor

TensorTuple = Tuple[Tensor, Tensor]


class Generator(neural_net.Module):
    def __init__(self, dim_feature_vect: int, dim_noise_vect: int, hidden_size: List[int], activation_fct: neural_net.Module):
        super().__init__()

        self._Z = dim_noise_vect

        self._layers, dim = neural_net.Sequential(), [dim_feature_vect + self._Z] + hidden_size
        for counter, (dim_in, dim_out) in enumerate(zip(dim[:-1], dim[1:])):
            self._layers.add_module("FF%02d" % counter, neural_net.Sequential(neural_net.Linear(dim_in, dim_out), activation_fct))

        layer = neural_net.Sequential(neural_net.Linear(dim[-1], dim_feature_vect), neural_net.Sigmoid())
        self._layers.add_module("FF%02d" % len(dim), layer)

    def forward(self, input_vect: torch.Tensor, noise_vect: torch.Tensor = None) -> TensorTuple:
        if noise_vect is None:
            num_ele = input_vect.shape[0]
            noise_vect = torch.rand((num_ele, self._Z))

        concat = torch.cat((input_vect, noise_vect), dim=1)
        concat = self._layers.forward(concat)
        g_theta = torch.max(input_vect, concat)  # Ensure binary bits only set positive

        m_prime = (g_theta > 0.5).float()
        return m_prime, g_theta
