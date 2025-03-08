import copy
import torch

from torch import nn, Tensor
from typing import Optional

from utils import _get_activation_fn
import copy

class Transformer(nn.Module):
    def __init__(self, num_points, d_model=256):
        super(Transformer, self).__init__()
        # structure encoding
        self.structure_encoding = nn.Parameter(torch.randn(1, num_points, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        bs, num_feat, len_feat = src.size()

        structure_encoding = self.structure_encoding.repeat(bs, 1, 1).permute(1, 0, 2)

        return structure_encoding
