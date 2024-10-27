import torch
import torch.nn as nn
import torch.nn.functional as F
from .or_dor import DOR_MLP


class Lo2_Block(nn.Module):
    def __init__(self, in_channels, img_size):
        super(Lo2_Block, self).__init__()
        self.dor_mlp = DOR_MLP(in_channels, img_size)
        self.dsc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)  # Depthwise separable conv

    def forward(self, x):
        dor_mlp_out = self.dor_mlp(x)
        dsc_out = self.dsc(x)
        return dor_mlp_out + dsc_out
