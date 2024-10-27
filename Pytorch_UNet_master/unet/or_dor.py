import torch
import torch.nn as nn
import torch.nn.functional as F
from .r_mlp import R_MLP



class OR_MLP(nn.Module):
    def __init__(self, in_channels, img_size):
        super(OR_MLP, self).__init__()
        self.r_mlp_width = R_MLP(in_channels, img_size)
        self.r_mlp_height = R_MLP(in_channels, img_size)

    def forward(self, x):
        out_width = self.r_mlp_width(x)
        out_height = self.r_mlp_height(x.permute(0, 1, 3, 2))  # permute to apply along height
        return out_width + out_height.permute(0, 1, 3, 2)

class DOR_MLP(nn.Module):
    def __init__(self, in_channels, img_size):
        super(DOR_MLP, self).__init__()
        self.or_mlp_1 = OR_MLP(in_channels, img_size)
        self.or_mlp_2 = OR_MLP(in_channels, img_size)

    def forward(self, x):
        out_1 = self.or_mlp_1(x)
        out_2 = self.or_mlp_2(x)
        return out_1 + out_2
