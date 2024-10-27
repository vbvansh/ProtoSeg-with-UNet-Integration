import torch
import torch.nn as nn

class R_MLP(nn.Module):
    def __init__(self, in_channels,img_size):
        super(R_MLP, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size  # Store it if needed
        # Projection layer for each spatial location
        self.projection = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # Rolling operation: shift and crop along one axis (e.g., width)
        rolled_x = torch.roll(x, shifts=1, dims=3)  # shifts along width (dim=3)

        # Get batch size, channels, height, width
        B, C, H, W = rolled_x.shape
        
        # Ensure the projection layer's input size matches the number of channels
        if C != self.in_channels:
            self.projection = nn.Linear(C, C).to(x.device)  # Create a new layer if input channels differ
        
        # Reshape to (batch_size * height * width, channels) for linear layer
        rolled_x = rolled_x.reshape(B * H * W, C)  # Change view() to reshape()
        
        # Apply the linear projection per spatial location
        out = self.projection(rolled_x)  # (batch_size * height * width, in_channels)
        
        # Reshape back to (batch_size, channels, height, width)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # Permute to match original dimensions

        return out
