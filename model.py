import torch
import torch.nn as nn
from unet.better_unet import UNet

class FlowUNetwork(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FlowUNetwork, self).__init__()
        self.unet = UNet(n_classes = 1)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

    def forward(self, x):
        residual = self.unet(x)
        #residual = torch.sigmoid(residual) * 2 - 1
        # split the 4 channel image input to retrieve the intermediate mask
        #int_mask = torch.split(x.permute(1, 0, 2, 3), [self.n_classes, self.n_channels-1])[0]
        #output = residual + int_mask
        return residual