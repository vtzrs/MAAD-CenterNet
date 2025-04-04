# ------------------------------------------------------------------------------
# Created by Vasileios Tzouras, 2024
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    """
    Domain Discriminator.
    """

    def __init__(self, in_channels=64):
        """
        Initialize the Discriminator module.

        Args:
            in_channels (int): Number of input channels.
        """
        super(Discriminator, self).__init__()
        torch.manual_seed(111)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    a=0.2,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the domain
            discriminator.
        """
        x = self.layers(x)
        x = x.view(-1, 1)
        return x
