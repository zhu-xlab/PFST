import torch.nn as nn
from ..builder import DISCRIMINATORS

@DISCRIMINATORS.register_module()
class FCDiscriminator(nn.Module):
    def __init__(self, num_in_channels, ndf=64):
        super(FCDiscriminator, self).__init__()
        net = nn.Sequential(
            nn.Conv2d(num_in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.net = net

    def forward(self, x):
        return self.net(x)
