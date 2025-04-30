from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(SimpleFCN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class PixelMLPSegmentation(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)
