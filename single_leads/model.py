"""The six-lead branch's three-block 1D ResNet with a configurable input-channel count (1 here)."""
import torch.nn as nn
from common import six_lead_model

ResidualBlock = six_lead_model.ResidualBlock


class ECGResNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU())
        self.features = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 64, stride=2), ResidualBlock(64, 128, stride=2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.stem(x))).squeeze(-1))
