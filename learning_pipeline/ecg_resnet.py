"""Six-lead 1D ResNet for native 500 Hz, 5,000-sample ECGs."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, padding=3
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class ECGResNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        if in_channels != 6:
            raise ValueError("Exactly six ECG leads are required")

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (6, 5000):
            raise ValueError("Expected input shape (N, 6, 5000) at 500 Hz")
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)
