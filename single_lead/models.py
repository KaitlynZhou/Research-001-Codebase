import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,out_channels,7,stride=stride,padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels,out_channels,7,padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = (nn.Sequential(nn.Conv1d(in_channels,out_channels,1,stride=stride),nn.BatchNorm1d(out_channels))
                         if stride != 1 or in_channels != out_channels else nn.Identity())
    def forward(self,x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

def stem_branch(n,k):
    return nn.Sequential(nn.Conv1d(1,n,k,padding=k//2),nn.BatchNorm1d(n),nn.ReLU())

class MultiScaleStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_small = stem_branch(8,5)
        self.branch_medium = stem_branch(12,11)
        self.branch_large = stem_branch(12,21)
    def forward(self,x):
        return torch.cat([self.branch_small(x),self.branch_medium(x),self.branch_large(x)],dim=1)

class SingleLeadResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = stem_branch(32,7)
        self.features = nn.Sequential(ResidualBlock(32,32),ResidualBlock(32,64,2),ResidualBlock(64,128,2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128,2)
    def forward(self,x):
        return self.classifier(self.pool(self.features(self.stem(x))).squeeze(-1))

class ImprovedSingleLeadResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = MultiScaleStem()
        self.features = nn.Sequential(ResidualBlock(32,32),ResidualBlock(32,64,2),ResidualBlock(64,128,2))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(.2)
        self.classifier = nn.Linear(256,2)
    def forward(self,x):
        x = self.features(self.stem(x))
        return self.classifier(self.dropout(torch.cat([self.avg_pool(x).squeeze(-1),self.max_pool(x).squeeze(-1)],dim=1)))

ARCHITECTURES = {"baseline": SingleLeadResNet, "improved": ImprovedSingleLeadResNet}
