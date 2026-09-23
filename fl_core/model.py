"""Small CNN for CIFAR-10, sized to be cheap to train across many FedAvg rounds."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Two conv blocks + two FC layers. ~550K params — fast enough for many
    short local-training rounds on free-tier GPU/CPU."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # CIFAR-10 is 32x32 -> after two 2x2 pools -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
