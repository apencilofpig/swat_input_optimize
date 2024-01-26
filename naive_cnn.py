import torch
from torch import nn


class Naive_CNN(nn.Module):
    def __init__(self):
        super(Naive_CNN, self).__init__()
        self.embedding = nn.Embedding(51, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, (5, 1), padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, (5, 1), padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, (5, 1), padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 36)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x = self.fc(x)
        return x
