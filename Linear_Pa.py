import torch.nn as nn


class Linear_Pa(nn.Module):
    def __init__(self, features, num_class):
        super(Linear_Pa, self).__init__()
        self.Linear1 = nn.Sequential(
            nn.Linear(features, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(10, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.Linear3 = nn.Sequential(
            nn.Linear(100, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.FC = nn.Linear(10, num_class)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        x = self.FC(x)
        return x
