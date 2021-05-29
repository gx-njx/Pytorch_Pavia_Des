import torch.nn as nn


class CNN_Pa(nn.Module):
    def __init__(self, img_size, num_class):
        super(CNN_Pa, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fullconnect = nn.Linear(32 * (img_size // 4) * (img_size // 4), num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fullconnect(x)
        return output
