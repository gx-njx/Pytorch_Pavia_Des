import torch.nn as nn


class CNN_Pa1(nn.Module):
    def __init__(self, img_size, num_class):
        super(CNN_Pa1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.fullconnect = nn.Linear(16 * img_size, num_class)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        output = self.fullconnect(x)
        return output
