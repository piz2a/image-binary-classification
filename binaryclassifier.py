import torch.nn as nn


def conv_block(c_in, c_out, dropout, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
        nn.BatchNorm2d(num_features=c_out),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    )


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 256, kernel_size=5, padding=2, dropout=0.1)
        self.conv2 = conv_block(256, 128, kernel_size=3, padding=1, dropout=0.1)
        self.conv3 = conv_block(128, 64, kernel_size=3, padding=1, dropout=0.1)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=32)  # output channel is 2 (binary)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # this makes size half

    def forward(self, x):
        # (3, 128, 128)
        x = self.conv1(x)
        # (256, 128, 128)
        x = self.pool(x)
        # (256, 64, 64)

        x = self.conv2(x)
        # (128, 64, 64)
        x = self.conv3(x)
        # (64, 64, 64)
        x = self.pool(x)
        # (64, 32, 32)

        x = self.conv4(x)
        # (2, 1, 1)
        return x
