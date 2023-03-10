import numpy as np
from torch import nn


class SimpleCNN(nn.Module):

    def __init__(self, n_output_nodes=4):
        super(SimpleCNN, self).__init__()

        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([64, 64]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(16 * 7 * 7, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, n_output_nodes),
            nn.ReLU(True)
        )

    def forward(self, x):
        # image dimensions [128, 128]
        x = self.conv_model(x)
        # dimensions after convolution [7,7]

        # flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = self.fc_model(x)
        return x


class SimpleFCN(nn.Module):
    def __init__(self, image_shape, n_output_nodes=4):
        super(SimpleFCN, self).__init__()
        self.image_shape = image_shape
        self.dimensionality = np.prod(self.image_shape)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dimensionality, 250),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(250, 250),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(250, 250),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(250, 250),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(250, n_output_nodes)
        )

    def forward(self, image):
        return self.model(image)
