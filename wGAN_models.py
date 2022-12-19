import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, num_classes, sigma=None, kernel=None):
        super(Generator, self).__init__()

        if sigma is not None and kernel is not None:
            self.gaussian_filter = torchvision.transforms.GaussianBlur(kernel, sigma)
        else:
            self.gaussian_filter = None

        self.conv_noise = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )

        self.conv_label = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )

        self.model = nn.Sequential(
            # # input is Z, going into a convolution
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))

    def forward(self, z, label):
        x = self.conv_noise(z)
        y = self.conv_label(label)
        combined_input = torch.cat([x, y], 1)
        if self.gaussian_filter is None:
            return torch.clamp(self.model(combined_input), -1, 1)
        else:
            return torch.clamp(
                self.gaussian_filter(
                    self.model(combined_input)
                ), -1, 1
            )


class CNNCritic(nn.Module):
    def __init__(self, img_shape, nc, ndf, num_classes):
        super(CNNCritic, self).__init__()
        self.img_shape = img_shape

        self.conv_image = nn.Sequential(
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_label = nn.Sequential(
            nn.Conv2d(num_classes, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.LayerNorm([4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, label):
        x = self.conv_image(img)
        y = self.conv_label(label)
        combined_input = torch.cat([x, y], 1)
        return self.model(combined_input)

class FCNCritic(nn.Module):
    def __init__(self, img_shape):
        super(FCNCritic, self).__init__()
        self.image_shape = img_shape
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
            nn.Linear(250, 4) # four nodes of which only the 'conditioned' value is used (see return in forward)
        )

    def forward(self, image, label):
        flattened_label = label[:, :, 0 , 0].argmax(1) # let's try w/o embedding first, thus remove image dimensions
        return self.model(image)[flattened_label]