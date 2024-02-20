import torch
import config
import torch.nn as nn
from torch.autograd import Variable


def bn_conv_relu(in_channels, out_channels):
    """
    usually features = [64,96]
    """
    conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

    return conv


def std_upconv(in_channels, out_channels, top=False):
    if top:
        upconv = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0),
        )

        return upconv

    upconv = nn.Sequential(
        nn.BatchNorm2d(in_channels * 2),
        nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(in_channels),
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        ),
        nn.ReLU(),
    )
    return upconv


class Unet_3(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet_3, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                config.hyperparameters["hidden_layer"][0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.conv11 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][0],
            out_channels=config.hyperparameters["hidden_layer"][1],
        )
        self.conv12 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][1],
            out_channels=config.hyperparameters["hidden_layer"][1],
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][1],
            out_channels=config.hyperparameters["hidden_layer"][2],
        )
        self.conv22 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][2],
            out_channels=config.hyperparameters["hidden_layer"][2],
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][2],
            out_channels=config.hyperparameters["hidden_layer"][3],
        )
        self.conv32 = bn_conv_relu(
            in_channels=config.hyperparameters["hidden_layer"][3],
            out_channels=config.hyperparameters["hidden_layer"][3],
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(config.hyperparameters["hidden_layer"][3]),
            nn.Conv2d(
                config.hyperparameters["hidden_layer"][3],
                config.hyperparameters["hidden_layer"][3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(config.hyperparameters["hidden_layer"][3]),
            nn.Conv2d(
                config.hyperparameters["hidden_layer"][3],
                config.hyperparameters["hidden_layer"][3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.hyperparameters["hidden_layer"][3],
                config.hyperparameters["hidden_layer"][3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.upconv1 = std_upconv(
            config.hyperparameters["hidden_layer"][3],
            config.hyperparameters["hidden_layer"][2],
        )
        self.upconv2 = std_upconv(
            config.hyperparameters["hidden_layer"][2],
            config.hyperparameters["hidden_layer"][1],
        )
        self.upconv3 = std_upconv(
            config.hyperparameters["hidden_layer"][1],
            config.hyperparameters["hidden_layer"][0],
            top=True,
        )

    def forward(self, x):
        x = self.preconv(x)  # (b,3,h,w) -> (b,64,h,w) [b, 64, 192, 192]
        saved_part = []

        x = self.conv11(x)  # [b, 64, 192, 192]
        x = self.conv12(x)  # [b, 64, 192, 192]
        saved_part.append(x)
        x = self.maxpool1(x)  # [b, 64, 96, 96]
        x = self.conv21(x)  # [b, 64, 96, 96]
        x = self.conv22(x)  # [b, 64, 96, 96]
        saved_part.append(x)
        x = self.maxpool1(x)  # [b, 64, 48, 48]
        x = self.conv31(x)  # [b, 64, 48, 48]
        x = self.conv32(x)  # [b, 64, 48, 48]
        saved_part.append(x)
        x = self.bottom(x)  # [b, 64, 48, 48]

        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1)  # [b, 128, 48, 48]
        x = self.upconv1(x)  # [b, 64, 96, 96]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1)  # [b, 128, 96, 96]
        x = self.upconv2(x)  # [b, 64, 192, 192]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1)  # [b, 128, 192, 192]
        x = self.upconv3(x)  # [b, 1, 192, 192]

        return x
