#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:00:06 2018
@author: Chenxi
"""
import config
import torch.nn as nn

def bn_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """
    usually features = [64,96]
    """
    conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.1),
    )

    return conv

class shallowNet_224(nn.Module):
    def __init__(self, in_channels=3):
        super(shallowNet_224, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, config.hyperparameters['hidden_layer'][0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][0],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.conv12 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.conv22 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.conv32 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][3],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential()
        for l in range(len(config.hyperparameters["fc_layer"]) - 1):
            self.fc.add_module(f'fc_linear_{l}',  nn.Linear(config.hyperparameters['fc_layer'][l],
                                                config.hyperparameters['fc_layer'][l+1]))
            self.fc.add_module(f'fc_lr_{l}', nn.LeakyReLU(0.1))
        self.fc.add_module(f'fc_output', nn.Linear(config.hyperparameters['fc_layer'][-1], 1))


    def forward(self, x):
        x = self.preconv(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.avgpool(x)
        x = x.view(-1, config.hyperparameters['fc_layer'][0])
        x = self.fc(x)
        return x

class shallowNet_128(nn.Module):
    def __init__(self, in_channels=3):
        super(shallowNet_224, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, config.hyperparameters['hidden_layer'][0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][0],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.conv12 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.conv22 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.conv32 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][3],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential()
        for l in range(len(config.hyperparameters["fc_layer"]) - 1):
            self.fc.add_module(f'fc_linear_{l}',  nn.Linear(config.hyperparameters['fc_layer'][l],
                                                config.hyperparameters['fc_layer'][l+1]))
            self.fc.add_module(f'fc_lr_{l}', nn.LeakyReLU(0.1))
        self.fc.add_module(f'fc_output', nn.Linear(config.hyperparameters['fc_layer'][-1], 1))


    def forward(self, x):
        x = self.preconv(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.avgpool(x)
        x = x.view(-1, config.hyperparameters['fc_layer'][0])
        x = self.fc(x)
        return x

class shallowNet_64(nn.Module):
    def __init__(self, in_channels=3):
        super(shallowNet_224, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, config.hyperparameters['hidden_layer'][0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][0],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.conv12 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.conv22 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.conv32 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][3],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential()
        for l in range(len(config.hyperparameters["fc_layer"]) - 1):
            self.fc.add_module(f'fc_linear_{l}',  nn.Linear(config.hyperparameters['fc_layer'][l],
                                                config.hyperparameters['fc_layer'][l+1]))
            self.fc.add_module(f'fc_lr_{l}', nn.LeakyReLU(0.1))
        self.fc.add_module(f'fc_output', nn.Linear(config.hyperparameters['fc_layer'][-1], 1))


    def forward(self, x):
        x = self.preconv(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.avgpool(x)
        x = x.view(-1, config.hyperparameters['fc_layer'][0])
        x = self.fc(x)
        return x

class shallowNet_32(nn.Module):
    def __init__(self, in_channels=3):
        super(shallowNet_224, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, config.hyperparameters['hidden_layer'][0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][0],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.conv12 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][1],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.conv22 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][2],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.conv32 = bn_conv_relu(in_channels=config.hyperparameters['hidden_layer'][3],
                                   out_channels=config.hyperparameters['hidden_layer'][3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential()
        for l in range(len(config.hyperparameters["fc_layer"]) - 1):
            self.fc.add_module(f'fc_linear_{l}',  nn.Linear(config.hyperparameters['fc_layer'][l],
                                                config.hyperparameters['fc_layer'][l+1]))
            self.fc.add_module(f'fc_lr_{l}', nn.LeakyReLU(0.1))
        self.fc.add_module(f'fc_output', nn.Linear(config.hyperparameters['fc_layer'][-1], 1))


    def forward(self, x):
        x = self.preconv(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.avgpool(x)
        x = x.view(-1, config.hyperparameters['fc_layer'][0])
        x = self.fc(x)
        return x