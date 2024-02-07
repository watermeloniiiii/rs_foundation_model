#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:00:06 2018
@author: Chenxi
"""
import config
import torch.nn as nn
from transformers import ViTForImageClassification, AdamW

class pretrainedViT(nn.Module):
    def __init__(self, in_channels=3):
        super(pretrainedViT, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch32-224-in21k',
                                                             num_labels=2)
        # self.vit.base_model.embeddings.patch_embeddings.projection = nn.Sequential(
        #     nn.BatchNorm2d(3),
        #     nn.Conv2d(3, 768 // 2, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(768 // 2),
        #     nn.Conv2d(768 // 2, 768, kernel_size=16, stride=16),
        #     nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.vit(x)
        return x


