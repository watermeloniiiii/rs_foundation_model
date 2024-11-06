import torch
import torch.nn as nn
import torch.nn.functional as F


class DETR(nn.module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        return
