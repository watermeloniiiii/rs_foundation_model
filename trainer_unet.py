#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authored by Chenxi
"""

import os
import time
import torch
import wandb
import config
import numpy as np
import torch.optim as optim
import torch.utils as utils
from dataset import UnetDataset
from torch.autograd import Variable

# from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torchmetrics import Precision, Recall
from torch.optim.lr_scheduler import (
    StepLR,
    CyclicLR,
    ReduceLROnPlateau,
    MultiStepLR,
    OneCycleLR,
)
from typing import Optional
from common.logger import logger
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from model_saving_obj import ModelSavingObject


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def accuracy(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().data.numpy()
    return (pred == target).mean()


def collate_fn(data):
    for i in range(0, len(data)):
        data[i]["label"] = data[i]["label"].sum(axis=0)
    patch = []
    name = []
    image = torch.stack([torch.from_numpy(b["image"]) for b in data], 0)
    label = torch.stack([torch.from_numpy(b["label"]) for b in data], 0)[
        :, np.newaxis, :, :
    ]
    patch = patch.append(b["patch"] for b in data)
    name = name.append(b["name"] for b in data)
    return {"image": image, "patch": patch, "name": name, "label": label}


class Trainer(object):
    def __init__(self, net):
        self.net = net
        self.opt = config.general["optimizer"].split("_")[0]
        self.root_dir = config.root_dir
        self.data_dir = config.data_dir
        self.model_dir = config.model_dir
        self.hyperparams = config.hyperparameters

    def select_optimizer(self):
        optimizer = None
        if self.opt == "Adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
        elif self.opt == "RMS":
            optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
        elif self.opt == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
                momentum=self.hyperparams[config.general["optimizer"]]["momentum"],
            )
        elif self.opt == "Adagrad":
            optimizer = optim.Adagrad(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
        elif self.opt == "Adadelta":
            optimizer = optim.Adadelta(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
        elif self.opt == "AdamW":
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.hyperparams[config.general["optimizer"]]["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
        return optimizer

    def makefolders(self):
        """
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        """
        model_folder = self.model_dir
        model_path = os.path.join(model_folder, config.name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_folder = model_folder
        self.model_path = model_path

    def train_model(
        self,
        epoch: int,
        train_loader: DataLoader,
        vali_loader: Optional[DataLoader] = None,
        gpu_id=None,
    ) -> None:
        """
        The main function to execute model training
        """
        self.makefolders()
        optimizer = self.select_optimizer()
        scheduler_type = config.general["optimizer"].split("_")[1]
        scheduler_param = self.hyperparams[config.general["optimizer"]]
        if scheduler_type == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_param["step_size"] * len(train_loader),
                gamma=scheduler_param["gamma"],
            )
        elif scheduler_type == "CLR":
            scheduler = CyclicLR(
                optimizer,
                base_lr=scheduler_param["base_lr"],
                max_lr=scheduler_param["max_lr"],
                step_size_up=scheduler_param["step_size"] * len(train_loader),
            )
        elif scheduler_type == "ONECLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=scheduler_param["max_lr"],
                steps_per_epoch=(len(train_loader) // 1),
                pct_start=scheduler_param["pct_start"],
                div_factor=scheduler_param["div_factor"],
                epochs=epoch,
            )

        train_loss = np.zeros([epoch])
        vali_loss = np.zeros([epoch])
        best_loss = None
        for i in range(epoch):
            self.net.train()
            for _, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False).type(
                    torch.FloatTensor
                )
                label = Variable(sample["label"], requires_grad=False).type(
                    torch.FloatTensor
                )  # B, C, H, W
                pos_weight = torch.tensor(
                    train_loader.dataset.weight_list[0]
                    / train_loader.dataset.weight_list[1]
                ).type(torch.FloatTensor)
                pos_weight = pos_weight.cuda() if config.cuda else pos_weight
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                if config.cuda:
                    image = image.cuda()
                    label = label.cuda()
                prediction = self.net(image)  # B, C, H, W
                loss = criterion(prediction, label)
                train_loss[i] += loss
                # Initialize precision and recall metrics
                precision_metric = Precision(
                    task="binary",
                    average="none",
                ).cuda()
                recall_metric = Recall(
                    task="binary",
                    average="none",
                ).cuda()
                precision = precision_metric(prediction, label)
                recall = recall_metric(prediction, label)
                loss.backward()
                optimizer.step()
                if self.hyperparams[config.general["optimizer"]] != "none":
                    scheduler.step()
            train_loss[i] = train_loss[i] / len(train_loader)
            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                    },
                    step=scheduler.last_epoch,
                )

            self.net.eval()
            with torch.no_grad():
                for _, sample in enumerate(vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False).type(
                        torch.FloatTensor
                    )
                    label = Variable(sample["label"], requires_grad=False).type(
                        torch.FloatTensor
                    )
                    criterion = torch.nn.BCEWithLogitsLoss()
                    if config.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    prediction = self.net(image)
                    loss = criterion(prediction, label)
                    vali_loss[i] += loss
                    # Initialize precision and recall metrics
                    precision_metric = Precision(
                        task="binary",
                        average="none",
                    ).cuda()
                    recall_metric = Recall(
                        task="binary",
                        average="none",
                    ).cuda()
                    precision = precision_metric(prediction, label)
                    recall = recall_metric(prediction, label)
                    precision_tensor_list = [
                        torch.zeros(precision.shape, dtype=precision.dtype).to(gpu_id)
                        for _ in range(dist.get_world_size())
                    ]
                    recall_tensor_list = [
                        torch.zeros(recall.shape, dtype=recall.dtype).to(gpu_id)
                        for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(precision_tensor_list, precision)
                    dist.all_gather(recall_tensor_list, recall)
                    average_precision = (
                        sum(precision_tensor_list) / dist.get_world_size()
                    )
                    average_recall = sum(recall_tensor_list) / dist.get_world_size()
                    if dist.get_rank() == 0:
                        logger.info(get_lr(optimizer))
                        wandb.log(
                            {
                                "vali_loss": loss.item(),
                                "olive_recall": average_recall.item(),
                                "olive_precision": average_precision.item(),
                                "olive_F1": 2
                                * average_recall.item()
                                * average_precision.item()
                                / (average_recall.item() + average_precision.item()),
                                "learning_rate": get_lr(optimizer),
                            },
                            step=scheduler.last_epoch,
                        )

                    model_instance = self.net.module
                    train_state_dict = {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                    }

                    cur_saving_obj = ModelSavingObject(
                        name=config.name,
                        model_instance=model_instance,
                        train_state_dict=train_state_dict,
                    )
                    cur_loss = vali_loss[i]
                    save_best_flag = False
                    if best_loss is None or best_loss > cur_loss:
                        best_loss = cur_loss
                        save_best_flag = True

                    if save_best_flag:
                        torch.save(
                            cur_saving_obj,
                            os.path.join(
                                self.model_dir, config.name, f"{config.name}_best.pth"
                            ),
                        )
                    torch.save(
                        cur_saving_obj,
                        os.path.join(
                            self.model_dir, config.name, f"{config.name}_latest.pth"
                        ),
                    )
