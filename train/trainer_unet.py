#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authored by Chenxi
"""

import os
import time
import torch
import wandb
import numpy as np
import torch.optim as optim
import torch.utils as utils
from data.dataset import SemanticSegmentationDataset
from torch.autograd import Variable
import torch.nn as nn
from typing import List
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
import deepspeed

# from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torchmetrics.classification import BinaryJaccardIndex
from torch.optim.lr_scheduler import (
    StepLR,
    CyclicLR,
    OneCycleLR,
)
from torch.cuda.amp import GradScaler, autocast
from transformers import MaskFormerImageProcessor, SegformerImageProcessor
from transformers.image_utils import make_list_of_images
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentationOutput,
)

from typing import Optional
from common.logger import logger
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from model_saving_obj import ModelSavingObject

import config.setup as config
from config.setup import (
    PATH,
    HYPERPARAM,
    MODEL_CONFIG,
    SCHEDULER,
    MODEL_TYPE,
    MODEL_NAME,
)

IMAGE_PROCESSOR = {
    "segformer": SegformerImageProcessor(),
    "maskformer": MaskFormerImageProcessor(),
}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


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


def make_cuda_list(data: List):
    data = [d.cuda() for d in data]
    return data


class Trainer(object):
    def __init__(self, net):
        self.net = net

    def _select_optimizer(self):
        optimizer = None
        if HYPERPARAM["optimizer"] == "Adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=1e-5,
                weight_decay=HYPERPARAM.get("weight_decay", 0),
            )
        elif HYPERPARAM["optimizer"] == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=1e-5,
                weight_decay=HYPERPARAM.get("weight_decay", 0),
                momentum=HYPERPARAM.get("momentum", 0),
            )
        elif HYPERPARAM["optimizer"] == "AdamW":
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=1e-5,
                weight_decay=HYPERPARAM.get("weight_decay", 0),
            )
        return optimizer

    def _makefolders(self):
        """
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        """
        model_folder = PATH["model_dir"]
        model_path = os.path.join(model_folder, MODEL_NAME)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(os.path.join(model_path, "latest"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "best"), exist_ok=True)
        self.model_folder = model_folder
        self.model_path = model_path

    def _select_scheduler(self):
        if HYPERPARAM["scheduler"] == "StepLR":
            return StepLR(
                self.optimizer,
                step_size=SCHEDULER["StepLR"]["step_size"] * len(self.train_loader),
                gamma=SCHEDULER["StepLR"]["gamma"],
            )
        elif HYPERPARAM["scheduler"] == "CLR":
            return CyclicLR(
                self.optimizer,
                base_lr=SCHEDULER["CLR"]["base_lr"],
                max_lr=SCHEDULER["CLR"]["max_lr"],
                step_size_up=SCHEDULER["CLR"]["step_size"] * len(self.train_loader),
            )
        elif HYPERPARAM["scheduler"] == "ONECLR":
            return OneCycleLR(
                self.optimizer,
                max_lr=SCHEDULER["ONECLR"]["max_lr"],
                steps_per_epoch=len(self.train_loader),
                pct_start=SCHEDULER["ONECLR"]["pct_start"],
                div_factor=SCHEDULER["ONECLR"]["div_factor"],
                epochs=self.epoch,
            )

    def training(self, epoch):
        if (
            self.accelerator.local_process_index == 0
            and config.mode == "run"
            and not self.login
        ):
            self.login = wandb.login(key="c10829f6b79edeea79554d3c1660588729eec616")
            config_wandb = {}
            config_wandb.update(config.HYPERPARAM)
            wandb.init(
                entity="chenxilin",
                config=config_wandb,
                project=f"morocco_tree_{config.MODEL_TYPE}",
                name=config.MODEL_NAME,
            )
        self.net, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.net, self.optimizer, self.train_loader, self.scheduler
            )
        )
        self.net.train()
        # Initialize the gradient scaler
        for _, sample in enumerate(self.train_loader, 0):
            self.optimizer.zero_grad()
            inputs, input_values = zip(*sample.items())
            tensor_type = (
                torch.cuda.FloatTensor
                if self.accelerator.device.type == "cuda"
                else torch.FloatTensor
            )
            if MODEL_TYPE == "maskformer":
                outputs = self.net(
                    input_values[0].type(tensor_type),
                    input_values[2],
                    input_values[3],
                )
            if MODEL_TYPE == "segformer":
                outputs = self.net(
                    input_values[0].type(tensor_type),
                    input_values[1].squeeze(),
                )
            loss = outputs.loss
            # ----- for debug purpose ----- #
            # outputs = MaskFormerImageProcessor().post_process_semantic_segmentation(
            #     outputs=outputs,
            #     target_sizes=[
            #         (
            #             MODEL_CONFIG["image_size"],
            #             MODEL_CONFIG["image_size"],
            #         )
            #     ]
            #     * input_values[0].shape[0],
            # )  # B, C, H, W
            # outputs = SegformerImageProcessor().post_process_semantic_segmentation(
            #     outputs=outputs, target_sizes=[(512, 512)] * image.shape[0]
            # )
            # IOU_metric = BinaryJaccardIndex().cuda()
            # prediction = torch.stack(outputs, dim=0)
            # label = input_values[-1]
            # prediction, label = accelerator.gather_for_metrics((prediction, label))
            # if len(label.shape) != len(prediction.shape):
            #     label = label.squeeze()
            #     prediction = prediction.squeeze()
            # IOU = IOU_metric(prediction, label)
            # IOU_reverse = IOU_metric((prediction - 1) ** 2, (label - 1) ** 2)
            # IOU_mean = (IOU + IOU_reverse) / 2
            # ----- for debug purpose ----- #
            self.train_loss[epoch] += loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            self.train_loss[epoch] = self.train_loss[epoch] / len(self.train_loader)
            if self.accelerator.local_process_index == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                    },
                    step=self.scheduler.scheduler.last_epoch,
                )
        logger.info(
            f"Epoch: {epoch} ----- Rank: {self.accelerator.local_process_index} ----- Learning_Rate: {get_lr(self.optimizer)}"
        )

    def evaluation(self, epoch):
        self.net.eval()
        # accelerator = Accelerator()
        self.vali_loader = self.accelerator.prepare(self.vali_loader)
        with torch.no_grad():
            for _, sample in enumerate(self.vali_loader, 0):
                inputs, input_values = zip(*sample.items())
                tensor_type = (
                    torch.cuda.FloatTensor
                    if self.accelerator.device.type == "cuda"
                    else torch.FloatTensor
                )
                if MODEL_TYPE == "maskformer":
                    outputs = self.net(
                        input_values[0].type(tensor_type),
                        input_values[2],
                        input_values[3],
                    )
                if MODEL_TYPE == "segformer":
                    outputs = self.net(
                        input_values[0].type(tensor_type),
                        input_values[1].squeeze(),
                    )
                loss = outputs.loss
                self.vali_loss[epoch] += loss
                outputs = IMAGE_PROCESSOR[
                    MODEL_TYPE
                ].post_process_semantic_segmentation(
                    outputs=outputs,
                    target_sizes=[
                        (
                            MODEL_CONFIG["image_size"],
                            MODEL_CONFIG["image_size"],
                        )
                    ]
                    * input_values[0].shape[0],
                )  # B, C, H, W
                IOU_metric = BinaryJaccardIndex().cuda()
                prediction = torch.stack(outputs, dim=0)
                label = input_values[-1]
                prediction, label = self.accelerator.gather_for_metrics(
                    (prediction, label)
                )
                if len(label.shape) != len(prediction.shape):
                    label = label.squeeze()
                    prediction = prediction.squeeze()
                IOU = IOU_metric(prediction, label)
                # IOU_reverse = IOU_metric((prediction - 1) ** 2, (label - 1) ** 2)
                # IOU_mean = (IOU + IOU_reverse) / 2
                # iou_tensor_lst = [
                #     torch.zeros(IOU.shape, dtype=IOU.dtype).to(self.gpu_id)
                #     for _ in range(dist.get_world_size())
                # ]
                # dist.all_gather(iou_tensor_lst, IOU)
                # average_IOU = sum(iou_tensor_lst) / dist.get_world_size()
                if self.accelerator.local_process_index == 0:
                    wandb.log(
                        {
                            "vali_loss": loss.item(),
                            "iou": IOU.item(),
                            "learning_rate": get_lr(self.optimizer),
                        },
                        step=self.scheduler.scheduler.last_epoch,
                    )

                model_instance = self.net
                train_state_dict = {
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.scheduler.state_dict(),
                }

                cur_saving_obj = ModelSavingObject(
                    name=MODEL_NAME,
                    model_instance=model_instance,
                    train_state_dict=train_state_dict,
                )
                cur_loss = self.vali_loss[epoch]
                save_best_flag = False
                if self.best_loss is None or self.best_loss > cur_loss:
                    self.best_loss = cur_loss
                    save_best_flag = True

                self._makefolders()
                if save_best_flag:
                    # self.accelerator.save(
                    #     cur_saving_obj,
                    #     os.path.join(
                    #         PATH["model_dir"], MODEL_NAME, f"{MODEL_NAME}_best.pth"
                    #     ),
                    # )
                    self.net.save_checkpoint(
                        save_dir=os.path.join(PATH["model_dir"], MODEL_NAME, "best"),
                    )
                # self.accelerator.save(
                #     cur_saving_obj,
                #     os.path.join(
                #         PATH["model_dir"], MODEL_NAME, f"{MODEL_NAME}_latest.pth"
                #     ),
                # )
                self.net.save_checkpoint(
                    save_dir=os.path.join(PATH["model_dir"], MODEL_NAME, "best"),
                )

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

        self.epoch = epoch
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.gpu_id = gpu_id
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()
        self.train_loss = np.zeros([epoch])
        self.vali_loss = np.zeros([epoch])
        self.best_loss = None
        self.login = False
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        for i in range(epoch):
            self.training(epoch=i)
            self.evaluation(epoch=i)
