#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authored by Chenxi
"""

import os
import time
import math
import torch
import wandb
import numpy as np
import torch.optim as optim
import torch.utils as utils
from dataset import SemanticSegmentationDataset
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from typing import List
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import deepspeed

# from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import MeanIoU
from torch.optim.lr_scheduler import (
    StepLR,
    CyclicLR,
    OneCycleLR,
)
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    MaskFormerImageProcessor,
    SegformerImageProcessor,
    Mask2FormerImageProcessor,
)
from transformers.image_utils import make_list_of_images
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentationOutput,
)

from typing import Optional
from common.logger import logger
from torch.utils.data import DataLoader
import torch.distributed as dist

from customized_segmention_model import Dinov2ForSemanticSegmentation

import warnings

warnings.filterwarnings("ignore")

import config.config_hf as config
from config.config_hf import (
    TASK,
    PATH,
    HYPERPARAM,
    MODEL_CONFIG,
    SCHEDULER,
    MODEL_TYPE,
    MODEL_NAME,
)

IMAGE_PROCESSOR = {
    "segformer": SegformerImageProcessor(),
    "dinov2": SegformerImageProcessor(),
    "maskformer": MaskFormerImageProcessor(),
    "mask2former": Mask2FormerImageProcessor(),
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
        """
        initialize an optimizer from either the definition of deepspeed optimizer or user-defined optimizer
        NOTE that the user-defined optimizer would be prioritized
        """
        user_defined_optimizer = HYPERPARAM["optimizer"] is not None
        deepspeed_defined_optimizer = not (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        assert (
            user_defined_optimizer or deepspeed_defined_optimizer
        ), "Please provide at least one optimizer from either deepspeed-defined or user-defined"

        optimizer = None
        # if user-defined optimizer is available
        if user_defined_optimizer:
            if "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                del self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "optimizer"
                ]
            if HYPERPARAM["optimizer"] == "Adam":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=1e-5,
                    weight_decay=HYPERPARAM.get("weight_decay", 0),
                )
            elif HYPERPARAM["optimizer"] == "SGD":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=1e-4,
                    weight_decay=HYPERPARAM.get("weight_decay", 0),
                    momentum=HYPERPARAM.get("momentum", 0),
                )
            elif HYPERPARAM["optimizer"] == "AdamW":
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=5e-3,
                    weight_decay=HYPERPARAM.get("weight_decay", 0),
                )
        # otherwise if there's no user-defined optimizer but a deepspeed optimizer
        if not user_defined_optimizer and deepspeed_defined_optimizer:
            optimizer_cls = DummyOptim
            lr = self.accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"][
                "params"
            ].get("lr", 1e-5)
            optimizer = optimizer_cls(self.net.parameters(), lr=lr)
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
        # os.makedirs(os.path.join(model_path, "latest"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "best"), exist_ok=True)
        self.model_folder = model_folder
        self.model_path = model_path

    def _select_scheduler(self):
        """
        initialize an optimizer from either the definition of deepspeed optimizer or user-defined optimizer
        NOTE that the user-defined optimizer would be prioritized
        """
        user_defined_scheduler = HYPERPARAM["scheduler"] is not None
        deepspeed_defined_scheduler = not (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        assert (
            user_defined_scheduler or deepspeed_defined_scheduler
        ), "Please provide at least one scheduler from either deepspeed-defined or user-defined"
        if user_defined_scheduler:
            if "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                del self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "scheduler"
                ]
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
                    steps_per_epoch=len(self.train_loader)
                    // self.accelerator.gradient_accumulation_steps,
                    pct_start=SCHEDULER["ONECLR"]["pct_start"],
                    div_factor=SCHEDULER["ONECLR"]["div_factor"],
                    epochs=self.epoch,
                )
        if not user_defined_scheduler and deepspeed_defined_scheduler:
            # remember to revise step-relevant parameters
            deepspeed_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if "warmup_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = (
                    math.ceil(len(self.train_loader) * self.epoch * 0.01)
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            if "total_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["total_num_steps"] = (
                    math.ceil(len(self.train_loader) * self.epoch * 0.2)
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            return DummyScheduler(self.optimizer)


    def _make_patch_labels_for_classification(
        self,
        arr: torch.Tensor,
        img_size: int = 224,
        patch_size: int = 32,
        threshold: float = 0.1,
    ):
        patch_labels = []
        for i in range(0, img_size, patch_size):
            for j in range(0, img_size, patch_size):
                # Extract the 16x16 grid
                grid = arr[:, :, i : i + patch_size, j : j + patch_size]
                patch_labels.append(
                    (grid.mean(axis=[2, 3]) > threshold).type(torch.cuda.LongTensor)
                )
        patch_labels = torch.stack(patch_labels, dim=1)
        return patch_labels

    def training(self, epoch):
        self.net.train()
        total_sample_met = 0
        for idx, sample in enumerate(self.train_loader, 0):
            total_sample_met += (
                self.train_loader.batch_sampler.batch_size * self.num_processes
            )
            with self.accelerator.accumulate(self.net):
                self.cur_step = (
                    (self.scheduler.scheduler.last_batch_iteration + 1)
                    * self.num_processes
                    if "scheduler"
                    in self.accelerator.state.deepspeed_plugin.deepspeed_config
                    else self.scheduler.scheduler.last_epoch
                )
                logger.info(
                    f"Batch: {idx}/{len(self.train_loader)} \
                    ----- Epoch: {epoch} \
                    ----- Rank: {self.accelerator.local_process_index}\
                    ----- Step: {self.cur_step}\
                    ----- lr: {get_lr(self.optimizer)}\
                    ----- sample_process: {self.train_loader.batch_sampler.batch_size * (self.accelerator.local_process_index + 1)}/{self.train_loader.batch_sampler.batch_size * self.num_processes}\
                    ----- sample_total: {total_sample_met}"
                )
                self.optimizer.zero_grad()
                inputs, input_values = zip(*sample.items())
                tensor_type = (
                    torch.cuda.FloatTensor
                    if self.accelerator.device.type == "cuda"
                    else torch.FloatTensor
                )
                if MODEL_TYPE in ["maskformer", "mask2former"]:
                    outputs = self.net(
                        input_values[inputs.index("pixel_values")].type(tensor_type),
                        input_values[inputs.index("mask_labels")],
                        input_values[inputs.index("class_labels")],
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["segformer"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type),
                        input_values[inputs.index("label")].squeeze(),
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["dinov2"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type),
                        input_values[inputs.index("label")].squeeze(),
                        input_values[inputs.index("date")].squeeze(),
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["unet"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type)
                    )
                    labels = input_values[inputs.index("label")].squeeze()
                    pos_weight = torch.tensor(config.HYPERPARAM["weight"][0]).type(
                        torch.FloatTensor
                    )
                    pos_weight = pos_weight.cuda() if config.cuda else pos_weight
                    if config.MODEL_CONFIG["num_classes"] > 1:
                        loss_fct = CrossEntropyLoss(
                            ignore_index=self.config.semantic_loss_ignore_index,
                            pos_weight=pos_weight,
                        )
                        loss = loss_fct(outputs, labels)
                    elif config.MODEL_CONFIG["num_classes"] == 1:
                        loss_fct = BCEWithLogitsLoss(
                            reduction="none", pos_weight=pos_weight
                        )
                        loss = loss_fct(outputs.squeeze(1), labels.float())
                        loss = loss.mean()
                    else:
                        raise ValueError(
                            f"Number of labels should be >=0: {self.config.num_labels}"
                        )
                if MODEL_TYPE == "vit":
                    """
                    1. label is at pixel level, need to convert to image level
                    2. need to plus patch level loss to the total loss
                    """
                    labels = (
                        torch.mean(input_values[inputs.index("label")], dim=(2, 3))
                        > 0.1
                    ).type(torch.cuda.LongTensor)
                    patch_labels = self._make_patch_labels_for_classification(
                        arr=input_values[inputs.index("label")],
                        img_size=224,
                        patch_size=32,
                        threshold=0.05,
                    )
                    outputs = self.net(
                        pixel_values=input_values[inputs.index("image")].type(
                            tensor_type
                        ),
                        labels=labels,
                    )
                    patch_logits = outputs.all_logits[:, 1:, :]
                    pos_weight = torch.tensor(config.HYPERPARAM["weight"][0]).type(
                        torch.FloatTensor
                    )
                    loss_fct = CrossEntropyLoss(
                        weight=torch.tensor([1, pos_weight]).cuda()
                    )
                    patch_loss = loss_fct(
                        patch_logits.reshape(-1, 2),
                        torch.tensor(patch_labels).type(torch.cuda.LongTensor).view(-1),
                    )
                    loss = outputs.loss + patch_loss
                gathered_loss = self.accelerator.gather_for_metrics(loss)
                self.train_loss[epoch] += torch.mean(gathered_loss)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {
                            "train_loss": torch.mean(gathered_loss).item(),
                            "learning_rate": get_lr(self.optimizer),
                        },
                        step=self.cur_step,
                    )

    def evaluation(self, epoch):
        self.net.eval()
        total_sample_met = 0
        with torch.no_grad():
            for idx, sample in enumerate(self.vali_loader, 0):
                total_sample_met += (
                    self.train_loader.batch_sampler.batch_size * self.num_processes
                )
                logger.info(
                    f"Batch: {idx}/{len(self.vali_loader)} \
                    ----- Epoch: {epoch} \
                    ----- Rank: {self.accelerator.local_process_index}\
                    ----- Step: {self.cur_step} \
                    ----- sample_process: {self.train_loader.batch_sampler.batch_size * (self.accelerator.local_process_index + 1)}/{self.train_loader.batch_sampler.batch_size * self.num_processes} \
                    ----- sample_total: {total_sample_met}"
                )
                inputs, input_values = zip(*sample.items())
                tensor_type = (
                    torch.cuda.FloatTensor
                    if self.accelerator.device.type == "cuda"
                    else torch.FloatTensor
                )
                if MODEL_TYPE in ["maskformer", "mask2former"]:
                    outputs = self.net(
                        input_values[inputs.index("pixel_values")].type(tensor_type),
                        input_values[inputs.index("mask_labels")],
                        input_values[inputs.index("class_labels")],
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["segformer"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type),
                        input_values[inputs.index("label")].squeeze(),
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["dinov2"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type),
                        input_values[inputs.index("label")].squeeze(),
                        input_values[inputs.index("date")].squeeze(),
                    )
                    loss = outputs.loss
                if MODEL_TYPE in ["unet"]:
                    outputs = self.net(
                        input_values[inputs.index("image")].type(tensor_type)
                    )
                    labels = input_values[inputs.index("label")].squeeze()
                    pos_weight = torch.tensor(config.HYPERPARAM["weight"][0]).type(
                        torch.FloatTensor
                    )
                    pos_weight = pos_weight.cuda() if config.cuda else pos_weight
                    if config.MODEL_CONFIG["num_classes"] > 1:
                        loss_fct = CrossEntropyLoss(
                            ignore_index=self.config.semantic_loss_ignore_index,
                            pos_weight=pos_weight,
                        )
                        loss = loss_fct(outputs, labels)
                    elif config.MODEL_CONFIG["num_classes"] == 1:
                        loss_fct = BCEWithLogitsLoss(
                            reduction="none", pos_weight=pos_weight
                        )
                        loss = loss_fct(outputs.squeeze(1), labels.float())
                        loss = loss.mean()
                    else:
                        raise ValueError(
                            f"Number of labels should be >=0: {self.config.num_labels}"
                        )
                if MODEL_TYPE == "vit":
                    """
                    1. label is at pixel level, need to convert to image level
                    2. need to plus patch level loss to the total loss
                    """
                    labels = (
                        torch.mean(input_values[inputs.index("label")], dim=(2, 3))
                        > 0.1
                    ).type(torch.cuda.LongTensor)
                    patch_labels = self._make_patch_labels_for_classification(
                        arr=input_values[inputs.index("label")],
                        img_size=224,
                        patch_size=32,
                        threshold=0.05,
                    )
                    outputs = self.net(
                        pixel_values=input_values[inputs.index("image")].type(
                            tensor_type
                        ),
                        labels=labels,
                    )
                    patch_logits = outputs.all_logits[:, 1:, :]
                    pos_weight = torch.tensor(config.HYPERPARAM["weight"][0]).type(
                        torch.FloatTensor
                    )
                    loss_fct = CrossEntropyLoss(
                        weight=torch.tensor([1, pos_weight]).cuda()
                    )
                    patch_loss = loss_fct(
                        patch_logits.reshape(-1, 2),
                        torch.tensor(patch_labels).type(torch.cuda.LongTensor).view(-1),
                    )
                    loss = outputs.loss + patch_loss
                if TASK == "segmentation" and MODEL_TYPE not in ["unet"]:
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
                    prediction = torch.stack(outputs, dim=0)
                elif TASK == "segmentation" and MODEL_TYPE in ["unet", "dinov2"]:
                    prediction = (torch.nn.functional.sigmoid(outputs) > 0.5).to(
                        torch.long
                    )
                else:
                    prediction, _ = torch.argmax(outputs.cls_logits, dim=1).type(
                        torch.cuda.LongTensor
                    ), torch.argmax(outputs.all_logits, dim=1).type(
                        torch.cuda.LongTensor
                    )
                IOU_metric = MeanIoU(num_classes=12).cuda()
                labels = (
                    input_values[inputs.index("label")]
                    if MODEL_TYPE != "vit"
                    else labels
                )
                if len(labels.shape) != len(prediction.shape):
                    labels = labels.squeeze()
                    prediction = prediction.squeeze()
                IOU = IOU_metric(prediction, labels.type(torch.cuda.LongTensor))
                IOU_gathered, gathered_loss = self.accelerator.gather_for_metrics(
                    (IOU, loss)
                )
                self.vali_loss[epoch] += torch.mean(gathered_loss)
                self.metric[epoch] += torch.mean(IOU_gathered)

            if self.accelerator.is_local_main_process:
                self.accelerator.log(
                    {
                        "vali_loss": self.vali_loss[epoch].item()
                        / len(self.vali_loader),
                        "iou": self.metric[epoch].item() / len(self.vali_loader),
                    },
                    step=self.cur_step,
                )

            cur_loss = self.vali_loss[epoch]
            save_best_flag = False
            if self.best_loss is None or self.best_loss > cur_loss:
                self.best_loss = cur_loss
                save_best_flag = True

            self._makefolders()
            if save_best_flag:
                self.net.save_checkpoint(
                    save_dir=os.path.join(PATH["model_dir"], MODEL_NAME), tag="best"
                )

    def train_model(
        self,
        epoch: int,
        train_loader: DataLoader,
        vali_loader: Optional[DataLoader] = None,
        accelerator=None,
        gpu_id=None,
    ) -> None:
        """
        The main function to execute model training
        """

        self.epoch = epoch
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.gpu_id = gpu_id
        self.accelerator = accelerator
        self.num_processes = self.accelerator.state.num_processes
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()
        self.train_loss = np.zeros([epoch])
        self.vali_loss = np.zeros([epoch])
        self.metric = np.zeros([epoch])
        self.best_loss = None
        self.login = False

        self.accelerator.init_trackers(
            project_name=f"DINOv2_segmentation_{config.MODEL_TYPE}",
            config=config.HYPERPARAM,
            init_kwargs={
                "wandb": {
                    "entity": "chenxilin",
                    "name": config.MODEL_NAME,
                }
            },
        )
        (
            self.net,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.vali_loader,
        ) = self.accelerator.prepare(
            self.net,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.vali_loader,
        )
        for i in range(epoch):
            self.training(epoch=i)
            self.evaluation(epoch=i)
        self.accelerator.end_training()
