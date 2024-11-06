#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import train.trainer_deepspeed as trainer_deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from transformers import DetrImageProcessor
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from common.logger import logger
from transformers import DetrForObjectDetection
from data.dataset import SemanticSegmentationDataset
from config.object_detection.setup import default_setup
from transformers import (
    SegformerImageProcessor,
)
import sys
from detr.models.detr import DETR
from detr.datasets.coco import CocoDetection
from PIL import Image
import requests

script_dir = os.path.dirname(__file__)  # Directory of the current script (main.py)
parent_dir = os.path.dirname(script_dir)  # Parent directory where dataset.py is located
sys.path.append(parent_dir)


def model_initialization(config):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return (model, processor)


def execute():
    torch.manual_seed(111)
    config = default_setup("./config/object_detection/object_detection_detr.yaml")
    DATASET = CocoDetection
    data_root = config.PATH.data_dir
    model_outdir = config.PATH.model_outdir
    model_name = config.MODEL_INFO.model_name
    os.makedirs(os.path.join(model_outdir, model_name), exist_ok=True)
    model, processor = model_initialization(config)
    total_params_all = sum(p.numel() for p in model.parameters())
    logger.info(f"The total number of parameter of the model is {total_params_all}")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",
        gradient_accumulation_steps=1,
        project_dir=os.path.join(
            config.PATH["model_outdir"], config.MODEL_INFO.model_name
        ),
    )

    train_data = DATASET(
        img_folder=os.path.join(data_root, "train2017"),
        ann_file=os.path.join(data_root, "annotations/instances_train2017.json"),
        transforms=None,
        return_masks=True,
    )

    collate_fn = DATASET.collate_fn if hasattr(DATASET, "collate_fn") else None
    train_loader = DataLoader(
        train_data,
        batch_size=config.MODEL.optimization.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(
                0, len(train_data), (config.MODEL.optimization.num_train_samples,)
            )
        ),
    )
    vali_data = DATASET(
        img_folder=os.path.join(data_root, "val2017"),
        ann_file=os.path.join(data_root, "annotations/instances_val2017.json"),
        transforms=None,
        return_masks=True,
    )
    vali_loader = DataLoader(
        vali_data,
        batch_size=config.MODEL.optimization.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(
                0, len(vali_data), (config.MODEL.optimization.num_vali_samples,)
            )
        ),
    )
    logger.info(f"{len(vali_loader)}")
    trainer = trainer_deepspeed.Trainer(net=model, config=config)

    trainer.train_model(
        epoch=config.MODEL.optimization.num_epoch,
        train_loader=train_loader,
        vali_loader=vali_loader,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    # import subprocess

    # if config.mode == "debug":
    #     from accelerate import debug_launcher

    #     debug_launcher(function=execute(), num_processes=1)
    # else:

    #     def launch_accelerate():
    #         # Define the command to execute
    #         command = [
    #             "accelerate",
    #             "launch",
    #             "--config_file",
    #             "/NAS6/Members/linchenxi/morocco/accelerate_deepspeed_config.yaml",
    #             # "--main_process_ip",
    #             # "localhost",
    #             # "--main_process_port",
    #             # "0",
    #             "main_accelerate.py",
    #         ]

    #         # Execute the command
    #         subprocess.run(command)

    #     # Call the function to launch accelerate
    #     launch_accelerate()
    execute()
