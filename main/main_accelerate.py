#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import train.trainer_deepspeed as trainer_deepspeed
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
import segmentation_models_pytorch as smp
import torch

from common.logger import logger

import config.setup as config
from data.dataset import (
    SemanticSegmentationDataset,
    ClassificationDataset,
)
from models.customized_segmention_model import Dinov2ForSemanticSegmentation
from config.setup import (
    MODEL_NAME,
)
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/model_config.yaml")

DATASET_DICT = {
    # "maskformer": MaskFormerDataset,
    # "mask2former": MaskFormerDataset,
    "vit": ClassificationDataset,
}

from transformers import (
    SegformerImageProcessor,
)
import config.setup as config
from config.setup import STATS_MEAN, STATS_STD


def model_initialization():
    if config.MODEL_TYPE == "dinov2":
        image_processor = SegformerImageProcessor(
            do_resize=False,
            image_mean=STATS_MEAN,
            image_std=STATS_STD,
            do_rescale=False,
            size=config.MODEL_CONFIG["image_size"],
        )
        label_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            size=config.MODEL_CONFIG["image_size"],
        )
        model = Dinov2ForSemanticSegmentation(
            "./config/model_config.yaml",
            num_class=len(cfg.MODEL.class_of_interest) + 1,
        )
    return (model, image_processor, label_processor)


def execute():
    import shutil

    DATASET = DATASET_DICT.get(config.MODEL_TYPE, SemanticSegmentationDataset)
    if "SLURM_PROCID" not in os.environ:
        data_root = config.PATH["data_dir"]
    else:
        data_root = config.PATH["data_dir"]

    model, image_processor, label_processor = model_initialization()
    total_params_all = sum(p.numel() for p in model.parameters())
    logger.info(f"The total number of parameter of the model is {total_params_all}")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",
        gradient_accumulation_steps=1,
        project_dir=os.path.join(config.PATH["model_outdir"], MODEL_NAME),
    )

    train_data = DATASET(
        root=data_root,
        split=SemanticSegmentationDataset.Split["TRAIN"],
        image_processor=image_processor,
        label_processor=label_processor,
        class_of_interest=config.class_of_interest,
    )

    collate_fn = DATASET.collate_fn if hasattr(DATASET, "collate_fn") else None
    train_loader = DataLoader(
        train_data,
        batch_size=config.HYPERPARAM["batch_size"],
        collate_fn=collate_fn,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(0, len(train_data), (cfg.MODEL.num_train_samples,))
        ),
    )
    vali_data = DATASET(
        root=data_root,
        split=SemanticSegmentationDataset.Split["TEST"],
        image_processor=image_processor,
        label_processor=label_processor,
        class_of_interest=config.class_of_interest,
    )
    vali_loader = DataLoader(
        vali_data,
        batch_size=config.HYPERPARAM["batch_size"],
        collate_fn=collate_fn,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(0, len(vali_data), (cfg.MODEL.num_vali_samples,))
        ),
    )
    logger.info(f"{len(vali_loader)}")
    trainer = trainer_deepspeed.Trainer(net=model)

    trainer.train_model(
        epoch=config.HYPERPARAM["epochs"],
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
