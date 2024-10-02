#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models.resnet import BasicBlock
import segmentation_models_pytorch as smp
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    MaskFormerConfig,
)
import config.config_hf as config
from config.config_hf import STATS_MEAN, STATS_STD
from ddp_utils import init_default_settings, ddp_setup
import trainer_unet
from dataset import SemanticSegmentationDataset, MaskFormerDataset

from common.logger import logger

cuda = True  # 是否使用GPU
seed = 11
gpu = 0

torch.manual_seed(seed)
DATASET_DICT = {"maskformer": MaskFormerDataset}


if __name__ == "__main__":
    import shutil

    if config.mode == "run":
        args = init_default_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    if config.mode == "debug":
        args = parser.parse_args()
        args.rank = 0
        args.world_size = 1
        args.gpu_id = 0

    if "SLURM_PROCID" not in os.environ:
        if args.rank == 0 and not os.path.exists(config.PATH["local_data_dir"]):
            shutil.copytree(
                config.PATH["data_dir"],
                config.PATH["local_data_dir"],
                dirs_exist_ok=True,
            )
        data_root = config.PATH["local_data_dir"]
    else:
        data_root = config.PATH["data_dir"]

    ddp_setup(args)
    DATASET = DATASET_DICT.get(config.MODEL_TYPE, SemanticSegmentationDataset)
    if config.TASK == "segmentation":
        if config.MODEL_TYPE == "FPN":
            model = smp.FPN(**config.MODEL_CONFIG).to(args.gpu_id)
        if config.MODEL_TYPE == "UNet++":
            model = smp.UnetPlusPlus(**config.MODEL_CONFIG).to(args.gpu_id)
        if config.MODEL_TYPE == "segformer":
            image_processor = SegformerImageProcessor(
                do_resize=True,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
            )
            label_processor = SegformerImageProcessor(
                do_resize=True, do_rescale=False, do_normalize=False
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                config.MODEL_CONFIG["pretrained_path"],
                num_labels=1,
                ignore_mismatched_sizes=True,
            ).to(args.gpu_id)
        if config.MODEL_TYPE == "maskformer":
            image_processor = MaskFormerImageProcessor(
                do_resize=True,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            label_processor = MaskFormerImageProcessor(
                do_resize=True,
                do_rescale=False,
                do_normalize=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            model = MaskFormerForInstanceSegmentation.from_pretrained(
                config.MODEL_CONFIG["pretrained_path"],
                num_labels=2,
                ignore_mismatched_sizes=True,
            ).to(args.gpu_id)

    total_params_all = sum(p.numel() for p in model.parameters())
    logger.info(f"The total number of parameter of the model is {total_params_all}")
    model = DDP(model, find_unused_parameters=True)
    train_data = DATASET(
        filepath_lst=DATASET.work_through_folder(
            data_root,
            config.DATA["train_regions"],
        ),
        class_of_interest=config.class_of_interests,
        image_processor=image_processor,
        label_processor=label_processor,
    )
    train_sampler = DistributedSampler(
        train_data, num_replicas=args.world_size, rank=args.rank
    )
    logger.info(
        f"Training: {train_sampler.num_samples} samples were assigned to gpu_id:{args.rank}"
    )
    collate_fn = DATASET.collate_fn if hasattr(DATASET, "collate_fn") else None
    train_loader = DataLoader(
        train_data,
        batch_size=config.HYPERPARAM["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    vali_data = DATASET(
        filepath_lst=DATASET.work_through_folder(
            data_root,
            config.DATA["vali_regions"],
        ),
        class_of_interest=config.class_of_interests,
        image_processor=image_processor,
        label_processor=label_processor,
    )
    vali_sampler = DistributedSampler(
        vali_data, num_replicas=args.world_size, rank=args.rank
    )
    logger.info(
        f"Validation: {vali_sampler.num_samples} samples were assigned to gpu_id:{args.rank}"
    )
    vali_loader = DataLoader(
        vali_data,
        batch_size=config.HYPERPARAM["batch_size"],
        sampler=vali_sampler,
        collate_fn=collate_fn,
    )
    trainer = trainer_unet.Trainer(net=model)

    if dist.get_rank() == 0 and config.mode == "run":
        wandb.login(key="c10829f6b79edeea79554d3c1660588729eec616")
        config_wandb = {}
        config_wandb.update(config.HYPERPARAM)
        wandb.init(
            entity="chenxilin",
            config=config_wandb,
            project=f"morocco_tree_{config.MODEL_TYPE}",
            name=config.MODEL_NAME,
        )
    trainer.train_model(
        epoch=config.HYPERPARAM["epochs"],
        train_loader=train_loader,
        vali_loader=vali_loader,
        gpu_id=args.rank,
    )
    dist.destroy_process_group()
