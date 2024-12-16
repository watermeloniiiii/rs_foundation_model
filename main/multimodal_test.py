#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# third-party libraries
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import SegformerImageProcessor
import sys

# PAII's library
from common.logger import logger

# internal files
from config.setup import default_setup
from data.sen12ms import SEN12MSDataset
import trainer_deepspeed as trainer_deepspeed
from rs_foundation_model.models.customized_segmention_model import (
    Dinov2ForSemanticSegmentation,
)

torch.manual_seed(111)


def model_initialization(config):
    image_processor_s1 = SegformerImageProcessor(
        do_resize=False,
        image_mean=config.PRETRAIN.statistics_sen12ms.mean_s1,
        image_std=config.PRETRAIN.statistics_sen12ms.standard_deviation_s1,
        do_rescale=False,
        size=config.MODEL.optimization.img_size,
    )
    image_processor_s2 = SegformerImageProcessor(
        do_resize=False,
        image_mean=config.PRETRAIN.statistics_sen12ms.mean_s2,
        image_std=config.PRETRAIN.statistics_sen12ms.standard_deviation_s2,
        do_rescale=False,
        size=config.MODEL.optimization.img_size,
    )
    label_processor = SegformerImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        size=config.MODEL.optimization.img_size,
    )
    model = Dinov2ForSemanticSegmentation(cfg=config, img_size=256, patch_size=16)
    return (model, (image_processor_s1, image_processor_s2), label_processor)


def execute():
    config = default_setup("./config/multimodal_test.yaml")
    data_root = config.PATH.data_dir
    model_outdir = config.PATH.model_outdir
    model_name = config.MODEL_INFO.model_name
    os.makedirs(os.path.join(model_outdir, model_name), exist_ok=True)

    # load model parameters

    model, image_processor, label_processor = model_initialization(config)
    if config.MODEL.architecture.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for module in [model.feature_fusion, model.classifier]:
            for param in module.parameters():
                param.requires_grad = True
    total_params_all = sum(p.numel() for p in model.parameters())
    logger.info(f"The total number of parameter of the model is {total_params_all}")

    # initialize HuggingFace accerlerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",  # I use wandb for recording purpose, can change this to other tools
        gradient_accumulation_steps=1,
        project_dir=os.path.join(
            config.PATH["model_outdir"], config.MODEL_INFO.model_name
        ),
    )

    # initialize dataset and dataloader
    DATASET = SEN12MSDataset
    train_data = DATASET(
        root=data_root,
        split=SEN12MSDataset.Split["TRAIN"],
        image_processor=image_processor,
        label_processor=label_processor,
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
        ),  # this sampler means that we only use a portion of all samples for effcient finetune
    )
    vali_data = DATASET(
        root=data_root,
        split=SEN12MSDataset.Split["VAL"],
        image_processor=image_processor,
        label_processor=label_processor,
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

    # initialize trainer -- the main training process
    trainer = trainer_deepspeed.Trainer(net=model, config=config)
    trainer.train_model(
        epoch=config.MODEL.optimization.num_epoch,
        train_loader=train_loader,
        vali_loader=vali_loader,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    execute()
