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
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import config.config_hf as config
from data.flood import Sen1FloodsDataset
from models.customized_segmention_model import (
    Dinov2ForSemanticSegmentation,
    LinearClassifier,
)
from config.config_hf import (
    MODEL_NAME,
)
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/model_config.yaml")

DATASET_DICT = {
    # "maskformer": MaskFormerDataset,
    # "mask2former": MaskFormerDataset,
    # "vit": ClassificationDataset,
}

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
)
import config.config_hf as config
from config.config_hf import STATS_MEAN_Sen1Flood as STATS_MEAN
from config.config_hf import STATS_STD_Sen1Flood as STATS_STD


def model_initialization():
    if config.TASK == "segmentation":
        if config.MODEL_TYPE == "segformer":
            image_processor = SegformerImageProcessor(
                do_resize=False,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            label_processor = SegformerImageProcessor(
                do_resize=True,
                do_rescale=False,
                do_normalize=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                config.MODEL_CONFIG["pretrained_path"],
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
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
            )
        if config.MODEL_TYPE == "mask2former":
            image_processor = Mask2FormerImageProcessor(
                do_resize=True,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            label_processor = Mask2FormerImageProcessor(
                do_resize=True,
                do_rescale=False,
                do_normalize=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                config.MODEL_CONFIG["pretrained_path"],
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        if config.MODEL_TYPE == "unet":
            image_processor = SegformerImageProcessor(
                do_resize=True,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            label_processor = SegformerImageProcessor(
                do_resize=True,
                do_rescale=False,
                do_normalize=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            model = smp.Unet(
                encoder_name=config.MODEL_CONFIG["model_version"],
                encoder_depth=5,
                encoder_weights="imagenet",
                classes=config.MODEL_CONFIG["num_classes"],
            )
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

    if config.TASK == "classification":
        if config.MODEL_TYPE == "vit":
            image_processor = ViTImageProcessor(
                do_resize=True,
                image_mean=STATS_MEAN,
                image_std=STATS_STD,
                do_rescale=False,
                size=config.MODEL_CONFIG["image_size"],
            )
            label_processor = None
            model = ViTForImageClassification.from_pretrained(
                config.MODEL_CONFIG["pretrained_path"],
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
    return (model, image_processor, label_processor)


def execute():
    import shutil

    DATASET = DATASET_DICT.get(config.MODEL_TYPE, Sen1FloodsDataset)
    if "SLURM_PROCID" not in os.environ:
        data_root = config.PATH["data_dir"]
    else:
        data_root = config.PATH["data_dir"]

    model, image_processor, label_processor = model_initialization()
    if cfg.FINETUNE.weights_finetune:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            cfg.FINETUNE.weights_finetune, tag=""
        )
        model.load_state_dict(state_dict, strict=False)
        model.classifier = LinearClassifier(
            model.hidden_size,
            model.width,
            model.height,
            num_class=len(cfg.MODEL.fine_tune_class) + 1,
        )
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
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
        root=os.path.join(data_root, "train"),
        split=Sen1FloodsDataset.Split["TRAIN"],
        image_processor=image_processor,
        label_processor=label_processor,
    )

    collate_fn = DATASET.collate_fn if hasattr(DATASET, "collate_fn") else None
    train_loader = DataLoader(
        train_data,
        batch_size=config.HYPERPARAM["batch_size"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    vali_data = DATASET(
        root=os.path.join(data_root, "val"),
        split=Sen1FloodsDataset.Split["VAL"],
        image_processor=image_processor,
        label_processor=label_processor,
    )
    vali_loader = DataLoader(
        vali_data,
        batch_size=config.HYPERPARAM["batch_size"],
        collate_fn=collate_fn,
        drop_last=True,
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
