#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import segmentation_models_pytorch as smp
from model_unet import Unet_3

import config
import trainer_unet
from dataset import UnetDataset

from common.logger import logger

cuda = True  # 是否使用GPU
seed = 11
gpu = 0

torch.manual_seed(seed)


def init_default_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    os.environ["RANK"] = str(rank)
    args.rank = rank
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu_id = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.gpu_id)
    return args


def ddp_setup(args):
    logger.info(
        f"Start initialization for rank {args.rank}, world_size:{args.world_size}, gpu_id:{args.gpu_id}"
    )
    if config.mode == "debug":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12316"
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_url,
    )

    dist.barrier()


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

    if args.rank == 0 and not os.path.exists(config.local_data_dir):
        shutil.copytree(config.data_dir, config.local_data_dir, dirs_exist_ok=True)

    ddp_setup(args)
    if config.general["mode"] == "unet":
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet" if config.encoder_weights is True else None,
            encoder_depth=3,
            decoder_channels=[64, 64, 64],
            in_channels=3,
            classes=1,
        ).to(args.gpu_id)
        # model = Unet_3().to(args.gpu_id)
        total_params_all = sum(p.numel() for p in model.parameters())
        logger.info(f"The total number of parameter of the model is {total_params_all}")
        model = DDP(model, find_unused_parameters=True)
        train_data = UnetDataset(
            filepath_lst=UnetDataset.work_through_folder(
                config.local_data_dir, config.train_regions
            ),
            class_of_interest=config.class_of_interests,
        )
        train_sampler = DistributedSampler(
            train_data, num_replicas=args.world_size, rank=args.rank
        )
        logger.info(
            f"Training: {train_sampler.num_samples} samples were assigned to gpu_id:{args.rank}"
        )
        train_loader = DataLoader(
            train_data,
            batch_size=config.hyperparameters["batch_size"],
            sampler=train_sampler,
        )
        vali_data = UnetDataset(
            filepath_lst=UnetDataset.work_through_folder(
                config.local_data_dir, config.vali_regions
            ),
            class_of_interest=config.class_of_interests,
        )
        vali_sampler = DistributedSampler(
            vali_data, num_replicas=args.world_size, rank=args.rank
        )
        logger.info(
            f"Validation: {vali_sampler.num_samples} samples were assigned to gpu_id:{args.rank}"
        )
        vali_loader = DataLoader(
            vali_data,
            batch_size=config.hyperparameters["batch_size"],
            sampler=vali_sampler,
        )
        trainer = trainer_unet.Trainer(net=model)

        if dist.get_rank() == 0 and config.mode == "run":
            wandb.login()
            config_wandb = config.general
            config_wandb.update(config.hyperparameters)
            wandb.init(
                entity="chenxilin",
                config=config_wandb,
                project="morocco_inditree_unet_paii",
                name=config.general["model_index"],
            )
        trainer.train_model(
            epoch=config.hyperparameters["epochs"],
            train_loader=train_loader,
            vali_loader=vali_loader,
            gpu_id=args.rank,
        )
        dist.destroy_process_group()
