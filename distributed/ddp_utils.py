import os
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed as dist
from common.logger import logger

import config.setup as config


def init_default_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])  # os.environ["SLURM_LOCALID"]
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    else:
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
