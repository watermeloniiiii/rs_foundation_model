from argparse import Namespace
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
import dinov2.distributed as distributed
from dinov2.logging_dinov2 import setup_logging
from dinov2.utils import utils
import logging
from typing import Any, List, Optional, Tuple
import torch.backends.cudnn as cudnn
from .dinov2.dinov2.models import build_model_from_cfg


def default_setup(args):
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def setup(args):
    """
    Create configs and perform basic setups.
    """
    model_cfg = OmegaConf.load("./config/model_config")
    cfg = OmegaConf.load(args.config_file)
    os.makedirs(model_cfg.log_outdir, exist_ok=True)
    os.makedirs(model_cfg.model_outdir, exist_ok=True)
    default_setup(args)
    write_config(cfg, args.output_dir)
    return cfg


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype, config


class DINOV2PretrainedModel(nn.Module):
    def __init__(self, cfg, weights) -> None:
        super().__init__()
        args = Namespace()
        args.config_file = cfg
        args.pretrained_weights = weights
        args.output_dir = os.path.join("logs", MODEL_NAME + ".txt")
        args.opts = []
        model, _, config = setup_and_build_model(args)
        self.model = model
        self.config = config

    def forward(self, input):
        return self.model(input, is_training=True)
