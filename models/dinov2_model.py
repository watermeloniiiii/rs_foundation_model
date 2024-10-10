import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
import dinov2.distributed as distributed
from dinov2.logging_dinov2 import setup_logging
from dinov2.utils import utils
import logging
from typing import Any, Tuple
import torch.backends.cudnn as cudnn
from dinov2.eval.setup import build_model_for_eval, get_autocast_dtype


def default_setup(cfg):
    seed = getattr(cfg, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=cfg.PATH.log_outdir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(cfg)).items()))
    )


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def setup(cfg):
    """
    Create configs and perform basic setups.
    """
    cfg_dino = OmegaConf.load(cfg.PRETRAIN.cfg_dino)
    os.makedirs(cfg.PATH.log_outdir, exist_ok=True)
    os.makedirs(cfg.PATH.model_outdir, exist_ok=True)
    default_setup(cfg)
    write_config(cfg, os.path.join(cfg.PATH.model_outdir, cfg.MODEL_INFO.model_name))
    return cfg_dino


def setup_and_build_model(model_cfg) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    cfg_dino = setup(model_cfg)
    model = build_model_for_eval(cfg_dino, model_cfg.PRETRAIN.weights_dino_pretrain)
    autocast_dtype = get_autocast_dtype(cfg_dino)
    return model, cfg_dino


class DINOV2PretrainedModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model, cfg = setup_and_build_model(cfg)
        self.model = model
        self.config = cfg

    def forward(self, input):
        return self.model(input, is_training=True)
