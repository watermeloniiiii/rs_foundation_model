import argparse
import torch
import torch.nn as nn
from dinov2.eval.setup import setup_and_build_model
from transformers import ViTForImageClassification, ViTConfig


class DINOV2PretrainedModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        args_parser = argparse.ArgumentParser()
        args = args_parser.parse_args()
        args.config_file = "/NAS6/Members/linchenxi/projects/DINOV2/model3/config.yaml"
        args.pretrained_weights = "/NAS6/Members/linchenxi/projects/DINOV2/model3/eval/training_12499/teacher_checkpoint.pth"
        args.output_dir = ""
        args.opts = []
        model, _, config = setup_and_build_model(args)
        self.model = model
        self.config = config

    def forward(self, input):
        return self.model(input, is_training=True)
