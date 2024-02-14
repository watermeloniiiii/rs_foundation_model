#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import wandb
import config
import torch.nn as nn
from model_unet import Unet_3
from dataset import UnetDataModule
from torchvision import models
from torchgeo.trainers import SemanticSegmentationTask
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning import LightningModule

cuda = True  #是否使用GPU
seed = 11
gpu=0

torch.manual_seed(seed)
if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     print('-------------------------')
    #     print (torch.backends.cudnn.version())
    #     print (torch.__version__)
    #     if not cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #     else:
    #         print("There are {} CUDA devices".format(torch.cuda.device_count()))
    #         print("Setting torch GPU to {}".format(gpu))
    #         torch.cuda.set_device(gpu)
    #         print("Using device:{} ".format(torch.cuda.current_device()))
    #         torch.cuda.manual_seed(seed)

    if config.general['mode'] == 'unet':
        # model = Unet_3()
        # trainer = trainer_unet.Trainer(net=model)
        # config_wandb = config.general
        # config_wandb.update(config.hyperparameters)
        # wandb.init(entity='chenxilin',
        #             config=config_wandb,
        #             project='morocco_inditree_unet',
        #             name=config.general['model_index']
        #         )
        # trainer.train_model(epoch=config.hyperparameters['epochs'],
        #                     bs=config.hyperparameters['batch_size'])
        task = SemanticSegmentationTask(model="unet",
                                        backbone="resnet50",
                                        num_classes=2)
        trainer = Trainer(accelerator="cpu")
        train_data_root = "/NAS6/Members/linchenxi/projects/morocco/data/patch"
        datamodule = UnetDataModule(data_dir=train_data_root)
        trainer.fit(model=task, datamodule=datamodule)
        

  



