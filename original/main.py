#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import wandb
import config
import torch.nn as nn
import trainer_vit
import trainer_unet
import trainer_shallow
import trainer_pretrained
from model_vit import ViT
from model_unet import Unet_3
from torchvision import models
from model_pretrain import pretrainedViT
from model_shallow import shallowNet_224, shallowNet_128, shallowNet_64, shallowNet_32

cuda = True  #是否使用GPU
seed = 11
gpu=0

torch.manual_seed(seed)
if __name__ == '__main__':
    if torch.cuda.is_available():
        print('-------------------------')
        print (torch.backends.cudnn.version())
        print (torch.__version__)
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            print("There are {} CUDA devices".format(torch.cuda.device_count()))
            print("Setting torch GPU to {}".format(gpu))
            torch.cuda.set_device(gpu)
            print("Using device:{} ".format(torch.cuda.current_device()))
            torch.cuda.manual_seed(seed)

    if config.general['mode'] == 'vit':
        model = ViT(
                image_size=config.hyperparameters['image_size'],
                patch_size=config.hyperparameters['patch_size'],
                num_classes=config.hyperparameters['num_classes'],
                dim=config.hyperparameters['dim'],
                depth=config.hyperparameters['depth'],
                heads=config.hyperparameters['heads'],
                mlp_dim=config.hyperparameters['mlp_dim'],
                dropout=config.hyperparameters['dropout'],
                emb_dropout=config.hyperparameters['emb_dropout'],
                pool=config.hyperparameters['pool']
            )
        if cuda:
            model = model.cuda()
        trainer = trainer_vit.Trainer(net=model)
        config_wandb = config.general
        config_wandb.update(config.hyperparameters)
        wandb.init(entity='chenxilin',
                    config=config_wandb,
                    project='morocco_inditree_vit',
                    name=config.general['model_index']
                )
        trainer.train_model(epoch=config.hyperparameters['epochs'],
                            bs=config.hyperparameters['batch_size'])

    if config.general['mode'] == 'unet':
        model = Unet_3()
        if cuda:
            model = model.cuda()
        trainer = trainer_unet.Trainer(net=model)
        config_wandb = config.general
        config_wandb.update(config.hyperparameters)
        wandb.init(entity='chenxilin',
                    config=config_wandb,
                    project='morocco_inditree_unet',
                    name=config.general['model_index']
                )
        trainer.train_model(epoch=config.hyperparameters['epochs'],
                            bs=config.hyperparameters['batch_size'])

    if config.general['mode'] == 'shallow':
        model = models.resnet18(pretrained=True)
        # model = models.densenet121(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(int(num_ftrs), 1)
        # model = shallowNet_224()
        if cuda:
            model = model.cuda()
        trainer = trainer_shallow.Trainer(net=model)
        config_wandb = config.general
        config_wandb.update(config.hyperparameters)
        wandb.init(entity='chenxilin',
                    config=config_wandb,
                    project='morocco_inditree_shallow',
                    name=config.general['model_index']
                )
        trainer.train_model(epoch=config.hyperparameters['epochs'],
                            bs=config.hyperparameters['batch_size'])

    if config.general['mode'] == 'pretrain':
        model = pretrainedViT()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.vit.classifier.parameters():
            param.requires_grad = True
        if cuda:
            model = model.cuda()
        trainer = trainer_pretrained.Trainer(net=model)
        config_wandb = config.general
        config_wandb.update(config.hyperparameters)
        wandb.init(entity='chenxilin',
                    config=config_wandb,
                    project='morocco_inditree_pretrain',
                    name=config.general['model_index']
                )
        trainer.train_model(epoch=config.hyperparameters['epochs'],
                            bs=config.hyperparameters['batch_size'])



