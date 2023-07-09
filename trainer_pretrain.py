#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: huijian
@modified by Chenxi
"""

import os
import time
import torch
import wandb
import config
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from dataio import PretrainDataset
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau, MultiStepLR


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().data.numpy()
    return (pred == target).mean()


class Trainer(object):
    def __init__(self, net):
        self.net = net
        self.root_dir = config.root_dir
        self.vali_dir = config.vali_dir
        self.model_dir = config.model_dir
        self.train_dir = config.train_dir
        self.hyperparams = config.hyperparameters
        self.opt = config.general['optimizer'].split('_')[0]

    def select_optimizer(self):
        optimizer = None
        if (self.opt == 'Adam'):
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                   lr=self.hyperparams[config.general['optimizer']]['lr'],
                                   weight_decay=self.hyperparams['weight_decay'])
        elif (self.opt == 'RMS'):
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.hyperparams[config.general['optimizer']]['lr'],
                                      weight_decay=self.hyperparams['weight_decay'])
        elif (self.opt == 'SGD'):
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                  lr=self.hyperparams[config.general['optimizer']]['lr'],
                                  weight_decay=self.hyperparams['weight_decay'],
                                  momentum=self.hyperparams[config.general['optimizer']]['momentum'])
        elif (self.opt == 'Adagrad'):
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.hyperparams[config.general['optimizer']]['lr'],
                                      weight_decay=self.hyperparams['weight_decay'])
        elif (self.opt == 'Adadelta'):
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.hyperparams[config.general['optimizer']]['lr'],
                                       weight_decay=self.hyperparams['weight_decay'])
        return optimizer

    def makefolders(self):
        model_folder = self.model_dir
        model_path = os.path.join(model_folder, config.general['model_index'] + '_notselected')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path

    def train_model(self, epoch, bs):
        self.net.apply(inplace_relu)
        since = time.time()
        ## make folders to store trained models
        self.makefolders()
        ## set random seed for training
        my_seed = 5568661
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(my_seed)
        optimizer = self.select_optimizer()
        train_data = PretrainDataset(img_dir=self.train_dir, 
                                     target=pd.read_csv(r"/home/jinzn/lin00370/Morocco/indi_tree/pretrain/160/training/label.csv", index_col=0))
        train_loader = utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        vali_data = PretrainDataset(img_dir=self.vali_dir,
                                   target=pd.read_csv(r"/home/jinzn/lin00370/Morocco/indi_tree/pretrain/160/validation/label.csv",index_col=0))
        vali_loader = utils.data.DataLoader(vali_data, batch_size=1, shuffle=True, num_workers=0)
        scheduler_type = config.general['optimizer'].split('_')[1]
        scheduler_param = self.hyperparams[config.general['optimizer']]
        if scheduler_type == 'StepLR':
            scheduler = StepLR(optimizer,
                               step_size=scheduler_param['step_size'] * len(train_loader),
                               gamma=scheduler_param['gamma'])
        if scheduler_type == 'CLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=scheduler_param['base_lr'],
                                 max_lr=scheduler_param['max_lr'],
                                 step_size_up=scheduler_param['step_size'] * len(train_loader))
        train_loss1 = np.zeros([epoch])
        train_loss2 = np.zeros([epoch])
        train_loss3 = np.zeros([epoch])
        vali_loss1 = np.zeros([epoch])
        vali_loss2 = np.zeros([epoch])
        vali_loss3 = np.zeros([epoch])
        for i in range(epoch):
            last_epoch_record = torch.zeros(len(train_loader), 49)
            self.net.train()
            lr = get_lr(optimizer)
            for k, sample in enumerate(train_loader, 0):
                total_loss_training = 0
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False).type(torch.FloatTensor)
                label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
                weights = torch.zeros_like(label)
                weights[label == 0] = config.hyperparameters['weight'][0]
                weights[label == 1] = config.hyperparameters['weight'][1]
                criterion1 = nn.BCELoss(weight=weights.cuda())
                # criterion2 = nn.BCELoss()
                if config.cuda:
                    image = image.cuda()
                    label = label.cuda()
                _, _, last_layer = self.net(image, label)[:3]
                patch_loss1 = criterion1(torch.mean(torch.nn.functional.sigmoid(last_layer), dim=1).squeeze(dim=-1), label)
                total_loss_training += patch_loss1
                patch_loss2 = 0
                for j in range(49):       
                    mask = torch.bitwise_and(torch.abs(label - torch.nn.functional.sigmoid(last_layer[:, j].squeeze(dim=-1))) < config.hyperparameters['threshold'][0],
                                torch.abs(label - torch.nn.functional.sigmoid(last_layer[:, j].squeeze(dim=-1))) > config.hyperparameters['threshold'][1])
                    if mask.type(torch.float32).mean() == 0:
                        continue
                    else:
                        ll_weights = torch.zeros_like(label[mask])
                        ll_weights[label[mask] == 0] = config.hyperparameters['sub_weight'][0]
                        ll_weights[label[mask] == 1] = config.hyperparameters['sub_weight'][1]
                        criterion2 = nn.BCELoss(weight=ll_weights.cuda())
                        patch_loss2 += criterion2(torch.nn.functional.sigmoid(last_layer[mask, j].squeeze(dim=-1)), \
                                        (label-1)[mask]**2) / config.hyperparameters['regularization']
                        # last_epoch_record[k, j] = criterion2(torch.nn.functional.sigmoid(last_layer[mask, j].squeeze(dim=-1)), \
                        #                 (label-1)[mask]**2) / config.hyperparameters['regularization'] / mask.type(torch.float32).sum() 
                total_loss_training += patch_loss2 
                patch_loss3 = torch.var(torch.nn.functional.sigmoid(last_layer))
                total_loss_training -= patch_loss3 
                total_loss_training.backward()
                train_loss1[i] += patch_loss1
                train_loss2[i] += patch_loss2
                train_loss3[i] += patch_loss3
                optimizer.step()
                if self.hyperparams[config.general['optimizer']] != 'none' and lr > 1e-7:
                    scheduler.step()
            train_loss1[i] = train_loss1[i] / len(train_loader)
            train_loss2[i] = train_loss2[i] / len(train_loader)
            train_loss3[i] = train_loss3[i] / len(train_loader)
            wandb.log({"train_loss_patch": train_loss1[i].item(),
                       "train_loss_sub": train_loss2[i],
                       "train_loss_var": train_loss3[i],
                       "train_loss_total": total_loss_training,
                       "learning_rate": lr}, step=i) 

            ######### validation #########           
            self.net.eval()
            pred_arr = []
            label_arr = []
            with torch.no_grad():
                for k, sample in enumerate(vali_loader, 0):
                    total_loss_vali = 0
                    image = Variable(sample["image"], requires_grad=False)[:, :, :, :].type(torch.FloatTensor)
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
                    criterion = nn.BCELoss()
                    if config.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    _, _, last_layer = self.net(image, label)[:3]
                    patch_loss1 = criterion(torch.mean(torch.nn.functional.sigmoid(last_layer), dim=1).squeeze(dim=-1), label)
                    patch_loss2 = 0
                    for j in range(49):
                        mask = torch.bitwise_and(torch.abs(label - torch.nn.functional.sigmoid(last_layer[:, j].squeeze(dim=-1))) < config.hyperparameters['threshold'][0],
                                torch.abs(label - torch.nn.functional.sigmoid(last_layer[:, j].squeeze(dim=-1))) > config.hyperparameters['threshold'][1])
                        if mask.type(torch.float32).mean() == 0:
                            continue
                        else:
                            patch_loss2 += criterion(torch.nn.functional.sigmoid(last_layer[mask, j].squeeze(dim=-1)), \
                                            (label-1)[mask]**2) / config.hyperparameters['regularization']
                    patch_loss3 = torch.var(torch.nn.functional.sigmoid(last_layer))
                    vali_loss1[i] += patch_loss1
                    vali_loss2[i] += patch_loss2
                    vali_loss3[i] += patch_loss3
                    total_loss_vali = patch_loss1 + patch_loss2 - patch_loss3
                    pred_label = torch.gt(torch.mean(torch.nn.functional.sigmoid(last_layer), dim=1).squeeze(dim=-1), 0.5).type(torch.cuda.FloatTensor).item()
                    pred_arr.append(pred_label)
                    label_arr.append(label.item())
                vali_loss1[i] = vali_loss1[i] / len(vali_loader)
                vali_loss2[i] = vali_loss2[i] / len(vali_loader)
                vali_loss3[i] = vali_loss3[i] / len(vali_loader)
                wandb.log({"vali_loss_patch": vali_loss1[i].item(),
                           "vali_loss_sub": vali_loss2[i],
                           "vali_loss_var": vali_loss3[i],
                           "vali_loss_total": total_loss_vali
                           }, step=i)

            # if i % 1 == 0:
            #     cm = confusion_matrix(label_arr, pred_arr)
            #     olive_recall = cm[-1, -1] / sum(cm[-1, :])
            #     olive_precision = cm[-1, -1] / sum(cm[:, -1])
            #     nonolive_recall = cm[0, 0] / sum(cm[:, 0])
            #     nonolive_precision = cm[0, 0] / sum(cm[0, :])
            #     wandb.log({"olive_recall": olive_recall,
            #                "olive_precision": olive_precision,
            #                "olive_F1": 2 * olive_recall * olive_precision /
            #                            (olive_recall + olive_precision),
            #                "nonolive_recall": nonolive_recall,
            #                "nonolive_precision": nonolive_precision,
            #                "nonolive_F1": 2 * nonolive_recall * nonolive_precision /
            #                               (nonolive_recall + nonolive_precision),
            #                'OA': (cm[0, 0] + cm[1, 1]) / sum(cm)
            #                }, step=i)
            
            # if i == 0:
            #     best_acc = 2 * olive_recall * olive_precision / (olive_recall + olive_precision) + \
            #                2 * nonolive_recall * nonolive_precision / (nonolive_recall + nonolive_precision)
            #     if np.isnan(best_acc):
            #         best_acc = 0
            #     self.save_model(i)
            # elif 2 * olive_recall * olive_precision / (olive_recall + olive_precision) + \
            #             2 * nonolive_recall * nonolive_precision / (nonolive_recall + nonolive_precision) > best_acc:
            #     best_acc = 2 * olive_recall * olive_precision / (olive_recall + olive_precision) + \
            #             2 * nonolive_recall * nonolive_precision / (nonolive_recall + nonolive_precision)
            #     self.save_model(i)
            #     print ("the best accuacy is now {}".format(best_acc))
            
            if i % 5 == 0:
                self.save_model(i)

            elapse = time.time() - since
            print(
                "Epoch:{}/{}\n"
                "Time_elapse:{}\n'".format(
                    i + 1, epoch,
                    elapse))

    def save_model(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.model_path, config.general['model_index'] + 'e_' + str(epoch) + ".pkl"))

    def restore_model(self, dir=None, user_defined=False):
        if not user_defined:
            self.net = self.net.load_state_dict(torch.load(os.path.join(self.model_path, config.general['model_index'] + ".pkl")))
        if user_defined:
            self.net = torch.load(dir)

    def predict(self, image):
        self.net.eval()
        prediction = self.net(image)
        return prediction
