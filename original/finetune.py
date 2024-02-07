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
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils as utils
from dataio import BuildingDataset
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
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
    def __init__(self, net, file_dir, finetune_dir, vali_dir, model_dir, cuda=False, hyperparams=None):
        self.file_dir = file_dir
        self.finetune_dir = finetune_dir
        self.vali_dir = vali_dir
        self.model_dir = model_dir
        self.net = net
        self.hyperparams = hyperparams
        self.opt = hyperparams['optimizer']
        self.learn_rate = hyperparams['lr']
        self.cuda = cuda
        self.identifier = hyperparams['model_index']
        self.lr_schedule = hyperparams['lr_scheduler']
        self.weight = hyperparams['weight']
        self.wd = hyperparams['weight_decay']

    def select_optimizer(self):
        optimizer = None
        if (self.opt == 'Adam'):
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                   lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'RMS'):
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'SGD'):
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                  lr=self.learn_rate, momentum=0.9, weight_decay=self.wd)
        elif (self.opt == 'Adagrad'):
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'Adadelta'):
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.learn_rate, weight_decay=self.wd)
        return optimizer

    def makefolders(self):
        '''
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        '''
        model_folder = self.model_dir
        model_path = os.path.join(model_folder, self.identifier+'_notselected')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_folder = model_folder
        self.model_path = model_path

    def train_model(self, epoch, bs):
        torch.backends.cudnn.deterministic = True
        self.makefolders()
        since = time.time()
        optimizer = self.select_optimizer()
        train_data = BuildingDataset(dir=self.finetune_dir, transform=None,
                                     target=pd.read_csv(r"F:\DigitalAG\morocco\unsupervised\label.csv", index_col=0))
        train_loader = utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=8)
        vali_data = BuildingDataset(dir=self.vali_dir, transform=None,
                                    target=pd.read_csv(r"F:\DigitalAG\morocco\unsupervised\label.csv", index_col=0))
        vali_loader = utils.data.DataLoader(vali_data, batch_size=1, shuffle=True, num_workers=8)
        self.net.apply(inplace_relu)
        if self.lr_schedule == 'CLR':
            # scheduler = StepLR(optimizer, step_size=4 * len(train_loader), gamma=0.75)
            # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.hyperparams['lr_scheduler_params'][0],
                                 max_lr=self.hyperparams['lr_scheduler_params'][1],
                                 step_size_up= self.hyperparams['lr_scheduler_params'][2] * len(train_loader))
        train_loss = np.zeros([epoch])
        vali_loss = np.zeros([epoch])
        vali_acc = np.zeros([epoch])
        criterion = torch.nn.BCEWithLogitsLoss()
        for i in range(epoch):
            self.net.train()
            print(get_lr(optimizer))
            for j, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False).type(torch.FloatTensor)
                label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
                # weights = torch.zeros([label.size(0), 1, 128, 128])
                # weights[label == 1] = self.hyperparams['weight']
                # weights[label == 0] = 1
                ## when your positive samples and negative samples have unbalanced size, using weight parameter
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()
                prediction = self.net(image).squeeze()
                loss = criterion(prediction, label)
                train_loss[i] += loss
                loss.backward()
                optimizer.step()
                if self.lr_schedule != 'none':
                    scheduler.step()
            train_loss[i] = train_loss[i]/ len(train_loader)
            wandb.log({"train_loss": train_loss[i].item()}, step=i)
            #
            # self.net.eval()
            # pred_arr = []
            # label_arr = []
            # with torch.no_grad():
            #     for k, sample in enumerate(vali_loader, 0):
            #         image = Variable(sample["image"], requires_grad=False)[:, :, :, :].type(torch.FloatTensor)
            #         label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
            #         if self.cuda:
            #             image = image.cuda()
            #             label = label.cuda()
            #         prediction = self.net(image).squeeze()
            #         loss = criterion(prediction, label)
            #         vali_loss[i] += loss
            #         # vali_acc[i] += accuracy(torch.gt(torch.sigmoid(prediction), 0.5), label)
            #         pred_arr.append(torch.ge(torch.sigmoid(prediction), 0.5).type(torch.cuda.FloatTensor).item())
            #         label_arr.append(label.item())
            #     vali_loss[i] = vali_loss[i] / len(vali_loader)
            #     cm = confusion_matrix(label_arr, pred_arr)
            #     vali_acc[i] = vali_acc[i] / len(vali_loader)
            #     olive_recall = cm[-1, -1] / sum(cm[-1, :])
            #     olive_precision = cm[-1, -1] / sum(cm[:, -1])
            #     nonolive_recall = cm[0, 0] / sum(cm[:, 0])
            #     nonolive_precision = cm[0, 0] / sum(cm[0, :])
            #     wandb.log({"vali_loss": vali_loss[i].item(),
            #                "vali_acc": vali_acc[i],
            #                "olive_recall": olive_recall,
            #                "olive_precision": olive_precision,
            #                "olive_F1": 2*olive_recall*olive_precision /
            #                            (olive_recall+olive_precision),
            #                "nonolive_recall": nonolive_recall,
            #                "nonolive_precision": nonolive_precision,
            #                "nonolive_F1": 2 * nonolive_recall * nonolive_precision /
            #                               (nonolive_recall + nonolive_precision),
            #                'OA': (cm[0,0]+cm[1,1])/sum(cm)
            #                }, step=i)
            # self.save_model(i)
            #
            elapse = time.time() - since
            print(
                "Epoch:{}/{}\n"
                "Train_loss:{}\n"
                "Time_elapse:{}\n'".format(
                i + 1, epoch,
                round(train_loss[i], 5),
                elapse))

            # if i >= 0:
            #     test_data = BuildingDataset(dir=self.test_dir, transform=None, target=False)
            #     test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
            #     precision = np.zeros(len(test_loader))
            #     recall = np.zeros(len(test_loader))
            #     count = 0
            #     for l, sample in enumerate(test_loader, 0):
            #         image = Variable(sample["image"], requires_grad=False)[:, :, :,:]
            #         patch = sample['patch']
            #         if self.cuda:
            #             image = image.cuda()
            #         pred = self.net(image)
            #         balance = 0.9
            #         pred = torch.ge(pred, balance).type(torch.cuda.FloatTensor)
            #         cm, pred_img, target_img = project_to_target(pred, patch, 512, crop=self.hyperparams['crop'], use_cm=True, use_mask=False)
            #         TP = cm[1, 1]
            #         TN = cm[0, 0]
            #         FP = cm[0, 1]
            #         FN = cm[1, 0]
            #         if (pred_img != 0).sum() > 512 ** 2 * 0.1:
            #             precision[l] = TP / (TP + FP)
            #             recall[l] = TP / (TP + FN)
            #             count += 1
            #     wandb.log({"test_pre": precision.sum()/count, "test_rec": recall.sum()/count}, step=i)

    def save_model(self, epoch):
        torch.save(self.net, os.path.join(self.model_path, self.identifier + 'e_' + str(epoch) + ".pkl"))

    def restore_model(self, dir=None, user_defined=False):
        if not user_defined:
            self.net = torch.load(os.path.join(self.model_path, self.identifier + ".pkl"))
        if user_defined:
            self.net = torch.load(dir)


    def predict(self, image):
        self.net.eval()
        prediction = self.net(image)
        return prediction
