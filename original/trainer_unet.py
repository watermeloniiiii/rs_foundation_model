#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authored by Chenxi
"""

import os
import time
import torch
import wandb
import config
import numpy as np
import torch.optim as optim
import torch.utils as utils
from dataio import UNetDataset
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

def collate_fn(data):
    for i in range(0, len(data)):
        data[i]['label'] = data[i]['label'].sum(axis=0)
    patch = []
    name = []
    image = torch.stack([torch.from_numpy(b['image']) for b in data], 0)
    label = torch.stack([torch.from_numpy(b['label']) for b in data], 0)[:, np.newaxis, :, :]
    patch = patch.append(b['patch'] for b in data)
    name = name.append(b['name'] for b in data)
    return {'image': image, 'patch': patch, 'name':name, 'label':label}


class Trainer(object):
    def __init__(self, net):
        self.net = net
        self.opt = config.general['optimizer'].split('_')[0]
        self.root_dir = config.root_dir
        self.vali_dir = config.train_dir
        self.model_dir = config.model_dir
        self.train_dir = config.train_dir
        self.hyperparams = config.hyperparameters

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
        '''
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        '''
        model_folder = self.model_dir
        model_path = os.path.join(model_folder, config.general['model_index']+'_notselected')
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
        train_data = UNetDataset(dir=self.train_dir, transform=None, target=True)
        train_loader = utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)
        vali_data = UNetDataset(dir=self.vali_dir, transform=None)
        vali_loader = utils.data.DataLoader(vali_data, batch_size=bs, shuffle=True, num_workers=1)
        self.net.apply(inplace_relu)
        scheduler_type = config.general['optimizer'].split('_')[1]
        scheduler_param = self.hyperparams[config.general['optimizer']]
        if scheduler_type == 'StepLR':
            scheduler = StepLR(optimizer,
                               step_size= scheduler_param['step_size'] * len(train_loader),
                               gamma=scheduler_param['gamma'])
        if scheduler_type == 'CLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=scheduler_param['base_lr'],
                                 max_lr=scheduler_param['max_lr'],
                                 step_size_up= scheduler_param['step_size'] * len(train_loader))
        train_loss = np.zeros([epoch])
        vali_loss = np.zeros([epoch])
        vali_acc = np.zeros([epoch])
        cm = np.zeros([2, 2])
        for i in range(epoch):
            self.net.train()
            print(get_lr(optimizer))
            for j, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False).type(torch.FloatTensor)
                label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
                weights = torch.zeros([bs, 180, 180])
                test = config.hyperparameters['weight'][1]
                weights[label == 0] = config.hyperparameters['weight'][0]
                weights[label == 1] = config.hyperparameters['weight'][1]
                criterion = torch.nn.BCEWithLogitsLoss(weight=weights.cuda())
                if config.cuda:
                    image = image.cuda()
                    label = label.cuda()
                prediction = self.net(image).squeeze()
                loss = criterion(prediction, label)
                train_loss[i] += loss
                loss.backward()
                optimizer.step()
                if self.hyperparams[config.general['optimizer']] != 'none':
                    scheduler.step()
            train_loss[i] = train_loss[i]/ len(train_loader)
            wandb.log({"train_loss": train_loss[i].item()}, step=i)

            self.net.eval()
            with torch.no_grad():
                for k, sample in enumerate(vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False).type(torch.FloatTensor)
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    if config.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    prediction = self.net(image).squeeze()
                    loss = criterion(prediction, label)
                    vali_loss[i] += loss
                    # vali_acc[i] += accuracy(torch.gt(torch.sigmoid(prediction), 0.5), label)
                    cm += confusion_matrix(label.cpu().detach().data.numpy().flatten(),
                                           torch.gt(torch.sigmoid(prediction), 0.5).cpu().detach().data.numpy().astype(np.int).flatten())
                vali_loss[i] = vali_loss[i] / len(vali_loader)
                vali_acc[i] = vali_acc[i] / len(vali_loader)
                olive_recall = cm[-1, -1] / sum(cm[-1, :])
                olive_precision = cm[-1, -1] / sum(cm[:, -1])
                nonolive_recall = cm[0, 0] / sum(cm[:, 0])
                nonolive_precision = cm[0, 0] / sum(cm[0, :])
                wandb.log({"vali_loss": vali_loss[i].item(),
                           "vali_acc": vali_acc[i],
                           "olive_recall": olive_recall,
                           "olive_precision": olive_precision,
                           "olive_F1": 2*olive_recall*olive_precision /
                                       (olive_recall+olive_precision),
                           "nonolive_recall": nonolive_recall,
                           "nonolive_precision": nonolive_precision,
                           "nonolive_F1": 2 * nonolive_recall * nonolive_precision /
                                          (nonolive_recall + nonolive_precision),
                           'OA': (cm[0,0]+cm[1,1])/sum(cm)
                           }, step=i)
            if i % 10 == 0:
                self.save_model(i)

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
        torch.save(self.net, os.path.join(self.model_path, config.general['model_index'] + 'e_' + str(epoch) + ".pkl"))

    def restore_model(self, dir=None, user_defined=False):
        if not user_defined:
            self.net = torch.load(os.path.join(self.model_path, config.general['model_index'] + ".pkl"))
        if user_defined:
            self.net = torch.load(dir)


    def predict(self, image):
        self.net.eval()
        prediction = self.net(image)
        return prediction
