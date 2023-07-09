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
import torch.optim as optim
import torch.utils as utils
from dataio import BuildingDataset
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau, MultiStepLR

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):
    def __init__(self, net, file_dir, train_dir, vali_dir, model_dir, cuda=False, identifier=None, hyperparams=None):
        self.file_dir = file_dir
        self.train_dir = train_dir
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

    def kl_criterion(self, p, q):
        res = torch.sum(p * torch.log(p / q), dim=-1)
        return res

    def get_init_center(self, data):
        ftr_list = []
        loader = utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
        pretrained_AE = torch.load(r"F:\DigitalAG\morocco\unsupervised\model\pretrained.pkl")
        for l, sample in enumerate(loader):
            image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
            if self.cuda:
                image = image.cuda()
            pretrained_ftr, _ = pretrained_AE(image)
            ftr_list.append(pretrained_ftr.cpu().data.numpy().squeeze())
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(ftr_list))
        return kmeans.cluster_centers_

    def train_model(self, epoch, bs):
        torch.backends.cudnn.deterministic = True
        self.makefolders()
        since = time.time()
        optimizer = self.select_optimizer()
        txt = open(os.path.join(self.model_dir, self.identifier + ".txt"), 'w')
        txt.writelines(self.identifier + '\n')
        txt.close()
        train_data = BuildingDataset(dir=self.train_dir, transform=None, target=None)
        train_loader = utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=8)
        vali_data = BuildingDataset(dir=self.vali_dir, transform=None, target=None)
        vali_loader = utils.data.DataLoader(vali_data, batch_size=bs, shuffle=True, num_workers=8)
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
        train_kl_loss = np.zeros([epoch])
        train_rec_loss = np.zeros([epoch])
        vali_loss = np.zeros([epoch])
        vali_kl_loss = np.zeros([epoch])
        vali_rec_loss = np.zeros([epoch])
        # init_center = self.get_init_center(train_data)
        # self.net.updateClusterCenter(np.expand_dims(init_center, 0))
        reconstruction_criterion = torch.nn.MSELoss()
        for i in range(epoch):
            self.net.train()
            print(get_lr(optimizer))
            loss_all_training = 0
            loss_all_vali = 0
            for j, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False).type(torch.FloatTensor)
                label = image
                # weights = torch.zeros([label.size(0), 1, 128, 128])
                # weights[label == 1] = self.hyperparams['weight']
                # weights[label == 0] = 1
                ## when your positive samples and negative samples have unbalanced size, using weight parameter
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()
                vector, prediction = self.net(image)
                reconstruction_loss = reconstruction_criterion(prediction, label)
                q = self.net.getTDistribution(vector)
                p = self.net.getTargetDistribution(q)
                kl_loss = self.kl_criterion(p, q).mean()
                loss = 1.0 * reconstruction_loss + 1.0 * kl_loss
                train_loss[i] += loss
                train_kl_loss[i] += kl_loss
                train_rec_loss[i] += reconstruction_loss
                loss.backward()
                optimizer.step()
                if self.lr_schedule != 'none':
                    scheduler.step()
            train_loss[i] = train_loss[i]/ len(train_loader)
            train_kl_loss[i] = train_kl_loss[i] / len(train_loader)
            train_rec_loss[i] = train_rec_loss[i] / len(train_loader)
            wandb.log({"train_loss": train_loss[i].item(),
                       'train_kl_loss': train_kl_loss[i].item(),
                       'train_rec_loss': train_rec_loss[i].item()}, step=i)

            self.net.eval()
            with torch.no_grad():
                for k, sample in enumerate(vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False)[:, :, :, :].type(torch.FloatTensor)
                    label = image
                    if self.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    vector, prediction = self.net(image)
                    reconstruction_loss = reconstruction_criterion(prediction, label)
                    q = self.net.getTDistribution(vector)
                    p = self.net.getTargetDistribution(q)
                    kl_loss = self.kl_criterion(p, q).mean()
                    loss = 1.0 * reconstruction_loss + 1.0 * kl_loss
                    vali_loss[i] += loss
                    vali_kl_loss[i] += kl_loss
                    vali_rec_loss[i] += reconstruction_loss
                vali_loss[i] = vali_loss[i] / len(vali_loader)
                vali_kl_loss[i] = vali_kl_loss[i] / len(vali_loader)
                vali_rec_loss[i] = vali_rec_loss[i] / len(vali_loader)
                wandb.log({"vali_loss": vali_loss[i].item(),
                           'vali_kl_loss': vali_kl_loss[i].item(),
                           'vali_rec_loss': vali_rec_loss[i].item()}, step=i)
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
