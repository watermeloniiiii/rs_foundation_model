#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chenxi
"""
import os
import numpy as np
import pandas as pd
from skimage import io
import torch.utils as utils
import matplotlib.pyplot as plt

class ViTDataset(utils.data.Dataset):
    def __init__(self, img_dir, vec_dir, transform=None, target=None, scale=True):
        self.img_dir = img_dir
        self.vec_dir = vec_dir
        self.transform = transform
        self.img_list = os.listdir(img_dir)
        self.scale = scale
        self.target = target

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        if self.img_list[index].endswith('tif'):
            image = io.imread(os.path.join(self.img_dir, self.img_list[index])).astype(np.int16)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            if self.scale:
                image = self.scale_percentile_img(image)

            sample = {}
            sample['image'] = image
            sample['patch'] = self.img_list[index].split('.')[0]
            sample['name'] = self.img_list[index]
            # sample['ndvi'] = self.scale_percentile_ts(pd.read_csv(os.path.join(self.ndvi_dir, sample['patch'][:-2]+'.csv')).to_numpy()[:, 1:])
            sample['ndvi'] = self.scale_percentile_ts(pd.read_csv(os.path.join(self.vec_dir, sample['patch']+'.csv')).to_numpy()[:, 1:])
            if self.target is not None:
                if "_" in sample['patch'][:-2]:
                    sample['label'] = self.target.loc[sample['patch'][:-2]]['label']
                else:
                    sample['label'] = self.target.loc[int(sample['patch'][:-2])]['label']

            if self.transform:
                sample = self.transform(sample)
            return sample

    def scale_percentile_img(self, matrix):
        matrix = matrix.transpose(2, 0, 1).astype(np.float)
        d, w, h = matrix.shape
        for i in range(d):
            mins = np.percentile(matrix[i][matrix[i] != 0], 1)
            maxs = np.percentile(matrix[i], 99)
            matrix[i] = matrix[i].clip(mins, maxs)
            matrix[i] = ((matrix[i] - mins) / (maxs - mins))
        return matrix

    def scale_percentile_ts(self, vector):
        vector = vector.astype(np.float)
        d, l = vector.shape
        for i in range(d):
            mins = np.percentile(vector[i][vector[i] != 0], 1)
            maxs = np.percentile(vector[i], 99)
            vector[i] = vector[i].clip(mins, maxs)
            vector[i] = ((vector[i] - mins) / (maxs - mins))
        return vector

class UNetDataset(utils.data.Dataset):
    def __init__(self, dir, transform=None, scale=True, target=True):
        self.dir = dir
        self.transform = transform
        self.img_list = os.listdir(dir)
        self.scale = scale
        self.target = target
        if self.target:
            self.target_list = os.listdir(dir.replace('img', 'target'))

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        if self.img_list[index].endswith('tif'):
            image = io.imread(os.path.join(self.dir, self.img_list[index])).astype(np.int16)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            if self.scale:
                image = self.scale_percentile_n(image)
            sample = {}
            sample['image'] = image
            sample['patch'] = self.img_list[index].split('.')[0]
            sample['name'] = self.img_list[index]
            if self.target:
                target = io.imread(os.path.join(self.dir.replace('img', 'target'), self.target_list[index]))
                sample['label'] = target

            if self.transform:
                sample = self.transform(sample)
            return sample


    def scale_percentile_n(self, matrix):
        matrix = matrix.transpose(2, 0, 1).astype(np.float)
        d, w, h = matrix.shape
        for i in range(d):
            if matrix[i].mean() == 0:
                continue
            mins = np.percentile(matrix[i][matrix[i] != 0], 1)
            maxs = np.percentile(matrix[i], 99)
            matrix[i] = matrix[i].clip(mins, maxs)
            matrix[i] = ((matrix[i] - mins) / (maxs - mins))
        return matrix

class shallowDataset(utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target=None, target_patch=None, scale=True):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = os.listdir(img_dir)
        self.scale = scale
        self.target = target
        self.target_patch = target_patch

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        if self.img_list[index].endswith('tif'):
            image = io.imread(os.path.join(self.img_dir, self.img_list[index])).astype(np.int16)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            if self.scale:
                image = self.scale_percentile_img(image)

            sample = {}
            sample['image'] = image
            sample['name'] = self.img_list[index].split('.')[0][:-2]
            if self.target is not None:
                ## for testing dataset, the image name is like "100_1.tif"
                ## for training and validation dataset, the image name is like "100_100_1.tif"
                if "_" in sample['name']:
                    sample['label'] = self.target.loc[sample['name']]['label']
                else:
                    sample['label'] = self.target.loc[int(sample['name'])]['label']

            if self.transform:
                sample = self.transform(sample)
            return sample

    def scale_percentile_img(self, matrix):
        matrix = matrix.transpose(2, 0, 1).astype(np.float)
        d, w, h = matrix.shape
        for i in range(d):
            if matrix[i].mean() == 0:
                continue
            mins = np.percentile(matrix[i][matrix[i] != 0], 1)
            maxs = np.percentile(matrix[i], 99)
            matrix[i] = matrix[i].clip(mins, maxs)
            matrix[i] = ((matrix[i] - mins) / (maxs - mins))
        return matrix

    def scale_percentile_ts(self, vector):
        vector = vector.astype(np.float)
        d, l = vector.shape
        for i in range(d):
            mins = np.percentile(vector[i][vector[i] != 0], 1)
            maxs = np.percentile(vector[i], 99)
            vector[i] = vector[i].clip(mins, maxs)
            vector[i] = ((vector[i] - mins) / (maxs - mins))
        return vector

def remove_file(dir):
    [os.remove(os.path.join(dir, file)) for file in os.listdir(dir)]

def pretrain_data(in_dir, split_dir, p):
    np.random.seed(11)
    import shutil
    assert isinstance(split_dir, tuple) or isinstance(split_dir, list), "please use tuples or lists"
    assert len(p) == len(split_dir), "please make sure the probabilities correspond to each split directories"
    for d in split_dir:
        if not os.path.exists(d):
            os.makedirs(d)
    for item in os.listdir(in_dir):
        flag = np.random.choice(range(0, len(split_dir)), 1, p)
        for i in range(len(split_dir)):
            if flag == i:
                shutil.copyfile(os.path.join(in_dir, item), os.path.join(split_dir[i], item))

def create_label(in_dir, out_dir):
    if os.path.exists(out_dir):
        os.remove(out_dir)
    df_array = []
    for item in os.listdir(in_dir):
        img_idx = '_'.join(item.split('.')[0].split('_')[:-1])
        img_lb = item.split('.')[0].split('_')[-1]
        df_array.append(np.array([img_idx, img_lb]))
    df = pd.DataFrame(data=df_array, columns=['img', 'label'], index=np.array(df_array)[:, 0])
    df.to_csv(out_dir)

def finetune_data(in_dir, out_dir, label_dir, number):
    ''' generate datasets for the finetune stage

    :parameters
    ----------
    in_dir: str
        the directory containing the raw data
    out_dir: str
        the directory to output the generated data
    label_dir: str
        the directory containing label for all raw data
    number: list[int]
        the number of training samples to generate

    :returns
    ----------
    none
    '''
    np.random.seed(11)
    labels = pd.read_csv(label_dir)
    neg_samples = labels[labels['label'] == 0]['img']
    pos_samples = labels[labels['label'] == 1]['img']
    neg_index = np.random.choice(neg_samples, number[0])
    pos_index = np.random.choice(pos_samples, number[1])
    folder_name = str(sum(number))
    if not os.path.exists(os.path.join(out_dir, folder_name)):
        os.makedirs(os.path.join(out_dir, folder_name))
    import shutil
    for f in np.concatenate([neg_index, pos_index], axis=0):
        f = str(f)
        shutil.copyfile(src=os.path.join(in_dir, f+'_0.tif'),
                        dst=os.path.join(out_dir, folder_name, f+'_0.tif'))








global root_dir
root_dir = r'./'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if __name__ == '__main__':
    # finetune_data(in_dir=r"F:\DigitalAG\morocco\unsupervised\data",
    #               out_dir=r"F:\DigitalAG\morocco\unsupervised\finetune",
    #               label_dir=r"F:\DigitalAG\morocco\unsupervised\label.csv",
    #               number=[5, 5])
    create_label(in_dir=r"F:\DigitalAG\morocco\unet\pretrain\training\visualization",
                 out_dir=r"F:\DigitalAG\morocco\unet\pretrain\training\label.csv")
    # pretrain_data(r"F:\DigitalAG\morocco\unsupervised\new_data",
    #            [r"F:\DigitalAG\morocco\unsupervised\training",
    #             r"F:\DigitalAG\morocco\unsupervised\validation",
    #             r"F:\DigitalAG\morocco\unsupervised\testing"],
    #            p=[0.3, 0.2, 0.5])
