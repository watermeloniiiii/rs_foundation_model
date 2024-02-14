from typing import Iterator, List, Literal
import numpy as np
import torch.distributed as dist
import os
from multiprocessing.pool import ThreadPool

import datetime
from tqdm import tqdm as tqdm
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import torch

from common.geoimage.raster_dataset import RasterDataset
from common.img_utils.img_geom import rotate, flip
import config as config
from config import (STATS_MEAN, STATS_STD)

class UnetDataset(Dataset):
    def __init__(
        self,
        filepath_lst: List[str],
        class_of_interest: List[str],
        max_process_num: int = 10,
        data_type: str = "train"
    ) -> None:
        """the iterarable dataset used for training

        Parameters
        ----------
        filepath_list: List[str]
            the list contains the file paths of all data
        class_of_interests: List[str]
            the list contains the crop names that will be used, e.g., ["negative", "corn", "soybeans"]
            make sure the crop name is consistent with the training labels
        max_process_num: int,
        data_type: str, by default "train"
        """
        super().__init__()

        self.filepath_lst = sorted(filepath_lst)
        self.class_of_interests = class_of_interest
        self.process_num = max_process_num
        self.sample_cnt = self._build_crop_cnt_list()
        if data_type != "test":
            self.filepath_lst_tgt = [p.replace("image", "label") for p in self.filepath_lst]
        if not config.customized_weight:
            self.weight_list = [i / sum(list(self.sample_cnt.values())) for i in list(self.sample_cnt.values())]
        else:
            self.weight_list = [i / sum(config.weight) for i in config.weight]
        if STATS_MEAN or STATS_STD:
            self.mean = STATS_MEAN
            self.std = STATS_STD
        else:
            self.mean, self.std = self._get_stats()

    def _scale_percentile_img(self, matrix):
        # matrix = matrix.transpose(2, 0, 1).astype(np.float)
        d, w, h = matrix.shape
        for i in range(d):
            mins = np.percentile(matrix[i][matrix[i] != 0], 1)
            maxs = np.percentile(matrix[i], 99)
            matrix[i] = matrix[i].clip(mins, maxs)
            matrix[i] = ((matrix[i] - mins) / (maxs - mins))
        return matrix

    def _get_mean_and_std(self, filepath, stats, band_names):
        file = RasterDataset.from_file(filepath).data
        for band in range(file.shape[0]):
            band_name = band_names[band]
            if  band_name not in stats:
                stats[band_name] = {"mean": [], "std": []}
            stats[band_name]["mean"].append(np.mean(file[band]))
            stats[band_name]["std"].append(np.std(file[band]))
        return stats     

    def _get_stats(self):
        pool = ThreadPool(30)
        pbar = tqdm(total=len(self.filepath_lst))
        band_names = {0: "red", 1: "green", 2: "blue"}
        stats = {}
        def update_pbar(arg=None):
            pbar.update(1)

        for cur_filepath in self.filepath_lst:
            pool.apply_async(
                self._get_mean_and_std,
                args=[cur_filepath, stats, band_names],
                callback=update_pbar,
                error_callback=print,
            )
        pool.close()
        pool.join()
        overall_mean = [np.mean(stats[band]["mean"]) for band in band_names.values()]
        overall_std = [np.mean(stats[band]["std"]) for band in band_names.values()]
        return (overall_mean, overall_std)

    def _get_sample_count(self, filepath, sample_count):
        file = RasterDataset.from_file(filepath.replace("image", "label"))
        file.data[file.data>=len(self.class_of_interests)] == 0
        for c_val, c in enumerate(self.class_of_interests):
            if c not in sample_count:
                sample_count[c] = (file.data==c_val).sum()
            else:
                sample_count[c] += (file.data==c_val).sum()
        
    def _build_crop_cnt_list(self):
        sample_count = {}
        pool = ThreadPool(self.process_num)
        pbar = tqdm(total=len(self.filepath_lst))

        def update_pbar(arg=None):
            pbar.update(1)

        for cur_filepath in self.filepath_lst:
            pool.apply_async(
                self._get_sample_count,
                args=[cur_filepath, sample_count],
                callback=update_pbar,
                error_callback=print,
            )
        pool.close()
        pool.join()

        return sample_count

    def __len__(self):
        return len(self.filepath_lst)

    def __getitem__(self, index):
        image = RasterDataset.from_file(self.filepath_lst[index])
        # standardization
        image.data = (image.data - np.array(STATS_MEAN)[:, None, None]) / np.array(STATS_STD)[:, None, None]
        sample = {}
        sample['image'] = image.data
        sample['id'] = self.filepath_lst[index].split('/')[-1].split(".")[0]
        if self.filepath_lst_tgt:
            target = RasterDataset.from_file(self.filepath_lst_tgt[index])
            sample['mask'] = target.data
        return sample
    
class UnetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "", 
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def work_through_folder(self, dir, exclude=""):
        train_fpath_list = []
        for root, _, files in os.walk(dir):
            if files == []:
                continue
            for filename in files:
                if filename.endswith(".tif") and "label" not in root:
                    if any([e in root for e in exclude]):
                        continue
                    train_fpath_list.append(os.path.join(root, filename))
        return train_fpath_list


    def setup(self, stage: str):
        # self.unet_train = UnetDataset(filepath_lst=self.work_through_folder(self.data_dir, ["region4"]), class_of_interest=["negative", "olive"])
        self.unet_val = UnetDataset(filepath_lst=self.work_through_folder(self.data_dir, ["region1", "region2", "region3"]), class_of_interest=["negative", "olive"])

    def train_dataloader(self):
        return DataLoader(self.unet_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.unet_val, batch_size=self.batch_size)


if __name__ == "__main__":
    import os
    from torch.utils.data.dataloader import DataLoader
    import torch

    train_data_root = "/NAS6/Members/linchenxi/projects/morocco/data/patch"
    test_module = UnetDataModule(data_dir=train_data_root)
