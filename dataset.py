from typing import Iterator, List, Literal
import pickle
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool

import datetime
from tqdm import tqdm as tqdm
from torch.utils.data import Dataset

from common.geoimage.raster_dataset import RasterDataset
from common.img_utils.img_geom import rotate, flip
import config as config

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
        """
        super().__init__()

        self.filepath_lst = sorted(filepath_lst)
        self.class_of_interests = class_of_interest
        self.process_num = max_process_num
        # self.sample_cnt = self._build_crop_cnt_list()
        if data_type != "test":
            self.filepath_lst_tgt = [p.replace("image", "label") for p in self.filepath_lst]
        # if not config.customized_weight:
        #     self.weight_list = [i / sum(list(self.sample_cnt.values())) for i in list(self.sample_cnt.values())]
        # else:
        #     self.weight_list = [i / sum(config.weight) for i in config.weight]
        if config.stats:
            self.mean = config.stats["mean"]
            self.std = config.stats["std"]
        else:
            self.mean, self.std = self._get_stats()

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
        pool = ThreadPool(80)
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
        overall_mean = np.mean(stats["mean"])
        overall_std = np.std(stats["mean"])
        print (overall_mean, overall_std)
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
        if self.filepath_lst[index].endswith('tif'):
            image = RasterDataset.from_file(self.filepath_lst[index]).data
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
    
    def get_class_of_interests(self):
        return self.class_of_interests


if __name__ == "__main__":
    import os
    from torch.utils.data.dataloader import DataLoader
    import torch

    train_data_root = "/NAS6/Members/linchenxi/projects/morocco/data/patch"
    exclude_region = ""
    train_fpath_list = []
    for root, dirs, files in os.walk(train_data_root):
        if files == []:
            continue
        for filename in files:
            if filename.endswith(".tif") and "label" not in root:
                if exclude_region not in root:
                    continue
                train_fpath_list.append(os.path.join(root, filename))

    class_of_interests = ["negative", "olive"]

    train_dataset = UnetDataset(
        train_fpath_list, class_of_interests, max_process_num=1
    )

    data_loader = DataLoader(train_dataset)

    for val in data_loader:
        data_x, data_x_mask, data_y, data_y_weight = val
        if torch.sum(data_y) > 1:
            print
        else:
            print
