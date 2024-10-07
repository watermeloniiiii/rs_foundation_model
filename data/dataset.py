from typing import Iterator, List, Literal, Optional, Union, Tuple
from enum import Enum
import numpy as np
import torch.distributed as dist
import os
import json
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from tqdm import tqdm as tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from torchvision.transforms import v2
from torchvision import transforms
from transformers.image_processing_utils import BaseImageProcessor
from transformers import MaskFormerImageProcessor

from common.geoimage.raster_dataset import RasterDataset
from common.img_utils.img_geom import rotate, flip
from common.logger import logger
from config import config_hf as config
from config.config_hf import STATS_MEAN, STATS_STD

BAND_ORDER = {"tci": 0, "b05": 0, "b06": 0, "b07": 0, "b08": 0, "b11": 0, "b12": 0}
BANDS = {"tci": (480427, 0, 52255), "all": (480329, 0, 52247)}

LANDCOVER = {
    0: "invalid",
    1: "water",
    2: "developed",
    3: "tree",
    4: "shrub",
    5: "grass",
    6: "crop",
    7: "bare",
    8: "snow",
    9: "wetland",
    10: "mangroves",
    11: "moss",
}

LANDCOVER_REVERSED = {
    "invalid": 0,
    "water": 1,
    "developed": 2,
    "tree": 3,
    "shrub": 4,
    "grass": 5,
    "crop": 6,
    "bare": 7,
    "snow": 8,
    "wetland": 9,
    "mangroves": 10,
    "moss": 11,
}


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: BANDS["all"][0],
            _Split.VAL: BANDS["all"][1],
            _Split.TEST: BANDS["all"][2],
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(
        self, actual_index: int, class_id: Optional[str] = None
    ) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class SemanticSegmentationDataset(Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        root: List[str],
        split: "SemanticSegmentationDataset.Split",
        max_process_num: int = 10,
        include_meta: bool = False,
        image_processor: BaseImageProcessor = None,
        label_processor: BaseImageProcessor = None,
        class_of_interest=None,
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
        self.root = root
        with open(
            os.path.join(self.root, split.value + "_segmentation.json"), "r"
        ) as file:
            self.filepath_lst = json.load(file)
        self.process_num = max_process_num
        self._split = split
        self.include_meta = include_meta
        self.image_processor = image_processor
        self.label_processor = label_processor
        self.class_of_interest = class_of_interest

        # if no customized weight is provided, will calculate the weights among different classes
        if not config.customized_weight:
            self.sample_cnt = self._build_crop_cnt_list()
            self.weight_list = [
                i / sum(list(self.sample_cnt.values()))
                for i in list(self.sample_cnt.values())
            ]
            logger.info(
                f"The weights among all classes are {self.weight_list} based on calculation"
            )
        else:
            self.weight_list = [
                i / sum(config.HYPERPARAM["weight"])
                for i in config.HYPERPARAM["weight"]
            ]
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
            matrix[i] = (matrix[i] - mins) / (maxs - mins)
        return matrix

    def _get_mean_and_std(self, filepath, stats, band_names):
        file = RasterDataset.from_file(filepath).data
        for band in range(file.shape[0]):
            band_name = band_names[band]
            if band_name not in stats:
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
        file = np.array(Image.open(filepath[1]))
        for c_val, c in LANDCOVER.items():
            sample_count[c] += (file == c_val).sum()

    def _build_crop_cnt_list(self):
        sample_count = defaultdict(int)
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

    def _str_2_doy(self, date):
        month = int(date[:2])
        day = int(date[2:])
        num_day_per_month = {
            1: 31,
            2: 28,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
        doy = 0
        for m in range(1, month):
            doy += num_day_per_month[m]
        doy += day
        return doy

    def __getitem__(self, index):
        tiles, target = self.filepath_lst[index]
        date = self._str_2_doy(tiles[0].split("/")[-2][-4:])
        tgt = np.array(Image.open(target))[..., None]
        if self.class_of_interest:
            tgt_copy = np.zeros_like(tgt)
            cls_label = 1
            for cois in self.class_of_interest:
                coi_idx = (
                    [LANDCOVER_REVERSED[coi] for coi in cois]
                    if not isinstance(cois, str)
                    else LANDCOVER_REVERSED[cois]
                )
                coi_idx = np.array(coi_idx)
                # Create a mask where tgt is in the coi_idx array
                mask = np.isin(tgt, coi_idx)
                # Set values in tgt to 0 where mask is False
                tgt_copy[mask] = cls_label
                cls_label += 1
            tgt = tgt_copy
        image = self._composite(tiles)
        image_id = tiles[0].split("/")[8] + "_" + tiles[0].split("/")[9]
        if self.image_processor:
            image = self.image_processor(image)["pixel_values"][0]
        if self.label_processor:
            tgt = self.label_processor(tgt)["pixel_values"][0]
        sample = {}
        sample["image"] = image
        sample["label"] = tgt
        sample["image_id"] = image_id
        sample["date"] = date
        return sample

    def _composite(self, dirs):
        res = []
        for dir in dirs:
            img = np.array(Image.open(dir))
            if len(img.shape) == 2:
                img = img[..., None]
            res.append(img)
        return np.concatenate([r for r in res], axis=2)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        filepath_lst: List[str],
        class_of_interest: List[str],
        max_process_num: int = 10,
        data_type: str = "train",
        include_meta: bool = False,
        image_processor: BaseImageProcessor = None,
        label_processor: BaseImageProcessor = None,
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
        self.data_type = data_type
        self.include_meta = include_meta
        self.image_processor = image_processor
        self.label_processor = label_processor

        # for training and validation purposes, will need the filepath for label
        self.filepath_lst_lb = (
            [p.replace("image", "label") for p in self.filepath_lst]
            if data_type != "test"
            else None
        )

        # if no customized weight is provided, will calculate the weights among different classes
        if not config.customized_weight and self.filepath_lst_lb:
            self.sample_cnt = self._build_crop_cnt_list()
            self.weight_list = [
                i / sum(list(self.sample_cnt.values()))
                for i in list(self.sample_cnt.values())
            ]
            logger.info(
                f"The weights among all classes are {self.weight_list} based on calculation"
            )
        else:
            self.weight_list = [
                i / sum(config.hyperparameters["weight"])
                for i in config.hyperparameters["weight"]
            ]
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
            matrix[i] = (matrix[i] - mins) / (maxs - mins)
        return matrix

    @staticmethod
    def work_through_folder(dir, ratio, include=""):
        np.random.seed(1)
        train_fpath_list = []
        for root, _, files in os.walk(dir):
            if files == []:
                continue
            for filename in files:
                if filename.endswith(".tif") and "label" not in root:
                    if np.random.choice(2, 1, p=[ratio, 1 - ratio])[0] != 0:
                        continue
                    if all([e not in root for e in include]):
                        continue
                    train_fpath_list.append(os.path.join(root, filename))
        return train_fpath_list

    def __len__(self):
        return len(self.filepath_lst)

    def __getitem__(self, index):
        image = RasterDataset.from_file(self.filepath_lst[index])
        if self.image_processor:
            image.data = self.image_processor(image.data)["pixel_values"][0]
            image.data = self._scale_percentile_img(image.data)
        else:
            # standardization
            image.data = (image.data - np.array(STATS_MEAN)[:, None, None]) / np.array(
                STATS_STD
            )[:, None, None]
            image.data = self._scale_percentile_img(image.data)
        sample = {}
        sample["image"] = image.data.astype(np.double)
        sample["id"] = self.filepath_lst[index].split("/")[-1].split(".")[0]
        if self.include_meta:
            sample["meta"] = image.meta
        if self.data_type != "test":
            target = RasterDataset.from_file(self.filepath_lst_lb[index])
            if self.label_processor:
                target.data = self.label_processor(target.data)["pixel_values"][0]
            sample["label"] = target.data.astype(np.float32)
        return sample


if __name__ == "__main__":
    import os
    from torch.utils.data.dataloader import DataLoader
    import torch

    train_data_root = "/NAS6/Members/linchenxi/projects/morocco/data/patch"
    dataset = SemanticSegmentationDataset(
        root="/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas",
        split=SemanticSegmentationDataset.Split["TRAIN"],
        class_of_interest=["water"],
    )
