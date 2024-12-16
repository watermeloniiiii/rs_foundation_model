# @author: Chenxi Lin

from enum import Enum
import numpy as np
import json
import os
import rasterio
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor
from typing import List, Union

NUM_DAY_PER_MONTH = {
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


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 258045,
            _Split.VAL: 110757,
            _Split.TEST: 0,
        }
        return split_lengths[self]


class SEN12MSDataset(Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        root: List[str],
        split: "SEN12MSDataset.Split",  # this is called Forward reference, when we initialize, Split does not exist
        max_process_num: int = 10,
        include_meta: bool = False,
        image_processor: BaseImageProcessor = None,
        label_processor: BaseImageProcessor = None,
    ):
        super().__init__()
        self.root = root
        with open(os.path.join(self.root, split.value + "_all.json"), "r") as file:
            self.filepath_lst = json.load(file)
        self.process_num = max_process_num
        self._split = split
        self.include_meta = include_meta
        self.image_processor = image_processor
        self.label_processor = label_processor

    def __len__(self):
        return len(self.filepath_lst)

    def _str_2_doy(self, date):
        month = int(date[5:7])
        day = int(date[8:])
        doy = 0
        for m in range(1, month):
            doy += NUM_DAY_PER_MONTH[m]
        doy += day
        return doy

    def __getitem__(self, index) -> None:
        s1_tile, s2_tile = self.filepath_lst[index]
        date = self._str_2_doy(s1_tile.split("/")[-1].split("_")[5])
        s1 = rasterio.open(s1_tile).read().transpose(1, 2, 0)
        s2 = rasterio.open(s2_tile).read().astype(np.float32).transpose(1, 2, 0)
        if self.image_processor:
            s1 = self.image_processor[0](s1)["pixel_values"][0]
            s2 = self.image_processor[1](s2)["pixel_values"][0]
        if self.label_processor:
            tgt = np.random.randint(0, 5, (s1.shape[1], s1.shape[2], 1))
            tgt = self.label_processor(tgt)["pixel_values"][0]
            # NOTE as we dont have label for this sen12ms dataset, so I simply use random array with value from 0 to 4 (5 classes, to match with the number of COI defined in the config file)
        sample = {}
        sample["s1"] = s1
        sample["s2"] = s2
        sample["label"] = tgt
        sample["date"] = date
        return sample
