from typing import Iterator, List, Literal, Optional, Union, Tuple
from enum import Enum
import numpy as np
import os
import json
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from tqdm import tqdm as tqdm
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor
import shutil

from common.geoimage.raster_dataset import RasterDataset
from common.img_utils.img_geom import rotate, flip
from common.logger import logger

from config.setup import default_setup

config = default_setup("./config/model_config.yaml")

BAND_ORDER = [3, 2, 1, 4, 5, 6, 7, 10, 11]
LANDCOVER = {0: "Background", 1: "Water"}
BAND_NAME = {
    3: "red",
    2: "green",
    1: "blue",
    4: "rdeg1",
    5: "rdeg2",
    6: "rdeg3",
    7: "nir",
    11: "swir1",
    12: "swir2",
}
BANDS = {"all": (365, 95, 0)}
STATS_MEAN = [
    1189.2240032217355,
    1335.7876259041898,
    1369.555378215939,
    1432.4846319273197,
    2329.1874191028446,
    2776.599471278697,
    2559.5920587145415,
    1986.2344462964788,
    1150.1873175658327,
]
STATS_STD = [
    504.48473658532214,
    409.57012279339966,
    395.4498976071994,
    459.6583462832101,
    640.6401557563074,
    771.8186296160205,
    743.4712046089181,
    695.772232693198,
    522.1633726674359,
]

FLOOD_DATE = {
    "Bolivia": "20180215",
    "Colombia": "20180823",
    "Ghana": "20180919",
    "India": "20160812",
    "Mekong": "20180804",
    "Nigeria": "20180920",
    "Pakistan": "20170628",
    "Paraguay": "20181031",
    "Somalia": "20180505",
    "Spain": "20190918",
    "Sri-Lanka": "20170528",
    "USA": "20190522",
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


class Sen1FloodsDataset(Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        root: List[str],
        split: "Sen1FloodsDataset.Split",
        max_process_num: int = 10,
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
        self.root = root
        self.filepath_lst = [
            os.path.join(root, "img", file)
            for file in os.listdir(os.path.join(root, "img"))
        ]
        self.process_num = max_process_num
        self._split = split
        self.include_meta = include_meta
        self.image_processor = image_processor
        self.label_processor = label_processor
        # if no customized weight is provided, will calculate the weights among different classes
        if not config.MODEL_INFO.class_weight:
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
                i / sum(config.MODEL_INFO.class_weight)
                for i in config.MODEL_INFO.class_weight
            ]
        if STATS_MEAN and STATS_STD:
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
            if band in band_names:
                band_name = band_names[band]
                if band_name not in stats:
                    stats[band_name] = {"mean": [], "std": []}
                stats[band_name]["mean"].append(np.mean(file[band]))
                stats[band_name]["std"].append(np.std(file[band]))
        return stats

    def _get_stats(self):
        pool = ThreadPool(1)
        pbar = tqdm(total=len(self.filepath_lst))
        stats = {}

        def update_pbar(arg=None):
            pbar.update(1)

        for cur_filepath in self.filepath_lst:
            pool.apply_async(
                self._get_mean_and_std,
                args=[cur_filepath, stats, BAND_NAME],
                callback=update_pbar,
                error_callback=print,
            )
        pool.close()
        pool.join()
        overall_mean = [np.mean(stats[band]["mean"]) for band in BAND_NAME.values()]
        overall_std = [np.mean(stats[band]["std"]) for band in BAND_NAME.values()]
        return (overall_mean, overall_std)

    def _get_sample_count(self, filepath, sample_count):
        file = RasterDataset.from_file(
            filepath.replace("img", "label").replace("S2Hand", "LabelHand")
        ).data
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
        img_dir = self.filepath_lst[index]
        image_id = self.filepath_lst[index].split("/")[-1].split(".")[0]
        image = RasterDataset.from_file(img_dir).data[BAND_ORDER].transpose(1, 2, 0)
        date = self._str_2_doy(img_dir.split("/")[-1].split("_")[-1][4:8])
        tgt = (
            RasterDataset.from_file(
                img_dir.replace("img", "label").replace("S2Hand", "LabelHand")
            ).data
        ).transpose(1, 2, 0)
        tgt = np.clip(tgt, 0, 1)
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

    @staticmethod
    def _make_split_dataset(root, metadata):
        for folder in ["train", "val"]:
            for split in ["img", "label"]:
                os.makedirs(os.path.join(root, folder, split), exist_ok=True)
        img_root = os.path.join(root, "img")
        label_root = os.path.join(root, "label")
        img_suffix = "S2Hand"
        label_suffix = "LabelHand"
        np.random.seed(1234)
        for img in os.listdir(img_root):
            location = img.split("_")[0]
            date = FLOOD_DATE[location]
            correspond_label = img.replace(img_suffix, label_suffix)
            train = np.random.choice(2, 1, p=[0.2, 0.8])
            if train:
                shutil.copy(
                    os.path.join(img_root, img),
                    os.path.join(
                        root, "train", "img", img.replace(".tif", "_" + date + ".tif")
                    ),
                )
                shutil.copy(
                    os.path.join(label_root, correspond_label),
                    os.path.join(
                        root,
                        "train",
                        "label",
                        correspond_label.replace(".tif", "_" + date + ".tif"),
                    ),
                ),
            else:
                shutil.copy(
                    os.path.join(img_root, img),
                    os.path.join(
                        root, "val", "img", img.replace(".tif", "_" + date + ".tif")
                    ),
                )
                shutil.copy(
                    os.path.join(label_root, correspond_label),
                    os.path.join(
                        root,
                        "val",
                        "label",
                        correspond_label.replace(".tif", "_" + date + ".tif"),
                    ),
                )


if __name__ == "__main__":
    import os

    root = "/NAS6/Members/linchenxi/projects/RS_foundation_model/flood_prediction/"
    dataset = Sen1FloodsDataset(
        root="/NAS6/Members/linchenxi/projects/RS_foundation_model/flood_prediction/train",
        split=Sen1FloodsDataset.Split["TRAIN"],
    )
