import os
import json
import torch
import datetime
import torch.nn as nn
import torch.backends.cudnn
import torch.cuda.comm as comm
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, List
from common.geoimage.raster_dataset import RasterDataset
from dataset import UnetDataset
import config
from model_saving_obj import ModelSavingObject
from torch.autograd import Variable
from torch.nn.functional import softmax


def collate_fn(data):
    sample = {}
    keys = data[0].keys()
    for i in range(len(data)):
        if i == 0:
            for k in keys:
                sample[k] = [data[i][k]]
        else:
            for k in keys:
                sample[k].append(data[i][k])
    sample["image"] = torch.tensor(sample["image"])
    if "label" in keys:
        sample["label"] = torch.tensor(sample["label"])
    return sample


class Inference:
    def __init__(
        self,
        save_folder,
        gpu_ids: Optional[Tuple[int]],
        num_workers=4,
        thread_num=2,
    ) -> None:
        """
        Initialization of the CropRecognition Inference class.
        The result will be saved as tile inference result and save in save_folder path.
        Currently, gpu_ids list must be given

        Parameters
        ----------
        save_folder : str
            Where to save the inference result. The save_folder must exist before initializing
            the inference class
        gpu_ids : Optional[Tuple[int]]
            Which GPU to use, if not given, CPU will be used for inference.
            Currently, only GPU inference is supported. If None is given as input, Exception will raise.
        num_workers : int, optional
            How many workers (processes) to use for building input data, by default 5
        thread_num : int, optional
            How many thread to use for extracting raster images, by default 15

        Raises
        ------
        Exception
            _description_
        """

        if not os.path.exists(save_folder):
            raise Exception(
                "Target folder not found, please create target folder first."
            )
        if isinstance(gpu_ids, list):
            gpu_ids = tuple(gpu_ids)
        self.gpu_ids = gpu_ids
        self.save_folder = save_folder
        # TODO: patch size should be determined in inference func.
        # Basically, for different model and time length, the patch size
        # should be different. An idea is to keep trying different patch size,
        # to fit maximum GPU memory until the best one is selected.
        self.num_workers = num_workers
        self.thread_num = thread_num

    def update_savefolder(self, new_savefolder):
        self.save_folder = new_savefolder

    def inference_model_with_dataset_gpu(self, replicas, dataset: UnetDataset):
        # torch.backends.cudnn.enabled = False
        # For LSTM / RNN inference, enable cudnn will not fully utilize GPU memory.

        dataloader = DataLoader(
            dataset,
            8,
            False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                outputs_lst = []
                image = Variable(sample["image"], requires_grad=False).type(
                    torch.FloatTensor
                )
                meta = sample["meta"]
                filename = sample["id"]
                if config.cuda:
                    image = image.cuda()
                image = comm.scatter(image, devices=self.gpu_ids)
                outputs = nn.parallel.parallel_apply(replicas, image)  # B, C, H, W
                outputs = comm.gather(outputs, destination="cpu")
                outputs = torch.argmax(softmax(outputs), dim=1)
                outputs_lst.extend(zip(outputs, meta, filename))
                for img, meta, filename in outputs_lst:
                    output_rds = RasterDataset.from_ndarray(
                        img.numpy().astype(np.uint8), meta.update(n_bands=1)
                    )
                    output_rds.to_geotiff(os.path.join(save_folder, filename + ".tif"))

    def predict_aoi(
        self,
        model_path: str,
        skip_exists=True,
    ):
        """
        predict the AOI for specific crop type at specific year.
        Output will be uint 16 raster. Number of label value will be 2 +
        len(crop_type_list), where 0 is preserved as Nodata and 1 is preserved
        as negative. The others follow the index of crop_type_list.

        Parameters
        ----------
        model_path : str
            Path the model parameter path, must be given. If model path doesn't
            fit predefined model, exception will raise.

        skip_exists: bool
            Whether to skip the exists files.
        """

        model_meta_obj: ModelSavingObject = torch.load(model_path, map_location="cpu")
        print(f"now inferencing with {model_meta_obj.get_model_name()}")

        model = model_meta_obj.get_model_instance()
        model.to(torch.device("cuda:{}".format(self.gpu_ids[0])))
        model.eval()

        replicas = nn.parallel.replicate(model, self.gpu_ids)
        cur_dataset = UnetDataset(
            filepath_lst=UnetDataset.work_through_folder(
                config.local_data_dir, config.test_regions
            ),
            class_of_interest=config.class_of_interests,
            data_type="test",
        )

        output_lst = self.inference_model_with_dataset_gpu(replicas, cur_dataset)
        for img, meta, filename in output_lst:
            output_rds = RasterDataset.from_ndarray(img, meta.update(n_bands=1))
            output_rds.to_geotiff(os.path.join(save_folder, filename + ".tif"))


if __name__ == "__main__":
    save_folder = f"/NAS6/Members/linchenxi/projects/morocco/inference/{config.name}"
    os.makedirs(save_folder, exist_ok=True)
    model_path = f"/NAS6/Members/linchenxi/projects/morocco/model/{config.name}/{config.name}_best.pth"  # noqa:E501
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    shard_num = 1  # total machine
    cur_idx = 0  # machine index
    print(f"inference with idx {cur_idx} of shard {shard_num}")
    cur_output_folder = os.path.join(save_folder)
    if not os.path.exists(cur_output_folder):
        os.makedirs(cur_output_folder)
    inferencer = Inference(
        cur_output_folder,
        gpu_ids,
    )

    inferencer.predict_aoi(model_path=model_path)
