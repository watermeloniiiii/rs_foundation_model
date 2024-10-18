import os
import sys

# Modify sys.path to include the directory containing dataset.py
script_dir = os.path.dirname(__file__)  # Directory of the current script (main.py)
parent_dir = os.path.dirname(script_dir)  # Parent directory where dataset.py is located
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.backends.cudnn
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, List
from common.geoimage.raster_dataset import RasterDataset, mosaic_raster_datasets
from common.geoimage.scene_meta import ValueInterpretationMeta
from common.logger import logger
from data.dataset import (
    ClassificationDataset,
    SemanticSegmentationDataset,
)
import segmentation_models_pytorch as smp
from transformers import (
    SegformerImageProcessor,
    MaskFormerImageProcessor,
    ViTImageProcessor,
    SegformerForSemanticSegmentation,
    ViTForImageClassification,
)
from models.customized_segmention_model import Dinov2ForSemanticSegmentation
import config.setup as config
from config.setup import (
    PATH,
    MODEL_CONFIG,
    STATS_MEAN,
    STATS_STD,
)
import config.pretrained_model_path as MODEL_PATH

DATASET_DICT = {
    "vit": ClassificationDataset,
}
IMAGE_PROCESSOR = {
    "segformer": SegformerImageProcessor,
    "maskformer": MaskFormerImageProcessor,
    "vit": ViTImageProcessor,
    "unet": SegformerImageProcessor,
    "dinov2": SegformerImageProcessor,
}

IMAGE_SZIE = {
    "segformer": 512,
    "maskformer": 512,
    "mask2former": 512,
    "unet": 224,
    "vit": 224,
    "dinov2": 512,
}


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
    sample["image"] = np.array(sample["image"])
    sample["image"] = torch.tensor(sample["image"])
    if "label" in keys:
        sample["label"] = np.array(sample["label"])
        sample["label"] = torch.tensor(sample["label"])
    return sample


def make_cuda_list(data: List):
    data = [d.cuda() for d in data]
    return data


class Inference:
    def __init__(
        self,
        model_instance,
        model_id,
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
        self.model = model_instance
        self.id = model_id
        self.gpu_ids = gpu_ids
        self.save_folder = save_folder
        self.num_workers = num_workers
        self.thread_num = thread_num

    def update_savefolder(self, new_savefolder):
        self.save_folder = new_savefolder

    def apply_segmentation(
        self,
        replicas,
        dataset: SemanticSegmentationDataset,
        unique_id: str = "",
        return_lst: bool = False,
    ):
        dataloader = DataLoader(
            dataset,
            8,
            False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
        pred_lst, prob_lst = [], []
        tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                inputs, input_values = zip(*sample.items())
                net = replicas[0]
                tensor_type = torch.cuda.FloatTensor
                if self.id.split("_")[0] in ["maskformer", "mask2former"]:
                    outputs = net(
                        input_values[inputs.index("pixel_values")].type(tensor_type),
                    )
                if self.id.split("_")[0] in ["segformer", "unet", "dinov2"]:
                    outputs = net(
                        input_values[inputs.index("image")].type(tensor_type),
                    )
                if self.id.split("_")[0] not in ["unet"]:
                    outputs = IMAGE_PROCESSOR[
                        self.id.split("_")[0]
                    ]().post_process_semantic_segmentation(
                        outputs=outputs,
                        target_sizes=[
                            (
                                MODEL_CONFIG["image_size"],
                                MODEL_CONFIG["image_size"],
                            )
                        ]
                        * input_values[0].shape[0],
                        return_prob=True,
                    )  # B, C, H, W
                    pred, prob = outputs
                else:
                    prob = torch.nn.functional.sigmoid(outputs).cpu().numpy()
                    pred = (
                        (torch.nn.functional.sigmoid(outputs) > 0.5)
                        .to(torch.long)
                        .cpu()
                        .numpy()
                    )
                label = input_values[inputs.index("label")]
                image_id = sample["image_id"]
                for img, img_id in zip(pred, image_id):
                    Image.fromarray(img.cpu().numpy().astype(np.int16)).save(
                        os.path.join(self.save_folder, img_id + ".png")
                    )

        tp_neg, fp_neg, fn_neg, tn_neg = tn_all, fn_all, fp_all, tp_all
        metric = self.get_metrics(
            torch.tensor(tp_all).type(torch.LongTensor),
            torch.tensor(fp_all).type(torch.LongTensor),
            torch.tensor(fn_all).type(torch.LongTensor),
            torch.tensor(tn_all).type(torch.LongTensor),
            ["iou_score", "f1_score", "precision", "recall"],
        )
        metric_neg = self.get_metrics(
            torch.tensor(tp_neg).type(torch.LongTensor),
            torch.tensor(fp_neg).type(torch.LongTensor),
            torch.tensor(fn_neg).type(torch.LongTensor),
            torch.tensor(tn_neg).type(torch.LongTensor),
            ["iou_score", "f1_score", "precision", "recall"],
        )

        return pred_lst, prob_lst, metric, metric_neg

    def apply_classification(
        self,
        replicas,
        dataset: ClassificationDataset,
        unique_id: str = "",
        return_lst: bool = False,
    ):
        dataloader = DataLoader(
            dataset,
            1,
            False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
        pred_lst, prob_lst = [], []
        tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                outputs_lst = []
                inputs, input_values = zip(*sample.items())
                meta = sample["meta"]
                filename = sample["id"]
                net = replicas[0]
                tensor_type = torch.cuda.FloatTensor
                outputs = net(
                    input_values[inputs.index("image")].type(tensor_type),
                )
                pred = (
                    torch.argmax(outputs.all_logits, dim=2)
                    .type(torch.cuda.LongTensor)
                    .cpu()
                )
                prob = torch.nn.functional.sigmoid(outputs.all_logits[:, 1:, 1]).cpu()
                if self.data_type == "train":
                    label = input_values[inputs.index("label")]
                    outputs_lst.extend(zip(prob, pred, label, meta, filename))
                else:
                    outputs_lst.extend(zip(prob, pred, meta, filename))
                for output in outputs_lst:
                    label = None
                    if len(output) == 4:
                        prob, pred, meta, filename = output
                    else:
                        prob, pred, label, meta, filename = output
                    row, col = input_values[inputs.index("image")].shape[-2:]
                    prob_arr = torch.zeros([row, col])
                    pred_arr = torch.zeros([row, col])
                    for index, value in enumerate(zip(pred, prob)):
                        # Calculate the row and column start indices for the current grid
                        row_start = (index // 7) * 32
                        col_start = (index % 7) * 32

                        # Fill the (16x16) grid with the current value
                        pred_arr[
                            row_start : row_start + 32, col_start : col_start + 32
                        ] = value[0]
                        prob_arr[
                            row_start : row_start + 32, col_start : col_start + 32
                        ] = value[1]

                    prob_rds = RasterDataset.from_ndarray(
                        prob_arr.numpy().astype(np.float32),
                        meta.update(
                            n_bands=1,
                            value_interpretations=[
                                ValueInterpretationMeta(scale=1, offset=1, nodata=None)
                            ],
                            n_rows=row,
                            n_cols=col,
                        ),
                    )
                    pred_rds = RasterDataset.from_ndarray(
                        pred_arr.numpy().astype(np.uint8),
                        meta.update(
                            n_bands=1,
                            value_interpretations=[
                                ValueInterpretationMeta(scale=1, offset=1, nodata=None)
                            ],
                            n_rows=row,
                            n_cols=col,
                        ),
                    )
                    if label is not None:
                        tp, fp, fn, tn = smp.metrics.get_stats(
                            torch.argmax(outputs.cls_logits)[None, None, ...].cpu(),
                            label[None, None, ...],
                            mode="binary",
                        )
                        tp_all += tp.item()
                        fp_all += fp.item()
                        fn_all += fn.item()
                        tn_all += tn.item()

                    if return_lst:
                        pred_lst.append(pred_rds)
                        prob_lst.append(prob_rds)
                    else:
                        os.makedirs(
                            os.path.join(save_folder, unique_id, "prob"), exist_ok=True
                        )
                        os.makedirs(
                            os.path.join(save_folder, unique_id, "pred"), exist_ok=True
                        )
                        prob_rds.to_geotiff(
                            os.path.join(
                                save_folder, unique_id, "prob", filename + ".tif"
                            )
                        )
                        pred_rds.to_geotiff(
                            os.path.join(
                                save_folder, unique_id, "pred", filename + ".tif"
                            )
                        )
        tp_neg, fp_neg, fn_neg, tn_neg = tn_all, fn_all, fp_all, tp_all
        metric = self.get_metrics(
            torch.tensor(tp_all).type(torch.LongTensor),
            torch.tensor(fp_all).type(torch.LongTensor),
            torch.tensor(fn_all).type(torch.LongTensor),
            torch.tensor(tn_all).type(torch.LongTensor),
            ["iou_score", "f1_score", "precision", "recall"],
        )
        metric_neg = self.get_metrics(
            torch.tensor(tp_neg).type(torch.LongTensor),
            torch.tensor(fp_neg).type(torch.LongTensor),
            torch.tensor(fn_neg).type(torch.LongTensor),
            torch.tensor(tn_neg).type(torch.LongTensor),
            ["f1_score", "precision", "recall"],
        )

        return pred_lst, prob_lst, metric, metric_neg

    def retrieve_dataset(self):
        image_processor = IMAGE_PROCESSOR[self.id.split("_")[0]](
            do_resize=False,
            image_mean=STATS_MEAN,
            image_std=STATS_STD,
            do_rescale=False,
            size=IMAGE_SZIE[self.id.split("_")[0]],
        )
        label_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            size=config.MODEL_CONFIG["image_size"],
        )
        DATASET = DATASET_DICT.get(self.id.split("_")[0], SemanticSegmentationDataset)
        cur_dataset = DATASET(
            root=config.PATH["data_dir"],
            split=SemanticSegmentationDataset.Split["TRAIN"],
            image_processor=image_processor,
            label_processor=label_processor,
        )
        return cur_dataset

    def makedirs(self):
        os.makedirs(
            os.path.join(self.save_folder, "result", f"{self.id}"),
            exist_ok=True,
        )
        pred_outdir = os.path.join(
            save_folder,
            "result",
            f"{self.id}",
            f"{self.id}_pred.tif",
        )
        prob_outdir = os.path.join(
            save_folder,
            "result",
            f"{self.id}",
            f"{self.id}_prob.tif",
        )
        return pred_outdir, prob_outdir

    def main(
        self,
        task: str = "segmentation",
        skip_exists: bool = True,
        return_lst: bool = False,
        evaluate: bool = True,
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

        self.model.to(torch.device("cuda:{}".format(self.gpu_ids[0])))
        self.model.eval()
        replicas = nn.parallel.replicate(self.model, self.gpu_ids)
        df = None
        csv_outdir = os.path.join(self.save_folder, "accuracy", f"{self.id}.csv")
        if os.path.exists(csv_outdir):
            os.remove(csv_outdir)
        pred_outdir, prob_outdir = self.makedirs()
        if os.path.exists(pred_outdir) and os.path.exists(prob_outdir) and skip_exists:
            logger.info(f"the inference result already existed. Will skip")
        self.data_type = "train" if evaluate else "test"
        cur_dataset = self.retrieve_dataset()
        if task == "segmentation":
            pred_lst, prob_lst, metric_pos, metric_neg = self.apply_segmentation(
                replicas=replicas,
                dataset=cur_dataset,
                return_lst=return_lst,
            )


if __name__ == "__main__":
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        "/NAS6/Members/linchenxi/projects/RS_foundation_model/model/finetune_dinov2_flood_1/config.yaml"
    )
    save_folder = (
        f"/NAS6/Members/linchenxi/projects/RS_foundation_model/inference/{name}"
    )
    os.makedirs(os.path.join(save_folder, "result"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "accuracy"), exist_ok=True)
    model_path = f"/NAS6/Members/linchenxi/projects/RS_foundation_model/model/{name}/best"  # noqa:E501
    m = Dinov2ForSemanticSegmentation()
    state_dict = get_fp32_state_dict_from_zero_checkpoint(model_path, tag="")
    m.load_state_dict(state_dict)
    logger.info(f"now inferencing with {name}")
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    cur_output_folder = os.path.join(save_folder)
    if not os.path.exists(cur_output_folder):
        os.makedirs(cur_output_folder)
    inferencer = Inference(
        model_instance=m,
        model_id=name,
        save_folder=cur_output_folder,
        gpu_ids=gpu_ids,
    )
    inferencer.main(task=task, return_lst=True, skip_exists=False, evaluate=True)
