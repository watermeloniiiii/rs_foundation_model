import os
import sys

# Modify sys.path to include the directory containing dataset.py
script_dir = os.path.dirname(__file__)  # Directory of the current script (main.py)
parent_dir = os.path.dirname(script_dir)  # Parent directory where dataset.py is located
sys.path.append(parent_dir)

import collections
import json
import torch
import torch.nn as nn
import torch.backends.cudnn
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from common.geoimage.raster_dataset import RasterDataset, mosaic_raster_datasets
from common.geoimage.scene_meta import ValueInterpretationMeta
from common.logger import logger
from data.satlas import (
    ClassificationDataset,
    SemanticSegmentationDataset,
)
from transformers import (
    SegformerImageProcessor,
)
from models.customized_segmention_model import Dinov2ForSemanticSegmentation
import config.setup as config
import config.pretrained_model_path as MODEL_PATH
from torchmetrics.classification import (
    BinaryJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from torchmetrics.segmentation import MeanIoU

from data.sen1floods11 import Sen1FloodsDataset

IMAGE_PROCESSOR = {
    "dinov2": SegformerImageProcessor,
}


def make_cuda_list(data: List):
    data = [d.cuda() for d in data]
    return data


class Inference:
    def __init__(
        self,
        model_instance,
        model_id,
        config,
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
        self.config = config
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
    ):
        dataloader = DataLoader(
            dataset,
            8,
            False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        metrics = collections.defaultdict(list)
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                inputs, input_values = zip(*sample.items())
                tensor_type = torch.cuda.FloatTensor
                image = input_values[inputs.index("image")].cuda().type(tensor_type)
                label = input_values[inputs.index("label")].cuda().squeeze()
                date = input_values[inputs.index("date")].cuda()
                net = replicas[0]
                tensor_type = torch.cuda.FloatTensor
                outputs = net(image, label, date)
                outputs = SegformerImageProcessor().post_process_semantic_segmentation(
                    outputs=outputs,
                    target_sizes=[
                        (
                            self.config.MODEL.optimization.img_size,
                            self.config.MODEL.optimization.img_size,
                        )
                    ]
                    * input_values[0].shape[0],
                    return_prob=True,
                )  # B, C, H, W
                pred, prob = outputs
                label = input_values[inputs.index("label")]
                image_id = sample["image_id"]
                IOU_metric = MeanIoU(
                    num_classes=len(self.config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                Precision_metric = MulticlassPrecision(
                    num_classes=len(self.config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                Recall_metric = MulticlassRecall(
                    num_classes=len(self.config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                F1_metric = MulticlassF1Score(
                    num_classes=len(self.config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                stacked_pred = torch.stack(pred, dim=0)
                Prec = Precision_metric(
                    stacked_pred, label.squeeze().type(torch.cuda.LongTensor)
                )
                Rec = Recall_metric(
                    stacked_pred, label.squeeze().type(torch.cuda.LongTensor)
                )
                F1 = F1_metric(
                    stacked_pred, label.squeeze().type(torch.cuda.LongTensor)
                )
                metrics["Prec"].append(Prec.mean().cpu().numpy())
                metrics["Rec"].append(Rec.mean().cpu().numpy())
                metrics["F1"].append(F1.mean().cpu().numpy())

                for img, label, img_id in zip(pred, label, image_id):
                    # Image.fromarray(img.cpu().numpy().astype(np.int16)).save(
                    #     os.path.join(self.save_folder, img_id + ".png")
                    # )
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(4, 8))
                    plt.subplot(1, 2, 1)
                    plt.imshow(label.cpu().numpy().squeeze())
                    plt.subplot(1, 2, 2)
                    plt.imshow(img.cpu().numpy().squeeze())
                    plt.savefig(os.path.join(self.save_folder, img_id + ".png"))
        for k, v in metrics.items():
            metrics[k] = [sum(v) / len(v)]

        metrics["IOU"].append(
            (metrics["Prec"][0] * metrics["Rec"][0])
            / (
                metrics["Prec"][0]
                + metrics["Rec"][0]
                - metrics["Prec"][0] * metrics["Rec"][0]
            )
        )
        with open(os.path.join(self.save_folder, "metrics.json"), "w") as json_file:
            json.dump(metrics, json_file, indent=4)

    def retrieve_dataset(self):
        image_processor = SegformerImageProcessor(
            do_resize=False,
            image_mean=self.config.PRETRAIN.statistics_sen1flood.mean,
            image_std=self.config.PRETRAIN.statistics_sen1flood.standard_deviation,
            do_rescale=False,
            size=self.config.MODEL.optimization.img_size,
        )
        label_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            size=self.config.MODEL.optimization.img_size,
        )
        DATASET = Sen1FloodsDataset
        cur_dataset = DATASET(
            root=os.path.join(
                "/NAS3/Members/linchenxi/projects/foundation_model/flood_prediction",
                "val",
            ),
            split=Sen1FloodsDataset.Split["VAL"],
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
            self.apply_segmentation(
                replicas=replicas,
                dataset=cur_dataset,
            )


if __name__ == "__main__":
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        "/NAS3/Members/linchenxi/projects/foundation_model/model/finetune_dinov2_8/config.yaml"
    )
    name = cfg.MODEL_INFO.model_name
    save_folder = f"/NAS3/Members/linchenxi/projects/foundation_model/inference/{name}"
    os.makedirs(os.path.join(save_folder, "result"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "accuracy"), exist_ok=True)
    model_path = f"/NAS3/Members/linchenxi/projects/foundation_model/model/{name}/best"  # noqa:E501
    m = Dinov2ForSemanticSegmentation(cfg=cfg)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(model_path, tag="")
    m.load_state_dict(state_dict, strict=False)
    logger.info(f"now inferencing with {name}")
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    cur_output_folder = os.path.join(save_folder)
    if not os.path.exists(cur_output_folder):
        os.makedirs(cur_output_folder)
    inferencer = Inference(
        model_instance=m,
        model_id=name,
        config=cfg,
        save_folder=cur_output_folder,
        gpu_ids=gpu_ids,
    )
    inferencer.main(return_lst=True, skip_exists=False, evaluate=True)
